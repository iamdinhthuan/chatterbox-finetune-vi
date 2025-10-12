"""
Preprocessing script để tăng tốc training >4x
Based on: https://github.com/resemble-ai/chatterbox/issues/174

Idea:
- Pre-compute tất cả expensive operations (audio load, resample, tokenization, VE encoding)
- Save thành .pt files
- Training chỉ cần load .pt files → Nhanh hơn rất nhiều

Results: 2.2-2.9s/it → 1.2s/it (~2x speedup)
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import torch.multiprocessing as mp
from multiprocessing import Queue

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3tokenizer import S3_SR  # S3_SR = 16kHz for speech tokenizer
from chatterbox.models.t3.modules.t3_config import T3Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_silence_padding(audio: np.ndarray, sr: int, padding_ms: int = 300) -> np.ndarray:
    """
    Add silence padding at start and end of audio
    Helps with first/last word issues (from issue comments)
    """
    padding_samples = int(sr * padding_ms / 1000)
    silence = np.zeros(padding_samples, dtype=audio.dtype)
    return np.concatenate([silence, audio, silence])


def preprocess_sample(
    audio_path: Path,
    text: str,
    text_tokenizer,
    speech_tokenizer,
    voice_encoder,
    t3_config: T3Config,
    max_text_len: int,
    max_speech_len: int,
    audio_prompt_duration_s: float,
    add_silence: bool = True,
    silence_padding_ms: int = 500,
) -> Optional[Dict]:
    """
    Preprocess single sample: load audio, tokenize, encode
    """
    try:
        # 1. Load and preprocess audio
        wav, sr_orig = librosa.load(str(audio_path), sr=None, mono=True)
        
        # Add silence padding (helps with first/last word)
        if add_silence:
            wav = add_silence_padding(wav, sr_orig, silence_padding_ms)
        
        # Resample to S3 sample rate (16kHz for speech tokenizer)
        if sr_orig != S3_SR:
            wav = librosa.resample(wav, orig_sr=sr_orig, target_sr=S3_SR)
        
        # Create tensor: 1D for S3Tokenizer
        wav_tensor_1d = torch.from_numpy(wav).float()  # [T] for S3Tokenizer
        
        # 2. Tokenize text and add BOS/EOS
        text_normalized = punc_norm(text)
        raw_text_tokens = text_tokenizer.encode(text_normalized).ids
        
        # Add BOS (255) and EOS (0) tokens
        text_tokens = [t3_config.start_text_token] + raw_text_tokens + [t3_config.stop_text_token]
        
        # Truncate if too long (keep EOS)
        if len(text_tokens) > max_text_len:
            text_tokens = text_tokens[:max_text_len-1] + [t3_config.stop_text_token]
        
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        text_token_len = len(text_tokens)
        
        # 3. Speech tokenization and add BOS/EOS
        with torch.no_grad():
            device = next(speech_tokenizer.parameters()).device
            wav_1d_device = wav_tensor_1d.to(device)
            # Pass as list of 1D tensors, S3Tokenizer will add batch dim internally
            raw_speech_tokens_batch, speech_lengths_batch = speech_tokenizer.forward([wav_1d_device])
            raw_speech_tokens = raw_speech_tokens_batch[0].cpu()[:speech_lengths_batch[0].item()]
        
        # Add BOS (6561) and EOS (6562) tokens
        speech_tokens = torch.cat([
            torch.tensor([t3_config.start_speech_token], dtype=torch.long),
            raw_speech_tokens,
            torch.tensor([t3_config.stop_speech_token], dtype=torch.long)
        ])
        
        # Truncate if too long (keep EOS)
        if len(speech_tokens) > max_speech_len:
            speech_tokens = torch.cat([speech_tokens[:max_speech_len-1], torch.tensor([t3_config.stop_speech_token])])
        
        speech_token_len = len(speech_tokens)
        
        # 4. Voice encoding for speaker conditioning
        prompt_len = int(audio_prompt_duration_s * S3_SR)
        if len(wav) >= prompt_len:
            prompt_wav_np = wav[:prompt_len]
        else:
            # Pad if too short
            pad_len = prompt_len - len(wav)
            prompt_wav_np = np.pad(wav, (0, pad_len), mode='constant')
        
        with torch.no_grad():
            # Use embeds_from_wavs which internally converts to mel-spectrograms
            speaker_emb_np = voice_encoder.embeds_from_wavs([prompt_wav_np], sample_rate=S3_SR)
            speaker_emb = torch.from_numpy(speaker_emb_np[0])  # [D]
        
        # 5. Extract conditioning prompt speech tokens from beginning of audio
        cond_prompt_len = t3_config.speech_cond_prompt_len
        cond_audio_samples = int(audio_prompt_duration_s * S3_SR)
        cond_audio_segment = wav[:cond_audio_samples] if len(wav) >= cond_audio_samples else wav
        
        with torch.no_grad():
            if len(cond_audio_segment) > 0:
                cond_wav_tensor = torch.from_numpy(cond_audio_segment).float().to(device)
                cond_prompt_tokens_batch, _ = speech_tokenizer.forward([cond_wav_tensor], max_len=cond_prompt_len)
                cond_prompt_speech_tokens = cond_prompt_tokens_batch[0].cpu()
            else:
                cond_prompt_speech_tokens = torch.zeros(cond_prompt_len, dtype=torch.long)
        
        # Ensure correct length
        if cond_prompt_speech_tokens.shape[0] < cond_prompt_len:
            pad_len = cond_prompt_len - cond_prompt_speech_tokens.shape[0]
            cond_prompt_speech_tokens = torch.nn.functional.pad(cond_prompt_speech_tokens, (0, pad_len), value=0)
        elif cond_prompt_speech_tokens.shape[0] > cond_prompt_len:
            cond_prompt_speech_tokens = cond_prompt_speech_tokens[:cond_prompt_len]
        
        # 6. Emotion adversarial scalar (default from training code)
        emotion_adv_scalar = torch.tensor(0.5, dtype=torch.float)
        
        # 7. Return preprocessed data in training-compatible format
        return {
            "text_tokens": text_tensor,
            "text_token_lens": torch.tensor(text_token_len, dtype=torch.long),
            "speech_tokens": speech_tokens,
            "speech_token_lens": torch.tensor(speech_token_len, dtype=torch.long),
            "t3_cond_speaker_emb": speaker_emb,
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": emotion_adv_scalar,
            "audio_path": str(audio_path),
            "text": text,
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing {audio_path}: {e}")
        return None


def worker_process(
    worker_id: int,
    samples_queue: Queue,
    result_queue: Queue,
    model_dir: Path,
    tokenizer_path: Path,
    args
):
    """
    Worker process for parallel preprocessing
    Each worker loads its own models to avoid CUDA issues
    """
    try:
        # Determine device for this worker
        if torch.cuda.is_available():
            device = f"cuda:{worker_id % torch.cuda.device_count()}"
        else:
            device = "cpu"
        
        from tokenizers import Tokenizer
        text_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Load ChatterboxTTS for this worker
        tts = ChatterboxTTS.from_local(ckpt_dir=str(model_dir), device=device)
        speech_tokenizer = tts.s3gen.tokenizer
        voice_encoder = tts.ve
        voice_encoder.eval()
        t3_config = tts.t3.hp
        
        # Process samples from queue
        while True:
            item = samples_queue.get()
            if item is None:  # Poison pill to stop worker
                break
            
            idx, sample = item
            
            try:
                preprocessed = preprocess_sample(
                    audio_path=sample["audio_path"],
                    text=sample["text"],
                    text_tokenizer=text_tokenizer,
                    speech_tokenizer=speech_tokenizer,
                    voice_encoder=voice_encoder,
                    t3_config=t3_config,
                    max_text_len=args.max_text_len,
                    max_speech_len=args.max_speech_len,
                    audio_prompt_duration_s=args.audio_prompt_duration,
                    add_silence=args.add_silence,
                    silence_padding_ms=args.silence_padding_ms,
                )
                
                result_queue.put((idx, preprocessed, sample))
                
            except Exception as e:
                # If error, return None for this sample
                result_queue.put((idx, None, sample))
                
    except Exception as e:
        print(f"Worker {worker_id} fatal error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for faster training")
    parser.add_argument("--csv", type=str, required=True, help="Path to metadata.csv")
    parser.add_argument("--audio_dir", type=str, default=".", help="Audio directory")
    parser.add_argument("--output_dir", type=str, default="./preprocessed_data", help="Output directory for .pt files")
    parser.add_argument("--tokenizer", type=str, default="VietnameseTokenizer/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--max_text_len", type=int, default=256, help="Max text length")
    parser.add_argument("--max_speech_len", type=int, default=1200, help="Max speech length")
    parser.add_argument("--audio_prompt_duration", type=float, default=3.0, help="Audio prompt duration (seconds)")
    parser.add_argument("--add_silence", action="store_true", help="Add 300ms silence padding (recommended)")
    parser.add_argument("--silence_padding_ms", type=int, default=500, help="Silence padding in ms")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (default: 1, recommend: 4-8 for faster processing)")
    
    args = parser.parse_args()
    
    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return
    
    audio_dir = Path(args.audio_dir)
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found: {tokenizer_path}")
        logger.info("Please run: python train_tokenizer_from_corpus.py metadata.csv")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("PREPROCESSING DATASET FOR FASTER TRAINING")
    logger.info("="*80)
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Audio dir: {audio_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Tokenizer: {tokenizer_path}")
    logger.info(f"Add silence padding: {args.add_silence} ({args.silence_padding_ms}ms)")
    
    # Load model components
    logger.info("\nLoading model components...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Download model from HF Hub first
        from huggingface_hub import hf_hub_download
        import os
        
        model_dir = Path("./cache/chatterbox_model")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        repo_id = "ResembleAI/chatterbox"
        files_to_download = ["ve.safetensors", "s3gen.safetensors", "t3_cfg.safetensors"]
        
        logger.info(f"Downloading model from {repo_id}...")
        for file in files_to_download:
            if not (model_dir / file).exists():
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
        
        # Try to download conds.pt
        try:
            if not (model_dir / "conds.pt").exists():
                hf_hub_download(
                    repo_id=repo_id,
                    filename="conds.pt",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
        except:
            logger.info("conds.pt not found (optional)")
        
        # Copy custom tokenizer
        import shutil
        shutil.copy(tokenizer_path, model_dir / "tokenizer.json")
        
        # Load model with from_local
        logger.info("Loading model components...")
        tts = ChatterboxTTS.from_local(ckpt_dir=str(model_dir), device=device)
        
        # Get components
        from tokenizers import Tokenizer
        text_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Access correct attributes: tts.s3gen.tokenizer and tts.ve
        speech_tokenizer = tts.s3gen.tokenizer
        voice_encoder = tts.ve
        voice_encoder.eval()
        
        # Get T3 config for BOS/EOS tokens
        t3_config = tts.t3.hp
        
        logger.info("✅ Model components loaded")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load metadata
    logger.info("\nLoading metadata...")
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            if 'audio' in row and 'transcript' in row:
                audio_path = audio_dir / row['audio']
                if audio_path.exists():
                    samples.append({
                        "audio_path": audio_path,
                        "text": row['transcript']
                    })
    
    logger.info(f"✅ Found {len(samples)} samples")
    
    if len(samples) == 0:
        logger.error("No valid samples found!")
        return
    
    # Preprocess all samples
    logger.info("\nPreprocessing samples...")
    logger.info("This will take a while, but only needs to be done once!")
    
    num_workers = args.num_workers
    logger.info(f"Using {num_workers} workers for parallel processing")
    
    successful = 0
    failed = 0
    sample_list = []
    
    if num_workers == 1:
        # Single-threaded (original code)
        for idx, sample in enumerate(tqdm(samples, desc="Preprocessing")):
            output_file = output_dir / f"sample_{idx:06d}.pt"
            
            if output_file.exists():
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
                continue
            
            preprocessed = preprocess_sample(
                audio_path=sample["audio_path"],
                text=sample["text"],
                text_tokenizer=text_tokenizer,
                speech_tokenizer=speech_tokenizer,
                voice_encoder=voice_encoder,
                t3_config=t3_config,
                max_text_len=args.max_text_len,
                max_speech_len=args.max_speech_len,
                audio_prompt_duration_s=args.audio_prompt_duration,
                add_silence=args.add_silence,
                silence_padding_ms=args.silence_padding_ms,
            )
            
            if preprocessed is not None:
                torch.save(preprocessed, output_file)
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
            else:
                failed += 1
    
    else:
        # Multi-processing
        mp.set_start_method('spawn', force=True)
        
        # Create queues
        samples_queue = Queue(maxsize=num_workers * 2)
        result_queue = Queue()
        
        # Start workers
        workers = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(worker_id, samples_queue, result_queue, model_dir, tokenizer_path, args)
            )
            p.start()
            workers.append(p)
        
        # Filter unprocessed samples
        unprocessed_samples = []
        for idx, sample in enumerate(samples):
            output_file = output_dir / f"sample_{idx:06d}.pt"
            if output_file.exists():
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
            else:
                unprocessed_samples.append((idx, sample))
        
        # Enqueue unprocessed samples
        for item in unprocessed_samples:
            samples_queue.put(item)
        
        # Send poison pills
        for _ in range(num_workers):
            samples_queue.put(None)
        
        # Collect results with progress bar
        pbar = tqdm(total=len(unprocessed_samples), desc="Preprocessing")
        processed_count = 0
        
        while processed_count < len(unprocessed_samples):
            idx, preprocessed, sample = result_queue.get()
            
            output_file = output_dir / f"sample_{idx:06d}.pt"
            
            if preprocessed is not None:
                torch.save(preprocessed, output_file)
                successful += 1
                sample_list.append({
                    "idx": idx,
                    "pt_file": f"sample_{idx:06d}.pt",
                    "audio_path": str(sample["audio_path"]),
                    "text": sample["text"]
                })
            else:
                failed += 1
            
            processed_count += 1
            pbar.update(1)
        
        pbar.close()
        
        # Wait for workers
        for p in workers:
            p.join()
        
        logger.info(f"All workers finished")
    
    # Save metadata
    metadata = {
        "num_samples": successful,
        "max_text_len": args.max_text_len,
        "max_speech_len": args.max_speech_len,
        "audio_prompt_duration_s": args.audio_prompt_duration,
        "add_silence": args.add_silence,
        "silence_padding_ms": args.silence_padding_ms,
        "samples": sample_list,  # List of sample info for dataset loading
    }
    
    metadata_file = output_dir / "metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETED!")
    logger.info("="*80)
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"📊 Metadata: {metadata_file}")
    
    # Estimate size
    if successful > 0:
        sample_file = output_dir / "sample_000000.pt"
        if sample_file.exists():
            file_size_mb = sample_file.stat().st_size / (1024 * 1024)
            total_size_mb = file_size_mb * successful
            logger.info(f"💾 Estimated total size: ~{total_size_mb:.1f} MB")
    
    logger.info("\n💡 Next steps:")
    logger.info("   python train.py --csv metadata.csv --use_preprocessed --preprocessed_dir ./preprocessed_data")
    logger.info("\n🚀 Expected speedup: 2-4x faster training!")


if __name__ == "__main__":
    main()
