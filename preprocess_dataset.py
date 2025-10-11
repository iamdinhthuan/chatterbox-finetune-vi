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

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3tokenizer import S3_SR

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
    max_text_len: int,
    max_speech_len: int,
    audio_prompt_duration_s: float,
    add_silence: bool = True,
    silence_padding_ms: int = 300,
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
        
        # Resample to S3 sample rate (24kHz)
        if sr_orig != S3_SR:
            wav = librosa.resample(wav, orig_sr=sr_orig, target_sr=S3_SR)
        
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        
        # 2. Tokenize text
        text_normalized = punc_norm(text)
        text_tokens = text_tokenizer.encode(text_normalized).ids
        
        if len(text_tokens) > max_text_len:
            logger.warning(f"Text too long ({len(text_tokens)} > {max_text_len}): {audio_path}")
            return None
        
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        
        # 3. Speech tokenization
        with torch.no_grad():
            speech_tokens = speech_tokenizer.encode(wav_tensor).squeeze(0)  # [T]
        
        if speech_tokens.shape[0] > max_speech_len:
            logger.warning(f"Speech too long ({speech_tokens.shape[0]} > {max_speech_len}): {audio_path}")
            return None
        
        # 4. Voice encoding (for prompt)
        prompt_len = int(audio_prompt_duration_s * S3_SR)
        if wav_tensor.shape[-1] >= prompt_len:
            prompt_wav = wav_tensor[:, :prompt_len]
        else:
            # Pad if too short
            pad_len = prompt_len - wav_tensor.shape[-1]
            prompt_wav = F.pad(wav_tensor, (0, pad_len))
        
        with torch.no_grad():
            voice_emb = voice_encoder(prompt_wav)  # [1, D]
        
        # 5. Return preprocessed data
        return {
            "text_tokens": text_tensor,
            "speech_tokens": speech_tokens,
            "voice_emb": voice_emb.squeeze(0),  # [D]
            "audio_path": str(audio_path),
            "text": text,
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing {audio_path}: {e}")
        return None


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
    parser.add_argument("--silence_padding_ms", type=int, default=300, help="Silence padding in ms")
    
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
        
        repo_id = "tel4vn/chatterxbox"
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
        
        speech_tokenizer = tts.model.speech_tokenizer
        voice_encoder = tts.model.ve
        voice_encoder.eval()
        
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
    
    successful = 0
    failed = 0
    
    for idx, sample in enumerate(tqdm(samples, desc="Preprocessing")):
        output_file = output_dir / f"sample_{idx:06d}.pt"
        
        if output_file.exists():
            # Skip if already preprocessed
            successful += 1
            continue
        
        preprocessed = preprocess_sample(
            audio_path=sample["audio_path"],
            text=sample["text"],
            text_tokenizer=text_tokenizer,
            speech_tokenizer=speech_tokenizer,
            voice_encoder=voice_encoder,
            max_text_len=args.max_text_len,
            max_speech_len=args.max_speech_len,
            audio_prompt_duration_s=args.audio_prompt_duration,
            add_silence=args.add_silence,
            silence_padding_ms=args.silence_padding_ms,
        )
        
        if preprocessed is not None:
            torch.save(preprocessed, output_file)
            successful += 1
        else:
            failed += 1
    
    # Save metadata
    metadata = {
        "num_samples": successful,
        "max_text_len": args.max_text_len,
        "max_speech_len": args.max_speech_len,
        "audio_prompt_duration_s": args.audio_prompt_duration,
        "add_silence": args.add_silence,
        "silence_padding_ms": args.silence_padding_ms,
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
