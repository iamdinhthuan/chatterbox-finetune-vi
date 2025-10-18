"""
Pre-compute embeddings and tokens for faster training

Usage:
    # With GPU (fastest, recommended for single worker)
    python preprocess_dataset.py \
        --metadata_csv ./metadata.csv \
        --audio_dir ./wavs \
        --output_dir ./data/preprocessed \
        --checkpoint ./vietnamese/pretrained_model_download \
        --device cuda \
        --num_workers 1
    
    # With CPU and parallel processing (if GPU memory limited)
    python preprocess_dataset.py \
        --metadata_csv ./metadata.csv \
        --audio_dir ./wavs \
        --output_dir ./data/preprocessed \
        --checkpoint ./vietnamese/pretrained_model_download \
        --device cpu \
        --num_workers 8
"""

import argparse
import json
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, Manager
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.models.tokenizers import EnTokenizer


def punc_norm(text: str) -> str:
    """Quick cleanup func for punctuation"""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    
    # Capitalize first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Remove multiple space chars
    text = " ".join(text.split())
    
    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    
    return text


def process_single_item(args):
    """Process a single audio-text pair"""
    idx, item, audio_dir, output_dir, checkpoint_dir, config, device = args
    
    try:
        # Parse item
        parts = item.strip().split('|')
        if len(parts) < 2:
            return idx, False, "Invalid format"
        
        audio_path = parts[0]
        text = parts[1]
        
        # Make absolute path
        if not Path(audio_path).is_absolute():
            audio_path = Path(audio_dir) / audio_path
        else:
            audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return idx, False, f"Audio not found: {audio_path}"
        
        # Output path
        audio_id = f"{idx:06d}"
        output_path = Path(output_dir) / f"{audio_id}.pt"
        
        # Skip if already processed
        if output_path.exists():
            return idx, True, "Already processed"
        
        # Load audio
        wav, sr = torchaudio.load(str(audio_path))
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        wav_16k = wav.squeeze(0).numpy()
        
        # Check audio length
        if len(wav_16k) < 16000 * 0.5:  # Less than 0.5s
            return idx, False, "Audio too short"
        
        if len(wav_16k) > 16000 * 30:  # More than 30s
            return idx, False, "Audio too long"
        
        # Load model in this process (lazy loading)
        if not hasattr(process_single_item, 'model'):
            print(f"[Worker {idx%4}] Loading model on {device}...")
            process_single_item.model = ChatterboxTTS.from_local(checkpoint_dir, device=device)
            process_single_item.config = config
            process_single_item.device = device
        
        model = process_single_item.model
        
        # 1. Compute voice embedding (on device)
        speaker_emb_np = model.ve.embeds_from_wavs([wav_16k], sample_rate=16000)
        speaker_emb = torch.from_numpy(speaker_emb_np[0]).cpu()  # Move to CPU for saving
        
        # 2. Tokenize text
        normalized_text = punc_norm(text)
        raw_text_tokens = model.tokenizer.text_to_tokens(normalized_text).squeeze(0)
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=config['start_text_token'])
        text_tokens = F.pad(text_tokens, (0, 1), value=config['stop_text_token'])
        
        # Truncate if too long
        max_text_len = config.get('max_text_len', 512)
        if len(text_tokens) > max_text_len:
            text_tokens = text_tokens[:max_text_len-1]
            text_tokens = torch.cat([text_tokens, torch.tensor([config['stop_text_token']])])
        
        # 3. Tokenize speech (on device)
        raw_speech_tokens_batch, speech_token_lengths_batch = model.s3gen.tokenizer.forward([wav_16k])
        if raw_speech_tokens_batch is None:
            return idx, False, "Speech tokenization failed"
        
        raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()].cpu()
        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=config['start_speech_token'])
        speech_tokens = F.pad(speech_tokens, (0, 1), value=config['stop_speech_token'])
        
        # Truncate if too long
        max_speech_len = config.get('max_speech_len', 2048)
        if len(speech_tokens) > max_speech_len:
            speech_tokens = speech_tokens[:max_speech_len-1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([config['stop_speech_token']])])
        
        # 4. Conditioning prompt tokens
        enc_cond_audio_len = config.get('enc_cond_audio_len', 3.0)
        enc_cond_audio_len_samples = int(enc_cond_audio_len * 16000)
        speech_cond_prompt_len = config.get('speech_cond_prompt_len', 150)
        
        cond_audio_segment = wav_16k[:enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = model.s3gen.tokenizer.forward([cond_audio_segment], max_len=speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    cond_prompt_speech_tokens = torch.zeros(speech_cond_prompt_len, dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0).cpu()  # Move to CPU
            except:
                cond_prompt_speech_tokens = torch.zeros(speech_cond_prompt_len, dtype=torch.long)
        
        # Pad/truncate to exact length
        if cond_prompt_speech_tokens.size(0) != speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            if current_len > speech_cond_prompt_len:
                cond_prompt_speech_tokens = cond_prompt_speech_tokens[:speech_cond_prompt_len]
            else:
                cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, speech_cond_prompt_len - current_len), value=0)
        
        # 5. Save preprocessed data
        torch.save({
            'text_tokens': text_tokens.long(),
            'text_token_lens': torch.tensor(len(text_tokens), dtype=torch.long),
            'speech_tokens': speech_tokens.long(),
            'speech_token_lens': torch.tensor(len(speech_tokens), dtype=torch.long),
            't3_cond_speaker_emb': speaker_emb.float(),
            't3_cond_prompt_speech_tokens': cond_prompt_speech_tokens.long(),
            't3_cond_emotion_adv': torch.tensor(0.5, dtype=torch.float),
            'text': text,
            'audio_path': str(audio_path),
        }, output_path)
        
        return idx, True, "Success"
        
    except Exception as e:
        import traceback
        return idx, False, f"Error: {str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings and tokens for training")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preprocessed files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ChatterboxTTS checkpoint")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda, cpu, or mps (default: cuda)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start from this index (for resuming)")
    parser.add_argument("--end_idx", type=int, default=None, help="End at this index (optional)")
    
    args = parser.parse_args()
    
    metadata_path = Path(args.metadata_csv)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint)
    device = args.device
    
    # Auto-detect device if needed
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Warn about GPU memory with multiple workers
    if device == 'cuda' and args.num_workers > 1:
        print("⚠️  WARNING: Using GPU with multiple workers will load model multiple times!")
        print("   This may cause GPU OOM. Recommended: num_workers=1 with GPU")
        print("   Or use CPU with num_workers>1 for parallel processing.")
        import time
        time.sleep(3)
    
    # Validate paths
    if not metadata_path.exists():
        print(f"❌ Metadata CSV not found: {metadata_path}")
        return
    
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"📁 Metadata CSV: {metadata_path}")
    print(f"📁 Audio directory: {audio_dir}")
    print(f"🖥️  Device: {device}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header if it contains "audio" or "transcript"
    if lines and ('audio' in lines[0].lower() or 'transcript' in lines[0].lower()):
        print(f"📝 Skipping header line: {lines[0].strip()}")
        lines = lines[1:]
    
    print(f"📝 Found {len(lines)} items in metadata")
    
    # Apply start/end idx
    if args.end_idx:
        lines = lines[args.start_idx:args.end_idx]
    else:
        lines = lines[args.start_idx:]
    
    print(f"📝 Processing {len(lines)} items (from idx {args.start_idx})")
    
    # Load config
    print(f"📦 Loading config from checkpoint...")
    try:
        model = ChatterboxTTS.from_local(checkpoint_dir, device='cpu')
        config = {
            'start_text_token': model.t3.hp.start_text_token,
            'stop_text_token': model.t3.hp.stop_text_token,
            'start_speech_token': model.t3.hp.start_speech_token,
            'stop_speech_token': model.t3.hp.stop_speech_token,
            'max_text_len': 512,
            'max_speech_len': 2048,
            'enc_cond_audio_len': 3.0,
            'speech_cond_prompt_len': model.t3.hp.speech_cond_prompt_len,
        }
        print(f"✅ Config loaded")
        del model  # Free memory
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Prepare arguments for workers
    tasks = [
        (args.start_idx + i, line, str(audio_dir), str(output_dir), str(checkpoint_dir), config, device)
        for i, line in enumerate(lines)
    ]
    
    print(f"\n🚀 Starting preprocessing with {args.num_workers} workers...")
    print(f"📁 Output: {output_dir}\n")
    
    # Process with progress bar
    success_count = 0
    fail_count = 0
    errors = []
    
    if args.num_workers > 1:
        # Multi-processing
        with Pool(processes=args.num_workers) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_single_item, tasks), total=len(tasks), desc="Processing"):
                idx, success, msg = result
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    errors.append((idx, msg))
                results.append(result)
    else:
        # Single process
        for task in tqdm(tasks, desc="Processing"):
            idx, success, msg = process_single_item(task)
            if success:
                success_count += 1
            else:
                fail_count += 1
                errors.append((idx, msg))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS: {success_count}/{len(lines)}")
    print(f"❌ FAILED: {fail_count}/{len(lines)}")
    
    if errors:
        print(f"\n⚠️  Failed items:")
        for idx, msg in errors[:10]:  # Show first 10 errors
            print(f"   [{idx}] {msg}")
        if len(errors) > 10:
            print(f"   ... and {len(errors)-10} more errors")
    
    # Save summary
    summary = {
        'total': len(lines),
        'success': success_count,
        'failed': fail_count,
        'errors': [{'idx': idx, 'error': msg} for idx, msg in errors],
        'config': config,
    }
    
    summary_path = output_dir / 'preprocessing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved: {summary_path}")
    print(f"{'='*80}\n")
    
    print(f"✅ Preprocessing complete!")
    print(f"📁 Preprocessed files: {output_dir}")
    print(f"\n💡 Next step: Train with preprocessed data")
    print(f"   python src/finetune_t3_thai.py --preprocessed_dir {output_dir} ...")


if __name__ == "__main__":
    main()
