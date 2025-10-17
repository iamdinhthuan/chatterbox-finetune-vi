"""
Batch inference script - generate multiple audio files from a text file
Usage: python batch_infer.py --checkpoint ./checkpoints/vietnamese --input texts.txt --output_dir ./outputs
"""

import sys
import argparse
from pathlib import Path
import torch
import torchaudio as ta
import unicodedata
import re
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatterbox.tts import ChatterboxTTS


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for TTS"""
    if not text or len(text.strip()) == 0:
        return "Vui lòng nhập văn bản"
    
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = " ".join(text.split())
    
    text = text.replace("...", " ")
    text = text.replace("…", " ")
    text = text.replace(":", ",")
    text = text.replace(";", ",")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace('"', '')
    text = text.replace("'", '')
    
    text = re.sub(r'[^a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s\.\,\!\?0-9\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def main():
    parser = argparse.ArgumentParser(description="Batch Vietnamese TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input text file (one sentence per line)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--voice", type=str, default=None, help="Reference voice audio")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda/cpu/mps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="CFG weight")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration")
    parser.add_argument("--prefix", type=str, default="output", help="Output filename prefix")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print("="*80)
    print("BATCH VIETNAMESE TTS INFERENCE")
    print("="*80)
    print(f"🖥️  Device: {device}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📄 Input file: {args.input}")
    print(f"📁 Output dir: {args.output_dir}")
    print("="*80 + "\n")
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Read texts
    with open(input_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Found {len(texts)} texts to process\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"📦 Loading model...")
    try:
        model = ChatterboxTTS.from_local(str(args.checkpoint), device=device)
        print("✅ Model loaded!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Prepare voice
    if args.voice:
        voice_path = Path(args.voice)
        if not voice_path.exists():
            print(f"❌ Voice file not found: {voice_path}")
            return
        print(f"🎤 Loading reference voice...")
        model.prepare_conditionals(str(voice_path), exaggeration=args.exaggeration)
        print("✅ Voice prepared\n")
    else:
        if model.conds is None:
            print("⚠️  Using random conditioning")
            import numpy as np
            dummy_wav = np.random.randn(16000 * 3).astype(np.float32) * 0.01
            model.prepare_conditionals(dummy_wav, exaggeration=args.exaggeration)
    
    # Generate
    print(f"🎵 Generating {len(texts)} audio files...\n")
    
    success_count = 0
    failed_count = 0
    
    for i, text in enumerate(tqdm(texts, desc="Processing"), 1):
        try:
            # Normalize
            normalized = normalize_vietnamese(text)
            
            # Generate
            wav = model.generate(
                normalized,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight,
            )
            
            # Save
            output_path = output_dir / f"{args.prefix}_{i:04d}.wav"
            ta.save(str(output_path), wav, model.sr)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Failed on text {i}: {text[:50]}...")
            print(f"   Error: {e}")
            failed_count += 1
    
    print("\n" + "="*80)
    print("✅ BATCH INFERENCE COMPLETED!")
    print("="*80)
    print(f"✅ Success: {success_count}/{len(texts)}")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count}/{len(texts)}")
    print(f"📁 Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

