"""
Script inference đơn giản cho Vietnamese TTS với checkpoint đã train
Usage: python infer.py --checkpoint ./checkpoints/vietnamese/checkpoint-45000 --text "Xin chào"
"""

import sys
import argparse
from pathlib import Path
import torch
import torchaudio as ta
import unicodedata
import re
from safetensors.torch import load_file

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.models.t3 import T3
from src.chatterbox.models.s3gen import S3Gen
from src.chatterbox.models.voice_encoder import VoiceEncoder
from src.chatterbox.models.tokenizers import EnTokenizer


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for TTS"""
    if not text or len(text.strip()) == 0:
        return "Vui lòng nhập văn bản"
    
    # Normalize Unicode to NFC
    text = unicodedata.normalize('NFC', text)
    
    # Lowercase
    text = text.lower()
    
    # Basic cleanup
    text = " ".join(text.split())
    
    # Handle punctuation
    text = text.replace("...", " ")
    text = text.replace("…", " ")
    text = text.replace(":", ",")
    text = text.replace(";", ",")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    
    # Remove quotes
    text = text.replace('"', '')
    text = text.replace("'", '')
    
    # Keep only Vietnamese chars, numbers, basic punctuation
    text = re.sub(r'[^a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s\.\,\!\?0-9\-]', '', text)
    
    # Clean up spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def load_finetuned_model(checkpoint_path: Path, base_model_path: Path, device: str):
    """
    Load finetuned T3 model from checkpoint and combine with pretrained VE/S3Gen

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., checkpoints/vietnamese/checkpoint-45000)
        base_model_path: Path to base pretrained model directory
        device: Device to load model on
    """
    print(f"📦 Loading finetuned model from checkpoint...")

    # Check if checkpoint has full model files
    has_full_model = (checkpoint_path / "ve.safetensors").exists()

    if has_full_model:
        # Checkpoint has all files, load directly
        print("   ✓ Found complete model in checkpoint")
        model = ChatterboxTTS.from_local(str(checkpoint_path), device=device)
    else:
        # Checkpoint only has T3, need to combine with pretrained
        print("   ✓ Loading finetuned T3 from checkpoint")
        print("   ✓ Loading pretrained VE/S3Gen from base model")

        # Load pretrained components
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(base_model_path / "ve.safetensors"))
        ve.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(base_model_path / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        # Load finetuned T3
        t3 = T3()

        # Try different checkpoint filenames
        checkpoint_file = None
        for filename in ["model.safetensors", "pytorch_model.safetensors", "t3_cfg.safetensors"]:
            if (checkpoint_path / filename).exists():
                checkpoint_file = checkpoint_path / filename
                break

        if checkpoint_file is None:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_path}")

        print(f"   Loading from: {checkpoint_file.name}")
        t3_checkpoint = load_file(checkpoint_file)

        # Extract T3 state dict - checkpoint has "t3." prefix
        t3_state = {}
        for key, value in t3_checkpoint.items():
            if key.startswith("t3."):
                new_key = key.replace("t3.", "", 1)
                t3_state[new_key] = value

        if not t3_state:
            # No "t3." prefix, use as is
            t3_state = t3_checkpoint

        print(f"   Loaded {len(t3_state)} T3 parameters")
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        # Load tokenizer
        tokenizer_path = checkpoint_path / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = base_model_path / "tokenizer.json"
        tokenizer = EnTokenizer(str(tokenizer_path))

        # Load conds if available
        conds = None
        conds_path = checkpoint_path / "conds.pt"
        if not conds_path.exists():
            conds_path = base_model_path / "conds.pt"
        if conds_path.exists():
            from src.chatterbox.tts import Conditionals
            map_location = torch.device('cpu') if device in ["cpu", "mps"] else None
            conds = Conditionals.load(str(conds_path), map_location=map_location).to(device)

        model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)

    print("✅ Model loaded successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Vietnamese TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--base_model", type=str, default="./checkpoints/vietnamese/pretrained_model_download",
                        help="Path to base pretrained model (default: ./checkpoints/vietnamese/pretrained_model_download)")
    parser.add_argument("--text", type=str, required=True, help="Vietnamese text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file (default: output.wav)")
    parser.add_argument("--voice", type=str, default=None, help="Path to reference voice audio (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda/cpu/mps (auto-detect if not specified)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="CFG weight (default: 0.5)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (default: 0.5)")
    
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
    print("VIETNAMESE TTS INFERENCE")
    print("="*80)
    print(f"🖥️  Device: {device}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📁 Base model: {args.base_model}")
    print(f"📝 Text: {args.text}")
    if args.voice:
        print(f"🎤 Voice: {args.voice}")
    print(f"🎛️  Temperature: {args.temperature}")
    print(f"🎛️  CFG weight: {args.cfg_weight}")
    print(f"🎛️  Exaggeration: {args.exaggeration}")
    print("="*80 + "\n")

    # Check paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    base_model_path = Path(args.base_model)
    if not base_model_path.exists():
        print(f"❌ Base model not found: {base_model_path}")
        print(f"💡 Tip: The base model should be at: ./checkpoints/vietnamese/pretrained_model_download")
        return

    # Load model
    try:
        model = load_finetuned_model(checkpoint_path, base_model_path, device)
        print()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Normalize text
    normalized_text = normalize_vietnamese(args.text)
    print(f"📝 Original text: {args.text}")
    print(f"📝 Normalized text: {normalized_text}\n")
    
    # Prepare voice conditioning
    if args.voice:
        voice_path = Path(args.voice)
        if not voice_path.exists():
            print(f"❌ Voice file not found: {voice_path}")
            return
        print(f"🎤 Loading reference voice from: {voice_path}")
        model.prepare_conditionals(str(voice_path), exaggeration=args.exaggeration)
        print("✅ Voice conditioning prepared\n")
    else:
        if model.conds is None:
            print("⚠️  No built-in voice found, using random conditioning")
            import numpy as np
            dummy_wav = np.random.randn(16000 * 3).astype(np.float32) * 0.01
            model.prepare_conditionals(dummy_wav, exaggeration=args.exaggeration)
            print("✅ Random conditioning prepared\n")
        else:
            print("✅ Using built-in voice from checkpoint\n")
    
    # Generate speech
    print(f"🎵 Generating speech...")
    try:
        wav = model.generate(
            normalized_text,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
        )
        
        # Save output
        output_path = Path(args.output)
        ta.save(str(output_path), wav, model.sr)
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Audio saved: {output_path}")
        print(f"🎵 Sample rate: {model.sr} Hz")
        print(f"⏱️  Duration: {wav.shape[-1] / model.sr:.2f}s")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

