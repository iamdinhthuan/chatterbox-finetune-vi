"""
Script test đơn giản cho Vietnamese TTS
Usage: python test.py --model ./checkpoints/vietnamese --text "Xin chào"
"""

import sys
import argparse
from pathlib import Path
import torch
import torchaudio as ta
import unicodedata
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatterbox.tts import ChatterboxTTS


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


def main():
    parser = argparse.ArgumentParser(description="Test Vietnamese TTS")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Vietnamese text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="CFG weight")
    
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
    
    print(f"🖥️  Device: {device}")
    
    # Check model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    print(f"📦 Loading model from: {model_path}")
    try:
        model = ChatterboxTTS.from_local(str(model_path), device=device)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get text
    if args.text is None:
        test_texts = [
            "Xin chào, tôi là trợ lý ảo tiếng Việt.",
            "Hôm nay trời đẹp quá!",
            "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh.",
        ]
        
        print("\n📝 Choose a test sentence or enter your own:")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        print(f"  {len(test_texts) + 1}. Enter custom text")
        
        choice = input(f"\nChoice (1-{len(test_texts) + 1}): ").strip()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(test_texts):
                text = test_texts[choice_num - 1]
            else:
                text = input("Enter Vietnamese text: ").strip()
        except ValueError:
            text = input("Enter Vietnamese text: ").strip()
    else:
        text = args.text
    
    if not text:
        print("❌ No text provided!")
        return
    
    print(f"\n📝 Original: {text}")
    
    # Normalize
    normalized = normalize_vietnamese(text)
    print(f"📝 Normalized: {normalized}")
    
    # Generate
    print(f"\n🎵 Generating speech...")
    print(f"   Temperature: {args.temperature}")
    print(f"   CFG weight: {args.cfg_weight}")
    
    try:
        # Use built-in voice if available
        if model.conds is None:
            print("⚠️  No built-in voice, using random conditioning")
            import numpy as np
            dummy_wav = np.random.randn(16000 * 3).astype(np.float32) * 0.01
            model.prepare_conditionals(dummy_wav, exaggeration=0.5)
        
        # Generate
        wav = model.generate(
            normalized,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
        )
        
        # Save
        output_path = Path(args.output)
        ta.save(str(output_path), wav, model.sr)
        
        print(f"\n✅ Audio saved: {output_path}")
        print(f"   Sample rate: {model.sr} Hz")
        print(f"   Duration: {wav.shape[-1] / model.sr:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

