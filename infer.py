"""
Script inference đơn giản cho Vietnamese TTS với checkpoint đã train
Usage: python infer.py --checkpoint ./checkpoints/vietnamese/checkpoint-45000 --text "Xin chào"
"""

import sys
import argparse
from pathlib import Path
import torch
import torchaudio as ta
from safetensors.torch import load_file
import re
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.models.t3 import T3
from src.chatterbox.models.s3gen import S3Gen
from src.chatterbox.models.voice_encoder import VoiceEncoder
from src.chatterbox.models.tokenizers import EnTokenizer


# Text normalization and sentence splitting (inspired by F5-TTS)


def normalize_text(text: str) -> str:
    """
    Normalize text for TTS (inspired by F5-TTS)
    - Remove multiple spaces
    - Fix punctuation spacing
    - Normalize quotes and dashes
    """
    if not text or len(text.strip()) == 0:
        return text
    
    # Remove multiple spaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Fix punctuation - remove space before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Fix punctuation - add space after punctuation if missing
    text = re.sub(r'([,.!?;:])([^\s\d])', r'\1 \2', text)
    
    # Normalize quotes and dashes (Vietnamese style)
    replacements = [
        ('"', '"'),   # Smart quote to straight quote
        ('"', '"'),   # Smart quote to straight quote  
        (''', "'"),   # Smart apostrophe
        (''', "'"),   # Smart apostrophe
        ('–', '-'),   # En dash to hyphen
        ('—', '-'),   # Em dash to hyphen
        ('…', '...'), # Ellipsis
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def split_sentences(text: str, min_length=10):
    """
    Split text into sentences by punctuation (inspired by F5-TTS)
    Supports: . ? ! ; (both English and Vietnamese punctuation)
    
    Args:
        text: Input text
        min_length: Minimum sentence length to keep (default: 10 chars)
    
    Returns:
        List of sentences with punctuation preserved
    """
    # Split by sentence-ending punctuation
    # Matches: . ? ! ; and their Vietnamese/Chinese equivalents 。！？；
    pattern = r'([.?!。！？]+|[;；]+)'
    
    # Split while keeping delimiters
    parts = re.split(pattern, text)
    
    # Combine text with its punctuation
    sentences = []
    i = 0
    
    while i < len(parts):
        if i >= len(parts):
            break
            
        # Get text part
        sentence = parts[i].strip()
        
        # Get punctuation if exists
        if i + 1 < len(parts) and re.match(pattern, parts[i+1]):
            sentence += parts[i+1]
            i += 2
        else:
            i += 1
        
        # Only keep non-empty sentences above minimum length
        if sentence and len(sentence.strip()) >= min_length:
            sentences.append(sentence)
    
    # If no sentences found, use original text
    return sentences if sentences else [text]


def cross_fade_audio(audio_segments, sample_rate=24000, fade_duration_ms=250):
    """
    Merge audio segments with cross-fading
    
    Args:
        audio_segments: List of audio tensors [batch, samples]
        sample_rate: Audio sample rate (default 24000)
        fade_duration_ms: Cross-fade duration in milliseconds (default 100ms)
    
    Returns:
        Merged audio tensor
    """
    if len(audio_segments) == 0:
        return torch.zeros(1, 0)
    
    if len(audio_segments) == 1:
        return audio_segments[0]
    
    # Calculate fade samples
    fade_samples = int(sample_rate * fade_duration_ms / 1000)
    
    # Start with first segment
    result = audio_segments[0].clone()
    
    for i in range(1, len(audio_segments)):
        current_seg = audio_segments[i]
        
        # If segments are too short for fading, just concatenate
        if result.shape[1] < fade_samples or current_seg.shape[1] < fade_samples:
            result = torch.cat([result, current_seg], dim=1)
            continue
        
        # Create fade curves
        fade_out = torch.linspace(1, 0, fade_samples).to(result.device)
        fade_in = torch.linspace(0, 1, fade_samples).to(current_seg.device)
        
        # Apply fade to overlapping region
        result[:, -fade_samples:] = result[:, -fade_samples:] * fade_out
        
        # Overlap and add
        overlap_region = result[:, -fade_samples:] + current_seg[:, :fade_samples] * fade_in
        result[:, -fade_samples:] = overlap_region
        
        # Append the rest of current segment
        result = torch.cat([result, current_seg[:, fade_samples:]], dim=1)
    
    return result


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    parser.add_argument("--min_tokens", type=int, default=30, help="Minimum speech tokens to generate (default: 30)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--no_drop_tokens", action="store_true", help="Disable drop_invalid_tokens (for debugging)")
    parser.add_argument("--split_sentences", action="store_true", help="Split text by sentences and merge with cross-fading")
    parser.add_argument("--fade_duration", type=int, default=250, help="Cross-fade duration in ms (default: 100)")
    
    args = parser.parse_args()
    
    # Setup logging for debug
    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    
    # Set random seed if provided
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
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
    if args.seed is not None:
        print(f"🎲 Seed: {args.seed}")
    print(f"📏 Min tokens: {args.min_tokens}")
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
    
    # Normalize text (inspired by F5-TTS)
    normalized_text = normalize_text(args.text)
    print(f"📝 Original text: {args.text}")
    if normalized_text != args.text:
        print(f"📝 Normalized text: {normalized_text}")
    print()
    
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
        if args.split_sentences:
            # Split normalized text into sentences (inspired by F5-TTS)
            sentences = split_sentences(normalized_text)
            print(f"📝 Split into {len(sentences)} sentences:")
            for i, sent in enumerate(sentences, 1):
                print(f"   {i}. {sent}")
            print()
            
            # Generate audio for each sentence
            audio_segments = []
            for i, sentence in enumerate(sentences, 1):
                print(f"🎙️  Generating sentence {i}/{len(sentences)}: {sentence[:50]}...")
                
                wav_seg = model.generate(
                    sentence,
                    temperature=args.temperature,
                    cfg_weight=args.cfg_weight,
                    min_tokens=args.min_tokens,
                )
                audio_segments.append(wav_seg)
                
                seg_duration = wav_seg.shape[-1] / model.sr
                print(f"   ✓ Generated {seg_duration:.2f}s")
            
            # Merge audio segments with cross-fading
            print(f"\n🔗 Merging {len(audio_segments)} segments with {args.fade_duration}ms cross-fade...")
            wav = cross_fade_audio(audio_segments, sample_rate=model.sr, fade_duration_ms=args.fade_duration)
            print(f"   ✓ Merged successfully")
        else:
            # Generate full text at once
            wav = model.generate(
                normalized_text,  # Use normalized text
                temperature=args.temperature,
                cfg_weight=args.cfg_weight,
                min_tokens=args.min_tokens,
            )
        
        # Check audio quality
        duration = wav.shape[-1] / model.sr
        if duration < 0.5:
            print(f"⚠️  WARNING: Generated audio is very short ({duration:.2f}s)")
            print(f"   This may indicate an issue with the voice reference or generation parameters")
        
        # Save output
        output_path = Path(args.output)
        ta.save(str(output_path), wav, model.sr)
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Audio saved: {output_path}")
        print(f"🎵 Sample rate: {model.sr} Hz")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"📊 Audio shape: {wav.shape}")
        if args.split_sentences:
            print(f"📊 Sentences: {len(sentences)}")
            print(f"⏱️  Fade duration: {args.fade_duration}ms")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

