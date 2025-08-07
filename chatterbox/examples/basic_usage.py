#!/usr/bin/env python3
"""
Basic usage examples for Vietnamese TTS Voice Cloning
"""

import os
import sys
import torch
import torchaudio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def example_basic_tts():
    """Example: Basic Vietnamese TTS without voice cloning"""
    print("🎯 Example 1: Basic Vietnamese TTS")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Load model (using default English model for this example)
        model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate speech
        text = "Hello, this is a basic TTS example."
        audio = model.generate(text=text)
        
        # Save audio
        output_path = "examples/outputs/basic_tts.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, audio, 24000)
        
        print(f"✅ Basic TTS completed: {output_path}")
        
    except Exception as e:
        print(f"❌ Basic TTS failed: {e}")

def example_vietnamese_tts():
    """Example: Vietnamese TTS with trained model"""
    print("\n🇻🇳 Example 2: Vietnamese TTS")
    
    try:
        import json
        from chatterbox.tts import ChatterboxTTS
        
        # Check if Vietnamese model exists
        config_path = "chatterbox/model_path_vietnamese.json"
        if not os.path.exists(config_path):
            print("⚠️  Vietnamese model not found. Please train the model first.")
            return
        
        # Load Vietnamese model config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        current_dir = Path("chatterbox")
        model = ChatterboxTTS.from_specified(
            voice_encoder_path=current_dir / config["voice_encoder_path"],
            t3_path=current_dir / config["t3_path"],
            s3gen_path=current_dir / config["s3gen_path"],
            tokenizer_path=Path(config["tokenizer_path"]),
            conds_path=current_dir / config["conds_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Generate Vietnamese speech
        text = "Xin chào, tôi là trợ lý AI tiếng Việt."
        audio = model.generate(text=text)
        
        # Save audio
        output_path = "examples/outputs/vietnamese_tts.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, audio, 24000)
        
        print(f"✅ Vietnamese TTS completed: {output_path}")
        
    except Exception as e:
        print(f"❌ Vietnamese TTS failed: {e}")

def example_voice_cloning():
    """Example: Voice cloning with reference audio"""
    print("\n🎭 Example 3: Voice Cloning")
    
    try:
        import json
        from chatterbox.tts import ChatterboxTTS
        
        # Check if Vietnamese model exists
        config_path = "chatterbox/model_path_vietnamese.json"
        if not os.path.exists(config_path):
            print("⚠️  Vietnamese model not found. Please train the model first.")
            return
        
        # Check if reference audio exists
        reference_audio = "examples/reference_voice.wav"
        if not os.path.exists(reference_audio):
            print("⚠️  Reference audio not found. Please provide reference_voice.wav in examples/")
            return
        
        # Load Vietnamese model
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        current_dir = Path("chatterbox")
        model = ChatterboxTTS.from_specified(
            voice_encoder_path=current_dir / config["voice_encoder_path"],
            t3_path=current_dir / config["t3_path"],
            s3gen_path=current_dir / config["s3gen_path"],
            tokenizer_path=Path(config["tokenizer_path"]),
            conds_path=current_dir / config["conds_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Generate speech with voice cloning
        text = "Đây là ví dụ về voice cloning với tiếng Việt."
        audio = model.generate(
            text=text,
            audio_prompt_path=reference_audio,
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5
        )
        
        # Save audio
        output_path = "examples/outputs/voice_cloning.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, audio, 24000)
        
        print(f"✅ Voice cloning completed: {output_path}")
        
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")

def example_batch_processing():
    """Example: Batch processing multiple texts"""
    print("\n📦 Example 4: Batch Processing")
    
    try:
        import json
        from datetime import datetime
        from chatterbox.tts import ChatterboxTTS
        
        # Check if Vietnamese model exists
        config_path = "chatterbox/model_path_vietnamese.json"
        if not os.path.exists(config_path):
            print("⚠️  Vietnamese model not found. Using basic model.")
            model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
            texts = [
                "Hello, this is the first sentence.",
                "This is the second sentence for batch processing.",
                "And this is the third sentence to complete the batch."
            ]
        else:
            # Load Vietnamese model
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            current_dir = Path("chatterbox")
            model = ChatterboxTTS.from_specified(
                voice_encoder_path=current_dir / config["voice_encoder_path"],
                t3_path=current_dir / config["t3_path"],
                s3gen_path=current_dir / config["s3gen_path"],
                tokenizer_path=Path(config["tokenizer_path"]),
                conds_path=current_dir / config["conds_path"],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            texts = [
                "Đây là câu đầu tiên trong batch processing.",
                "Câu thứ hai để test khả năng xử lý hàng loạt.",
                "Và đây là câu cuối cùng để hoàn thành batch."
            ]
        
        # Process batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"examples/outputs/batch_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(texts, 1):
            print(f"Processing {i}/{len(texts)}: {text[:30]}...")
            
            audio = model.generate(text=text)
            output_path = os.path.join(output_dir, f"batch_{i:03d}.wav")
            torchaudio.save(output_path, audio, 24000)
            
            print(f"✅ Saved: {output_path}")
        
        print(f"✅ Batch processing completed: {output_dir}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")

def example_parameter_tuning():
    """Example: Parameter tuning for different effects"""
    print("\n⚙️ Example 5: Parameter Tuning")
    
    try:
        import json
        from chatterbox.tts import ChatterboxTTS
        
        # Check if Vietnamese model exists
        config_path = "chatterbox/model_path_vietnamese.json"
        if not os.path.exists(config_path):
            print("⚠️  Vietnamese model not found. Please train the model first.")
            return
        
        # Load Vietnamese model
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        current_dir = Path("chatterbox")
        model = ChatterboxTTS.from_specified(
            voice_encoder_path=current_dir / config["voice_encoder_path"],
            t3_path=current_dir / config["t3_path"],
            s3gen_path=current_dir / config["s3gen_path"],
            tokenizer_path=Path(config["tokenizer_path"]),
            conds_path=current_dir / config["conds_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        text = "Đây là ví dụ về điều chỉnh tham số."
        output_dir = "examples/outputs/parameter_tuning"
        os.makedirs(output_dir, exist_ok=True)
        
        # Different parameter combinations
        configs = [
            {"name": "conservative", "temperature": 0.5, "cfg_weight": 0.8},
            {"name": "balanced", "temperature": 0.8, "cfg_weight": 0.5},
            {"name": "creative", "temperature": 1.0, "cfg_weight": 0.2},
        ]
        
        for config in configs:
            print(f"Generating with {config['name']} settings...")
            
            audio = model.generate(
                text=text,
                temperature=config["temperature"],
                cfg_weight=config["cfg_weight"]
            )
            
            output_path = os.path.join(output_dir, f"{config['name']}.wav")
            torchaudio.save(output_path, audio, 24000)
            
            print(f"✅ {config['name']}: {output_path}")
        
        print(f"✅ Parameter tuning completed: {output_dir}")
        
    except Exception as e:
        print(f"❌ Parameter tuning failed: {e}")

def main():
    """Run all examples"""
    print("🚀 Vietnamese TTS Voice Cloning Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("examples/outputs", exist_ok=True)
    
    # Run examples
    example_basic_tts()
    example_vietnamese_tts()
    example_voice_cloning()
    example_batch_processing()
    example_parameter_tuning()
    
    print("\n🎉 All examples completed!")
    print("Check the 'examples/outputs' directory for generated audio files.")

if __name__ == "__main__":
    main()
