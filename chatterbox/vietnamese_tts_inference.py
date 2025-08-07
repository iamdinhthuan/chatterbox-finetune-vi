#!/usr/bin/env python3
"""
Vietnamese TTS Inference - Final Version
Complete inference script for trained Vietnamese TTS model.
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torchaudio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VietnameseTTS:
    """Vietnamese TTS Inference Class"""
    
    def __init__(self, model_config="model_path_vietnamese.json", device="auto"):
        """Initialize Vietnamese TTS"""
        self.device = self._setup_device(device)
        self.model = None
        self.load_model(model_config)
    
    def _setup_device(self, device):
        """Setup device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Using device: {device}")
        return device
    
    def load_model(self, config_path):
        """Load Vietnamese TTS model"""
        logger.info("🇻🇳 Loading Vietnamese TTS model...")
        
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            logger.error("❌ ChatterboxTTS not found. Install: pip install chatterbox-tts")
            raise
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use trained model if available
        if os.path.exists("model.safetensors"):
            logger.info("✅ Using trained model: model.safetensors")
            config["t3_path"] = "model.safetensors"
        
        # Convert to Path objects
        current_dir = Path.cwd()
        voice_encoder_path = current_dir / config["voice_encoder_path"]
        t3_path = current_dir / config["t3_path"]
        s3gen_path = current_dir / config["s3gen_path"]
        tokenizer_path = Path(config["tokenizer_path"])
        conds_path = current_dir / config["conds_path"] if config.get("conds_path") else None
        
        # Load model
        self.model = ChatterboxTTS.from_specified(
            voice_encoder_path=voice_encoder_path,
            t3_path=t3_path,
            s3gen_path=s3gen_path,
            tokenizer_path=tokenizer_path,
            conds_path=conds_path,
            device=self.device
        )
        
        logger.info("✅ Model loaded successfully!")
    
    def synthesize(self, text, output_path=None, **kwargs):
        """
        Synthesize Vietnamese text to speech
        
        Args:
            text: Vietnamese text to synthesize
            output_path: Output audio file path
            **kwargs: Additional arguments for generation
        
        Returns:
            audio: Generated audio tensor
        """
        logger.info(f"🎤 Synthesizing: '{text}'")
        
        # Generate audio
        audio = self.model.generate(text=text, **kwargs)
        
        # Save if output path provided
        if output_path:
            self.save_audio(audio, output_path)
            logger.info(f"🔊 Audio saved: {output_path}")
        
        return audio
    
    def save_audio(self, audio, output_path):
        """Save audio to file"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        audio = audio.cpu()
        
        torchaudio.save(output_path, audio, 24000)
    
    def batch_synthesize(self, texts, output_dir="inference_output"):
        """Batch synthesize multiple texts"""
        logger.info(f"📝 Batch synthesizing {len(texts)} texts...")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = []
        for i, text in enumerate(texts, 1):
            try:
                output_path = os.path.join(output_dir, f"vietnamese_tts_{i:03d}_{timestamp}.wav")
                audio = self.synthesize(text, output_path)
                
                results.append({
                    "text": text,
                    "output_path": output_path,
                    "success": True
                })
                logger.info(f"✅ [{i}/{len(texts)}] Completed")
                
            except Exception as e:
                logger.error(f"❌ [{i}/{len(texts)}] Failed: {e}")
                results.append({
                    "text": text,
                    "success": False,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r["success"])
        logger.info(f"🎉 Batch completed: {successful}/{len(texts)} successful")
        
        return results

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Vietnamese TTS Inference")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--texts", type=str, nargs="+", help="Multiple texts")
    parser.add_argument("--file", type=str, help="Text file to read from")
    parser.add_argument("--output", type=str, help="Output audio file")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory")
    parser.add_argument("--config", type=str, default="model_path_vietnamese.json", help="Model config")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--examples", action="store_true", help="Run example texts")
    
    args = parser.parse_args()
    
    # Initialize TTS
    try:
        tts = VietnameseTTS(model_config=args.config, device=args.device)
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        logger.info("🎤 Interactive Vietnamese TTS Mode")
        logger.info("Enter Vietnamese text (or 'quit' to exit)")
        
        while True:
            try:
                text = input("\n📝 Text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(args.output_dir, f"interactive_{timestamp}.wav")
                
                tts.synthesize(text, output_path)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    
    # Single text
    elif args.text:
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(args.output_dir, f"vietnamese_tts_{timestamp}.wav")
        
        tts.synthesize(args.text, output_path)
    
    # Multiple texts
    elif args.texts:
        tts.batch_synthesize(args.texts, args.output_dir)
    
    # Text file
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        tts.batch_synthesize(texts, args.output_dir)
    
    # Examples
    elif args.examples:
        example_texts = [
            "Xin chào, tôi là trợ lý AI tiếng Việt.",
            "Hôm nay là một ngày đẹp trời.",
            "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh.",
            "Việt Nam là một đất nước xinh đẹp.",
            "Cảm ơn bạn đã sử dụng hệ thống TTS tiếng Việt.",
            "Chúc bạn có một ngày tốt lành!",
            "Tôi có thể nói tiếng Việt rất tự nhiên.",
            "Hãy thử nghiệm với nhiều câu khác nhau."
        ]
        
        logger.info("🎯 Running Vietnamese TTS examples...")
        tts.batch_synthesize(example_texts, args.output_dir)
    
    # Default
    else:
        logger.info("🎯 Running default test...")
        test_text = "Xin chào, đây là model TTS tiếng Việt đã được training thành công!"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"test_{timestamp}.wav")
        
        tts.synthesize(test_text, output_path)
        logger.info(f"🎉 Test completed! Audio saved to: {output_path}")

if __name__ == "__main__":
    main()
