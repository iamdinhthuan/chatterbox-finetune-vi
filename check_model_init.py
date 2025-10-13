"""
Check if model is loading pretrained weights properly

Usage:
    python check_model_init.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_model_initialization():
    """Check if model loads pretrained weights"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Checking Model Initialization")
    logger.info(f"{'='*60}\n")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Load ChatterboxTTS (includes T3, S3Gen, Voice Encoder)
        logger.info("📥 Loading ChatterboxTTS model...")
        logger.info("This will download pretrained weights from HuggingFace...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {device}")
        
        model = ChatterboxTTS.from_pretrained(
            device=device,
            cache_dir="./.cache"
        )
        
        logger.info(f"Model loaded successfully")
        
        # Check T3 model specifically
        t3_model = model.t3
        logger.info(f"\nT3 model: {type(t3_model)}")
        
        # Check T3 weights specifically
        logger.info(f"\n🔍 Checking T3 weights...")
        
        param_stats = []
        for name, param in t3_model.named_parameters():
            std = param.std().item()
            mean = param.mean().item()
            param_stats.append({
                'name': name,
                'std': std,
                'mean': mean,
                'shape': param.shape
            })
        
        # Show first 10 T3 parameters
        logger.info(f"\nFirst 10 T3 parameters:")
        for i, stat in enumerate(param_stats[:10]):
            logger.info(f"  {stat['name']}")
            logger.info(f"    Shape: {stat['shape']}, Mean: {stat['mean']:.4f}, Std: {stat['std']:.4f}")
        
        # Analyze statistics
        avg_std = sum(s['std'] for s in param_stats) / len(param_stats)
        logger.info(f"\n📊 T3 Statistics:")
        logger.info(f"  Total T3 parameters: {len(param_stats)}")
        logger.info(f"  Average std: {avg_std:.4f}")
        
        if avg_std < 0.001:
            logger.error(f"\n❌ CRITICAL: Average std too low ({avg_std:.6f})")
            logger.error(f"Weights might be all zeros or not loaded properly!")
        elif avg_std < 0.01:
            logger.warning(f"\n⚠️ WARNING: Average std quite low ({avg_std:.4f})")
            logger.warning(f"Weights might be random initialization")
        else:
            logger.info(f"\n✅ T3 weights look reasonable (std={avg_std:.4f})")
        
        # Test TTS inference
        logger.info(f"\n🧪 Testing TTS inference with dummy text...")
        
        test_text = "Xin chào"
        
        try:
            with torch.no_grad():
                wav = model.text_to_speech(test_text)
            
            logger.info(f"\n📤 TTS inference result:")
            logger.info(f"  Output shape: {wav.shape}")
            logger.info(f"  Output dtype: {wav.dtype}")
            logger.info(f"  Has NaN: {torch.isnan(wav).any()}")
            logger.info(f"  Has Inf: {torch.isinf(wav).any()}")
            logger.info(f"  Min/Max: {wav.min():.4f}/{wav.max():.4f}")
            
            if torch.isnan(wav).any():
                logger.error(f"\n❌ CRITICAL: TTS output contains NaN!")
            elif torch.isinf(wav).any():
                logger.error(f"\n❌ CRITICAL: TTS output contains Inf!")
            else:
                logger.info(f"\n✅ TTS inference works! Model is properly loaded.")
                
        except Exception as e:
            logger.error(f"\n❌ TTS inference failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        logger.error(f"\n❌ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()


def main():
    check_model_initialization()


if __name__ == "__main__":
    main()
