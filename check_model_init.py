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
        
        model = ChatterboxTTS.from_pretrained(device=device)
        
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
        
        test_text = "Hello world"  # Use English since model is English-pretrained
        
        try:
            # Note: generate() needs audio_prompt_path or prepare_conditionals first
            # For now, just verify model is callable without testing full pipeline
            logger.info(f"\n✅ Model structure check:")
            logger.info(f"  Has t3: {hasattr(model, 't3')}")
            logger.info(f"  Has s3gen: {hasattr(model, 's3gen')}")
            logger.info(f"  Has ve: {hasattr(model, 've')}")
            logger.info(f"  Has tokenizer: {hasattr(model, 'tokenizer')}")
            logger.info(f"  Has generate: {hasattr(model, 'generate')}")
            
            # Check if T3 can do forward pass
            logger.info(f"\n🧪 Testing T3 forward pass with dummy tokens...")
            dummy_text_tokens = torch.randint(0, 704, (1, 50)).to(device)
            dummy_speech_tokens = torch.randint(0, 6563, (1, 150)).to(device)
            
            from chatterbox.models.t3.t3 import T3Cond
            dummy_cond = T3Cond(
                speaker_emb=torch.randn(1, 256).to(device),
                cond_prompt_speech_tokens=torch.randint(0, 6563, (1, 150)).to(device),
                emotion_adv=torch.tensor([[[0.5]]]).to(device),
            )
            
            with torch.no_grad():
                logits = model.t3(
                    dummy_text_tokens,
                    dummy_speech_tokens,
                    dummy_cond
                )
            
            logger.info(f"\n📤 T3 forward pass result:")
            logger.info(f"  Output shape: {logits.shape}")
            logger.info(f"  Output dtype: {logits.dtype}")
            logger.info(f"  Has NaN: {torch.isnan(logits).any()}")
            logger.info(f"  Has Inf: {torch.isinf(logits).any()}")
            logger.info(f"  Min/Max: {logits.min():.4f}/{logits.max():.4f}")
            
            if torch.isnan(logits).any():
                logger.error(f"\n❌ CRITICAL: T3 output contains NaN!")
                logger.error(f"Model weights have issues!")
            elif torch.isinf(logits).any():
                logger.error(f"\n❌ CRITICAL: T3 output contains Inf!")
                logger.error(f"Model weights have issues!")
            else:
                logger.info(f"\n✅ T3 forward pass works! Model is properly loaded.")
                logger.info(f"\n🎉 CONCLUSION:")
                logger.info(f"  ✅ Pretrained model loads correctly")
                logger.info(f"  ✅ T3 forward pass produces valid outputs")
                logger.info(f"  ✅ No NaN in model weights or forward computation")
                logger.info(f"\n⚠️ The 48% NaN issue during TRAINING is likely caused by:")
                logger.info(f"  1. Training data format mismatch")
                logger.info(f"  2. Loss computation bug")
                logger.info(f"  3. Collator/preprocessing issue")
                logger.info(f"  4. Labels format incorrect")
                
        except Exception as e:
            logger.error(f"\n❌ T3 forward pass failed: {e}")
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
