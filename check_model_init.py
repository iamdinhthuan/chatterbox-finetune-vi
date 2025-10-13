"""
Check if model is loading pretrained weights properly

Usage:
    python check_model_init.py
"""
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_model_initialization():
    """Check if model loads pretrained weights"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Checking Model Initialization")
    logger.info(f"{'='*60}\n")
    
    try:
        from chatterbox.models.t3.t3_for_causal_lm import ChatterboxT3ForCausalLM
        from chatterbox.models.t3.t3_config import T3Config
        from huggingface_hub import hf_hub_download
        
        # Download config
        logger.info("📥 Downloading model config...")
        config_path = hf_hub_download(
            repo_id="ResembleAI/chatterbox",
            filename="t3/config.json",
            cache_dir="./.cache"
        )
        
        logger.info(f"Config path: {config_path}")
        
        # Load config
        model_dir = Path(config_path).parent
        t3_config = T3Config.from_json_file(config_path)
        
        logger.info(f"Model dir: {model_dir}")
        logger.info(f"Config: {t3_config}")
        
        # Check if model files exist
        logger.info(f"\n📂 Checking model files in {model_dir}:")
        
        model_file = model_dir / "pytorch_model.bin"
        safetensors_file = model_dir / "model.safetensors"
        
        if model_file.exists():
            logger.info(f"  ✅ Found pytorch_model.bin ({model_file.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            logger.warning(f"  ❌ pytorch_model.bin not found")
        
        if safetensors_file.exists():
            logger.info(f"  ✅ Found model.safetensors ({safetensors_file.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            logger.warning(f"  ❌ model.safetensors not found")
        
        if not model_file.exists() and not safetensors_file.exists():
            logger.error(f"\n❌ CRITICAL: No model weights found!")
            logger.error(f"Model will be initialized with RANDOM weights!")
            logger.error(f"This explains the high loss and NaN issues!")
            return
        
        # Load model
        logger.info(f"\n🔧 Loading model...")
        model = ChatterboxT3ForCausalLM.from_local(
            model_dir=str(model_dir),
            config=t3_config
        )
        
        logger.info(f"Model loaded successfully")
        
        # Check if weights look pretrained (not random)
        logger.info(f"\n🔍 Checking if weights look pretrained...")
        
        # Random weights typically have std around 0.01-0.1
        # Pretrained weights typically have larger variance
        
        param_stats = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                std = param.std().item()
                mean = param.mean().item()
                param_stats.append({
                    'name': name,
                    'std': std,
                    'mean': mean,
                    'shape': param.shape
                })
        
        # Show first 10 parameters
        logger.info(f"\nFirst 10 trainable parameters:")
        for i, stat in enumerate(param_stats[:10]):
            logger.info(f"  {stat['name']}")
            logger.info(f"    Shape: {stat['shape']}, Mean: {stat['mean']:.4f}, Std: {stat['std']:.4f}")
        
        # Analyze statistics
        avg_std = sum(s['std'] for s in param_stats) / len(param_stats)
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Total trainable parameters: {len(param_stats)}")
        logger.info(f"  Average std: {avg_std:.4f}")
        
        if avg_std < 0.001:
            logger.error(f"\n❌ CRITICAL: Average std too low ({avg_std:.6f})")
            logger.error(f"Weights might be all zeros or not loaded properly!")
        elif avg_std < 0.01:
            logger.warning(f"\n⚠️ WARNING: Average std quite low ({avg_std:.4f})")
            logger.warning(f"Weights might be random initialization")
        else:
            logger.info(f"\n✅ Weights look reasonable (std={avg_std:.4f})")
        
        # Test forward pass with dummy input
        logger.info(f"\n🧪 Testing forward pass with dummy input...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        text_len = 50
        speech_len = 150
        
        dummy_input = {
            'text_tokens': torch.randint(0, 704, (batch_size, text_len)).to(device),
            'text_token_lens': torch.tensor([text_len, text_len]).to(device),
            'speech_tokens': torch.randint(0, 6563, (batch_size, speech_len)).to(device),
            'speech_token_lens': torch.tensor([speech_len, speech_len]).to(device),
            't3_cond_speaker_emb': torch.randn(batch_size, 256).to(device),
            't3_cond_prompt_speech_tokens': torch.randint(0, 6563, (batch_size, 150)).to(device),
            't3_cond_emotion_adv': torch.tensor([0.5, 0.5]).to(device),
        }
        
        with torch.no_grad():
            outputs = model(**dummy_input)
        
        loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        logger.info(f"\n📤 Dummy forward pass result:")
        logger.info(f"  Loss: {loss}")
        logger.info(f"  Is NaN: {torch.isnan(loss).any()}")
        logger.info(f"  Is Inf: {torch.isinf(loss).any()}")
        
        if torch.isnan(loss):
            logger.error(f"\n❌ CRITICAL: Model produces NaN even on dummy input!")
            logger.error(f"Model weights or architecture has serious issues!")
        elif loss.item() > 100:
            logger.error(f"\n❌ CRITICAL: Loss extremely high ({loss.item():.2f})")
            logger.error(f"Model likely not properly initialized!")
        elif loss.item() > 20:
            logger.warning(f"\n⚠️ WARNING: Loss quite high ({loss.item():.2f})")
            logger.warning(f"Model might not be pretrained")
        else:
            logger.info(f"\n✅ Model forward pass looks OK (loss={loss.item():.4f})")
        
    except Exception as e:
        logger.error(f"\n❌ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()


def main():
    check_model_initialization()


if __name__ == "__main__":
    main()
