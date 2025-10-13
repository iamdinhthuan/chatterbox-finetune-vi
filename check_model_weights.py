"""
Check if model weights contain NaN/Inf

Usage:
    python check_model_weights.py --checkpoint checkpoints/vietnamese/checkpoint-5000
"""
import argparse
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_checkpoint(checkpoint_path: Path):
    """Check model checkpoint for NaN/Inf"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Checking checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}\n")
    
    # Load model weights
    model_file = checkpoint_path / "pytorch_model.bin"
    if not model_file.exists():
        model_file = checkpoint_path / "model.safetensors"
    
    if not model_file.exists():
        logger.error(f"Model file not found in {checkpoint_path}")
        return
    
    logger.info(f"Loading weights from {model_file.name}...")
    
    if model_file.suffix == '.bin':
        try:
            state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        except:
            state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
    else:
        # safetensors
        from safetensors.torch import load_file
        state_dict = load_file(model_file)
    
    logger.info(f"Loaded {len(state_dict)} parameters\n")
    
    # Check each parameter
    nan_params = []
    inf_params = []
    zero_params = []
    
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        
        has_nan = torch.isnan(param).any()
        has_inf = torch.isinf(param).any()
        is_zero = (param == 0).all()
        
        if has_nan:
            nan_params.append(name)
            logger.warning(f"❌ NaN detected in: {name}")
            logger.warning(f"   Shape: {param.shape}")
            logger.warning(f"   NaN count: {torch.isnan(param).sum().item()}/{param.numel()}")
        
        if has_inf:
            inf_params.append(name)
            logger.warning(f"❌ Inf detected in: {name}")
            logger.warning(f"   Shape: {param.shape}")
            logger.warning(f"   Inf count: {torch.isinf(param).sum().item()}/{param.numel()}")
        
        if is_zero:
            zero_params.append(name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 SUMMARY:")
    logger.info(f"{'='*60}")
    logger.info(f"Total parameters: {len(state_dict)}")
    logger.info(f"Parameters with NaN: {len(nan_params)}")
    logger.info(f"Parameters with Inf: {len(inf_params)}")
    logger.info(f"Parameters all zeros: {len(zero_params)}")
    
    if nan_params:
        logger.warning(f"\n⚠️ NaN parameters:")
        for name in nan_params[:10]:
            logger.warning(f"  - {name}")
        if len(nan_params) > 10:
            logger.warning(f"  ... and {len(nan_params)-10} more")
    
    if inf_params:
        logger.warning(f"\n⚠️ Inf parameters:")
        for name in inf_params[:10]:
            logger.warning(f"  - {name}")
        if len(inf_params) > 10:
            logger.warning(f"  ... and {len(inf_params)-10} more")
    
    if zero_params:
        logger.info(f"\nℹ️ All-zero parameters (might be unused):")
        for name in zero_params[:10]:
            logger.info(f"  - {name}")
        if len(zero_params) > 10:
            logger.info(f"  ... and {len(zero_params)-10} more")
    
    if not nan_params and not inf_params:
        logger.info(f"\n✅ Model weights look healthy!")
    else:
        logger.error(f"\n❌ Model weights contain NaN/Inf - training may be unstable!")
        logger.error(f"\nRecommendation:")
        logger.error(f"  1. Load from earlier checkpoint")
        logger.error(f"  2. Reduce learning rate")
        logger.error(f"  3. Enable gradient clipping")
    
    logger.info(f"\n{'='*60}\n")


def check_training_state(checkpoint_path: Path):
    """Check training state (optimizer, scheduler)"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Checking training state...")
    logger.info(f"{'='*60}\n")
    
    # Check optimizer state
    optimizer_file = checkpoint_path / "optimizer.pt"
    if optimizer_file.exists():
        logger.info("Loading optimizer state...")
        try:
            optimizer_state = torch.load(optimizer_file, map_location='cpu', weights_only=False)
        except:
            logger.warning("Failed to load optimizer state")
            return
        
        # Check optimizer state for NaN
        if 'state' in optimizer_state:
            nan_found = False
            for param_id, state in optimizer_state['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any():
                            logger.warning(f"❌ NaN in optimizer state: param {param_id}, key {key}")
                            nan_found = True
                        if torch.isinf(value).any():
                            logger.warning(f"❌ Inf in optimizer state: param {param_id}, key {key}")
                            nan_found = True
            
            if not nan_found:
                logger.info("✅ Optimizer state looks healthy")
    else:
        logger.info("No optimizer state found (might be saved separately)")
    
    logger.info(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Check model weights for NaN/Inf")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint directory")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    check_checkpoint(checkpoint_path)
    check_training_state(checkpoint_path)


if __name__ == "__main__":
    main()
