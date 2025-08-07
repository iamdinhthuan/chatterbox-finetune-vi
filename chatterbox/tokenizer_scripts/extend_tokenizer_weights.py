#!/usr/bin/env python3
"""
Script to extend T3 model weights for a larger text vocabulary.
This handles resizing text_emb and text_head layers when extending the tokenizer.
"""

import torch
import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import argparse
# from beam import function, Image, Volume  # Removed for local training

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_safetensors(filepath):
    """Load weights from safetensors file"""
    weights = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def extend_embedding_layer(old_weight, new_vocab_size, init_method="normal"):
    """
    Extend an embedding layer weight matrix to support more tokens.
    
    Args:
        old_weight: Original weight tensor of shape [old_vocab_size, embed_dim]
        new_vocab_size: New vocabulary size
        init_method: How to initialize new token embeddings ("normal", "zero", "mean")
    
    Returns:
        New weight tensor of shape [new_vocab_size, embed_dim]
    """
    old_vocab_size, embed_dim = old_weight.shape
    
    if new_vocab_size <= old_vocab_size:
        logger.warning(f"New vocab size {new_vocab_size} <= old vocab size {old_vocab_size}")
        return old_weight[:new_vocab_size]
    
    # Create new weight matrix
    new_weight = torch.zeros(new_vocab_size, embed_dim, dtype=old_weight.dtype)
    
    # Copy existing weights
    new_weight[:old_vocab_size] = old_weight
    
    # Initialize new token embeddings
    num_new_tokens = new_vocab_size - old_vocab_size
    if init_method == "normal":
        # Use same std as existing embeddings
        std = old_weight.std().item()
        new_weight[old_vocab_size:].normal_(mean=0.0, std=std)
    elif init_method == "mean":
        # Initialize as mean of existing embeddings
        mean_embedding = old_weight.mean(dim=0, keepdim=True)
        new_weight[old_vocab_size:] = mean_embedding.expand(num_new_tokens, -1)
    elif init_method == "zero":
        # Already initialized to zero
        pass
    else:
        raise ValueError(f"Unknown init_method: {init_method}")
    
    logger.info(f"Extended embedding from {old_vocab_size} to {new_vocab_size} tokens using {init_method} initialization")
    return new_weight


def extend_projection_layer(old_weight, new_vocab_size, init_method="normal"):
    """
    Extend a projection layer (output head) weight matrix.
    
    Args:
        old_weight: Original weight tensor of shape [old_vocab_size, hidden_dim]
        new_vocab_size: New vocabulary size
        init_method: How to initialize new token projections
    
    Returns:
        New weight tensor of shape [new_vocab_size, hidden_dim]
    """
    return extend_embedding_layer(old_weight, new_vocab_size, init_method)


def extend_t3_weights(
    checkpoint_path, 
    output_path, 
    new_text_vocab_size, 
    init_method="normal",
    backup_original=True
):
    """
    Extend T3 model weights to support larger text vocabulary.
    
    Args:
        checkpoint_path: Path to original checkpoint
        output_path: Path to save extended checkpoint
        new_text_vocab_size: New text vocabulary size
        init_method: Initialization method for new tokens
        backup_original: Whether to backup original file
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    # Load weights first to check if we need to do anything
    logger.info(f"Loading weights from {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        weights = load_safetensors(checkpoint_path)
    else:
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Handle nested model structure in .pt files
    model_weights = weights
    if "model" in weights:
        logger.info("Found nested 'model' key in checkpoint")
        if isinstance(weights["model"], list):
            logger.info("Model is a list, taking first element")
            model_weights = weights["model"][0]
        else:
            model_weights = weights["model"]
    
    # Check current sizes
    text_emb_weight = model_weights.get("text_emb.weight")
    text_head_weight = model_weights.get("text_head.weight")
    
    if text_emb_weight is None or text_head_weight is None:
        logger.error("Could not find text_emb.weight or text_head.weight in checkpoint")
        logger.info(f"Available keys: {list(model_weights.keys())[:10]}...")
        return False
    
    old_text_vocab_size = text_emb_weight.shape[0]
    logger.info(f"Current text vocab size: {old_text_vocab_size}")
    logger.info(f"Target text vocab size: {new_text_vocab_size}")
    
    if new_text_vocab_size == old_text_vocab_size:
        logger.info("No extension needed - sizes match")
        return True
    
    # Backup original if requested
    if backup_original and checkpoint_path.exists():
        backup_path = checkpoint_path.with_suffix(f".backup{checkpoint_path.suffix}")
        if not backup_path.exists():
            logger.info(f"Creating backup: {backup_path}")
            if checkpoint_path.suffix == ".safetensors":
                # For safetensors, just copy the file
                import shutil
                shutil.copy2(checkpoint_path, backup_path)
            else:
                torch.save(weights, backup_path)
    
    # Extend text embedding layer
    logger.info("Extending text_emb.weight...")
    model_weights["text_emb.weight"] = extend_embedding_layer(
        text_emb_weight, new_text_vocab_size, init_method
    )
    
    # Extend text projection head
    logger.info("Extending text_head.weight...")
    model_weights["text_head.weight"] = extend_projection_layer(
        text_head_weight, new_text_vocab_size, init_method
    )
    
    # Save extended weights
    logger.info(f"Saving extended weights to {output_path}")
    if output_path.suffix == ".safetensors":
        # For safetensors, we need to save the flattened weights
        if "model" in weights:
            save_file(model_weights, output_path)
        else:
            save_file(weights, output_path)
    else:
        torch.save(weights, output_path)
    
    logger.info("Successfully extended T3 weights!")
    return True


# Removed Beam-specific code for local training
CHATTERBOX_PROJECT = "./chatterbox-project"


def main():
    parser = argparse.ArgumentParser(description="Extend T3 model weights for larger vocabulary")
    parser.add_argument("checkpoint_path", help="Path to original checkpoint")
    parser.add_argument("--output_path", help="Path to save extended checkpoint (default: same as input)")
    parser.add_argument("--new_text_vocab_size", type=int, required=True, help="New text vocabulary size")
    parser.add_argument("--init_method", choices=["normal", "zero", "mean"], default="normal",
                        help="Initialization method for new tokens")
    parser.add_argument("--no_backup", action="store_true", help="Don't backup original file")
    
    args = parser.parse_args()
    
    output_path = args.output_path or args.checkpoint_path
    
    # Run locally (removed Beam option)
    success = extend_t3_weights(
        checkpoint_path=args.checkpoint_path,
        output_path=output_path,
        new_text_vocab_size=args.new_text_vocab_size,
        init_method=args.init_method,
        backup_original=not args.no_backup
    )
    if success:
        print(f"✅ Successfully extended weights to support {args.new_text_vocab_size} text tokens")
    else:
        print("❌ Failed to extend weights")
        exit(1)


if __name__ == "__main__":
    main()

# Example usage:
# from extend_tokenizer_weights import extend_t3_weights
# result = extend_t3_weights("t3_cfg.safetensors", "t3_cfg_extended.safetensors", 2000)