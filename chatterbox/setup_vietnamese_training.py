#!/usr/bin/env python3
"""
Complete setup script for Vietnamese TTS training.
This script handles:
1. Creating Vietnamese tokenizer
2. Merging with original tokenizer  
3. Extending model weights
4. Preparing for training
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Add tokenizer_scripts to path
sys.path.append(str(Path(__file__).parent / "tokenizer_scripts"))

from make_new_tokenizer import create_japanese_tokenizer, analyze_tokenizer
from merge_tokenizers import merge_tokenizers
from extend_tokenizer_weights import extend_t3_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VietnameseTokenizerSetup:
    def __init__(self, 
                 text_file: str = "vietnamese_text.txt",
                 original_tokenizer: str = "chatterbox-project/chatterbox_weights/tokenizer.json",
                 model_checkpoint: str = "chatterbox-project/chatterbox_weights/t3_cfg.safetensors"):
        self.text_file = text_file
        self.original_tokenizer = original_tokenizer
        self.model_checkpoint = model_checkpoint
        self.vietnamese_tokenizer = "tokenizer_vietnamese_new.json"
        self.merged_tokenizer = "tokenizer_vi_merged.json"
        self.extended_checkpoint = "t3_cfg_vietnamese.safetensors"
        
    def step1_create_vietnamese_tokenizer(self, vocab_size: int = 500) -> bool:
        """Create Vietnamese tokenizer from text corpus"""
        logger.info("=== Step 1: Creating Vietnamese Tokenizer ===")
        
        if not os.path.exists(self.text_file):
            logger.error(f"Vietnamese text file not found: {self.text_file}")
            return False
        
        try:
            # Create Vietnamese tokenizer
            logger.info(f"Creating Vietnamese tokenizer with vocab size {vocab_size}")
            tokenizer_path = create_japanese_tokenizer(
                text_file=self.text_file,
                vocab_size=vocab_size,
                output_path=self.vietnamese_tokenizer,
                existing_tokenizer_path=self.original_tokenizer if os.path.exists(self.original_tokenizer) else None
            )
            
            # Analyze the created tokenizer
            total_tokens, special_count, vietnamese_count, other_count = analyze_tokenizer(tokenizer_path)
            logger.info(f"Vietnamese tokenizer created:")
            logger.info(f"  - Total tokens: {total_tokens}")
            logger.info(f"  - Special tokens: {special_count}")
            logger.info(f"  - Vietnamese tokens: {vietnamese_count}")
            logger.info(f"  - Other tokens: {other_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Vietnamese tokenizer: {e}")
            return False
    
    def step2_merge_tokenizers(self) -> bool:
        """Merge Vietnamese tokenizer with original tokenizer"""
        logger.info("=== Step 2: Merging Tokenizers ===")
        
        if not os.path.exists(self.original_tokenizer):
            logger.error(f"Original tokenizer not found: {self.original_tokenizer}")
            return False
            
        if not os.path.exists(self.vietnamese_tokenizer):
            logger.error(f"Vietnamese tokenizer not found: {self.vietnamese_tokenizer}")
            return False
        
        try:
            # Merge tokenizers
            logger.info(f"Merging {self.vietnamese_tokenizer} into {self.original_tokenizer}")
            merged_path = merge_tokenizers(
                tokenizer_a_path=self.original_tokenizer,
                tokenizer_b_path=self.vietnamese_tokenizer,
                output_path=self.merged_tokenizer
            )
            
            # Analyze merged tokenizer
            total_tokens, special_count, vietnamese_count, other_count = analyze_tokenizer(merged_path)
            logger.info(f"Merged tokenizer created:")
            logger.info(f"  - Total tokens: {total_tokens}")
            logger.info(f"  - Special tokens: {special_count}")
            logger.info(f"  - Vietnamese tokens: {vietnamese_count}")
            logger.info(f"  - Other tokens: {other_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge tokenizers: {e}")
            return False
    
    def step3_extend_model_weights(self) -> bool:
        """Extend model weights to match new tokenizer size"""
        logger.info("=== Step 3: Extending Model Weights ===")
        
        if not os.path.exists(self.model_checkpoint):
            logger.error(f"Model checkpoint not found: {self.model_checkpoint}")
            return False
            
        if not os.path.exists(self.merged_tokenizer):
            logger.error(f"Merged tokenizer not found: {self.merged_tokenizer}")
            return False
        
        try:
            # Get new vocab size from merged tokenizer
            with open(self.merged_tokenizer, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            new_vocab_size = len(tokenizer_data['model']['vocab'])
            logger.info(f"New vocabulary size: {new_vocab_size}")
            
            # Extend model weights
            logger.info(f"Extending model weights from {self.model_checkpoint}")
            success = extend_t3_weights(
                checkpoint_path=self.model_checkpoint,
                output_path=self.extended_checkpoint,
                new_text_vocab_size=new_vocab_size,
                init_method="normal",
                backup_original=True
            )
            
            if success:
                logger.info(f"Extended model weights saved to: {self.extended_checkpoint}")
                return True
            else:
                logger.error("Failed to extend model weights")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extend model weights: {e}")
            return False
    
    def step4_create_model_config(self) -> bool:
        """Create model configuration for training"""
        logger.info("=== Step 4: Creating Model Configuration ===")
        
        try:
            # Create model configuration
            model_config = {
                "voice_encoder_path": "chatterbox_weights/ve.safetensors",
                "t3_path": os.path.basename(self.extended_checkpoint),
                "s3gen_path": "chatterbox_weights/s3gen.safetensors", 
                "tokenizer_path": os.path.abspath(self.merged_tokenizer),
                "conds_path": "chatterbox_weights/conds.pt"
            }
            
            config_path = "model_path_vietnamese.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model configuration saved to: {config_path}")
            
            # Copy extended checkpoint to chatterbox_weights directory
            import shutil
            weights_dir = Path("chatterbox-project/chatterbox_weights")
            if weights_dir.exists():
                target_path = weights_dir / os.path.basename(self.extended_checkpoint)
                shutil.copy2(self.extended_checkpoint, target_path)
                logger.info(f"Copied extended checkpoint to: {target_path}")
                
                # Update config to use relative path
                model_config["t3_path"] = f"chatterbox_weights/{os.path.basename(self.extended_checkpoint)}"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model configuration: {e}")
            return False
    
    def run_complete_setup(self, vietnamese_vocab_size: int = 500) -> bool:
        """Run the complete setup process"""
        logger.info("🇻🇳 Starting Vietnamese TTS Training Setup")
        logger.info("=" * 60)
        
        # Step 1: Create Vietnamese tokenizer
        if not self.step1_create_vietnamese_tokenizer(vietnamese_vocab_size):
            return False
        
        # Step 2: Merge tokenizers
        if not self.step2_merge_tokenizers():
            return False
        
        # Step 3: Extend model weights
        if not self.step3_extend_model_weights():
            return False
        
        # Step 4: Create model config
        if not self.step4_create_model_config():
            return False
        
        logger.info("=" * 60)
        logger.info("✅ Vietnamese TTS Training Setup Complete!")
        logger.info("=" * 60)
        logger.info("📁 Generated files:")
        logger.info(f"  - Vietnamese tokenizer: {self.vietnamese_tokenizer}")
        logger.info(f"  - Merged tokenizer: {self.merged_tokenizer}")
        logger.info(f"  - Extended model: {self.extended_checkpoint}")
        logger.info(f"  - Model config: model_path_vietnamese.json")
        logger.info("")
        logger.info("🚀 Ready to start training!")
        logger.info("Run: python train_vietnamese_csv.py --model_config model_path_vietnamese.json")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Vietnamese TTS training")
    parser.add_argument("--text_file", default="vietnamese_text.txt",
                       help="Vietnamese text corpus file")
    parser.add_argument("--vocab_size", type=int, default=500,
                       help="Vietnamese vocabulary size to add")
    parser.add_argument("--original_tokenizer", 
                       default="chatterbox-project/chatterbox_weights/tokenizer.json",
                       help="Original tokenizer path")
    parser.add_argument("--model_checkpoint",
                       default="chatterbox-project/chatterbox_weights/t3_cfg.safetensors", 
                       help="Original model checkpoint path")
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = VietnameseTokenizerSetup(
        text_file=args.text_file,
        original_tokenizer=args.original_tokenizer,
        model_checkpoint=args.model_checkpoint
    )
    
    # Run complete setup
    success = setup.run_complete_setup(args.vocab_size)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now start training with:")
        print("python train_vietnamese_csv.py --model_config model_path_vietnamese.json")
    else:
        print("\n❌ Setup failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
