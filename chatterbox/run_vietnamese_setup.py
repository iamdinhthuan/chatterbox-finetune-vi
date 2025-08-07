#!/usr/bin/env python3
"""
Complete Vietnamese TTS Training Setup Script
This script automates the entire process from CSV data to ready-to-train model.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: list, description: str) -> bool:
    """Run a command and return success status"""
    try:
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {description}: {e}")
        return False

def check_prerequisites() -> bool:
    """Check if all required files exist"""
    logger.info("🔍 Checking prerequisites...")
    
    required_files = [
        "train.csv",
        "val.csv", 
        "chatterbox-project/chatterbox_weights/tokenizer.json",
        "chatterbox-project/chatterbox_weights/t3_cfg.safetensors",
        "chatterbox-project/chatterbox_weights/ve.safetensors",
        "chatterbox-project/chatterbox_weights/s3gen.safetensors"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info("✅ All required files found")
    return True

def main():
    print("🇻🇳 Vietnamese TTS Training Setup")
    print("=" * 50)
    print("This script will:")
    print("1. Extract Vietnamese text from CSV files")
    print("2. Create Vietnamese tokenizer")
    print("3. Merge with original tokenizer")
    print("4. Extend model weights")
    print("5. Prepare training configuration")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites check failed. Please ensure all required files exist.")
        return False
    
    # Step 1: Extract Vietnamese text corpus
    logger.info("📝 Step 1: Extracting Vietnamese text corpus...")
    success = run_command([
        sys.executable, "extract_vietnamese_text.py",
        "--train_csv", "train.csv",
        "--val_csv", "val.csv", 
        "--output", "vietnamese_text_corpus.txt",
        "--analyze"
    ], "Extract Vietnamese text corpus")
    
    if not success:
        logger.error("❌ Failed to extract text corpus")
        return False
    
    # Step 2: Run complete tokenizer setup
    logger.info("🔧 Step 2: Setting up Vietnamese tokenizer and model...")
    success = run_command([
        sys.executable, "setup_vietnamese_training.py",
        "--text_file", "vietnamese_text_corpus.txt",
        "--vocab_size", "500"
    ], "Setup Vietnamese tokenizer and model")
    
    if not success:
        logger.error("❌ Failed to setup tokenizer and model")
        return False
    
    # Step 3: Validate setup
    logger.info("✅ Step 3: Validating setup...")
    
    expected_files = [
        "tokenizer_vietnamese_new.json",
        "tokenizer_vi_merged.json", 
        "t3_cfg_vietnamese.safetensors",
        "model_path_vietnamese.json"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ Setup validation failed. Missing files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    # Step 4: Test dataset loading
    logger.info("🧪 Step 4: Testing dataset loading...")
    success = run_command([
        sys.executable, "check_dataset.py",
        "--train_csv", "train.csv",
        "--val_csv", "val.csv",
        "--max_samples", "100"
    ], "Test dataset loading")
    
    if not success:
        logger.warning("⚠️ Dataset validation had issues, but continuing...")
    
    # Final success message
    print("\n" + "=" * 60)
    print("🎉 VIETNAMESE TTS SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Generated files:")
    print("  ✅ vietnamese_text_corpus.txt - Text corpus for tokenizer")
    print("  ✅ tokenizer_vietnamese_new.json - Vietnamese tokenizer")
    print("  ✅ tokenizer_vi_merged.json - Merged tokenizer")
    print("  ✅ t3_cfg_vietnamese.safetensors - Extended model weights")
    print("  ✅ model_path_vietnamese.json - Model configuration")
    print("")
    print("🚀 READY TO START TRAINING!")
    print("=" * 60)
    print("Run the following command to start training:")
    print("")
    print("python train_vietnamese_csv.py \\")
    print("    --model_config model_path_vietnamese.json \\")
    print("    --train_csv train.csv \\")
    print("    --val_csv val.csv \\")
    print("    --output_dir checkpoints/vietnamese_tts \\")
    print("    --num_train_epochs 3 \\")
    print("    --per_device_train_batch_size 4")
    print("")
    print("📊 Monitor training with:")
    print("tensorboard --logdir checkpoints/vietnamese_tts/runs")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
