"""
Script train đơn giản cho Vietnamese TTS
Chỉ cần: python train.py --csv metadata.csv --audio_dir ./
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import HfArgumentParser
from src.finetune_t3_thai import (
    ModelArguments,
    DataArguments,
    CustomTrainingArguments,
    run_training
)


def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese TTS")

    # Data arguments - support both single CSV and separate train/val CSVs
    parser.add_argument("--csv", type=str, help="Path to metadata CSV file (will be split for train/val)")
    parser.add_argument("--train_csv", type=str, help="Path to train metadata CSV file")
    parser.add_argument("--val_csv", type=str, help="Path to validation metadata CSV file")
    parser.add_argument("--audio_dir", type=str, default=".", help="Directory containing audio files (default: same as CSV)")

    # Optional training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vietnamese", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps (default: 5000)")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Evaluate every N steps (default: 5000)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (default: -1 for full training)")

    args = parser.parse_args()

    # Validate inputs - check if using separate train/val or single CSV
    use_separate_files = args.train_csv and args.val_csv
    use_single_file = args.csv

    if not use_separate_files and not use_single_file:
        print("❌ Error: You must provide either:")
        print("   1. --csv for single file (will be split), OR")
        print("   2. Both --train_csv and --val_csv for separate files")
        return

    if use_separate_files and use_single_file:
        print("⚠️  Warning: Both --csv and --train_csv/--val_csv provided. Using separate files.")

    # Validate file existence
    if use_separate_files:
        train_csv_path = Path(args.train_csv)
        val_csv_path = Path(args.val_csv)

        if not train_csv_path.exists():
            print(f"❌ Train CSV file not found: {args.train_csv}")
            return

        if not val_csv_path.exists():
            print(f"❌ Validation CSV file not found: {args.val_csv}")
            return
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌ CSV file not found: {args.csv}")
            return

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {args.audio_dir}")
        return
    
    # Check Vietnamese tokenizer
    tokenizer_path = Path("VietnameseTokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        print("❌ Vietnamese tokenizer not found!")
        print("   Please run: python train_tokenizer_from_corpus.py metadata.csv")
        return
    
    print("="*80)
    print("VIETNAMESE TTS TRAINING")
    print("="*80)

    if use_separate_files:
        print(f"\n📁 Train CSV: {train_csv_path}")
        print(f"📁 Val CSV: {val_csv_path}")
    else:
        print(f"\n📁 CSV file: {csv_path}")

    print(f"📁 Audio directory: {audio_dir}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"🔄 Epochs: {args.epochs}")
    if args.max_steps > 0:
        print(f"⚡ Max steps: {args.max_steps} (will override epochs)")
    print(f"💾 Save every: {args.save_steps} steps")
    print(f"📊 Eval every: {args.eval_steps} steps")
    print("="*80 + "\n")

    # Count samples
    if use_separate_files:
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
            num_train_samples = len(train_lines) - 1  # Exclude header

        with open(val_csv_path, 'r', encoding='utf-8') as f:
            val_lines = f.readlines()
            num_val_samples = len(val_lines) - 1  # Exclude header

        print(f"📊 Found {num_train_samples} training samples")
        print(f"📊 Found {num_val_samples} validation samples")

        # Check a few audio files from train set
        print("\n🔍 Checking training audio files...")
        missing_count = 0
        for i, line in enumerate(train_lines[1:6]):  # Check first 5
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = parts[0]
                audio_path = audio_dir / audio_file
                if audio_path.exists():
                    print(f"  ✓ {audio_file}")
                else:
                    print(f"  ✗ {audio_file} - NOT FOUND")
                    missing_count += 1
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_samples = len(lines) - 1  # Exclude header

        print(f"📊 Found {num_samples} samples in CSV")

        # Check a few audio files
        print("\n🔍 Checking audio files...")
        missing_count = 0
        for i, line in enumerate(lines[1:6]):  # Check first 5
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = parts[0]
                audio_path = audio_dir / audio_file
                if audio_path.exists():
                    print(f"  ✓ {audio_file}")
                else:
                    print(f"  ✗ {audio_file} - NOT FOUND")
                    missing_count += 1

    if missing_count > 0:
        response = input(f"\n⚠️  Some audio files not found. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    print("\n🚀 Starting training...\n")
    
    # Create model arguments
    model_args = ModelArguments(
        model_name_or_path="tel4vn/chatterxbox",
        cache_dir="./cache",
        freeze_voice_encoder=True,
        freeze_s3gen=True,
        tokenizer_path=str(tokenizer_path),
    )
    
    # Create data arguments
    if use_separate_files:
        data_args = DataArguments(
            train_metadata_file=str(train_csv_path),
            val_metadata_file=str(val_csv_path),
            audio_dir=str(audio_dir),
            dataset_dir=None,
            dataset_name=None,
            eval_split_size=0.0,  # Not used when separate files provided
            max_text_len=256,
            max_speech_len=1200,
            audio_prompt_duration_s=3.0,
            preprocessing_num_workers=8,
            ignore_verifications=True,
            use_streaming=False,
        )
    else:
        data_args = DataArguments(
            metadata_file=str(csv_path),
            audio_dir=str(audio_dir),
            dataset_dir=None,
            dataset_name=None,
            eval_split_size=0.01,
            max_text_len=256,
            max_speech_len=1200,
            audio_prompt_duration_s=3.0,
            preprocessing_num_workers=12,
            ignore_verifications=True,
            use_streaming=False,
        )

    # Create training arguments
    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,

        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.lr,
        warmup_steps=5000,
        lr_scheduler_type="cosine",

        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        data_seed=42,
        bf16=True,
        dataloader_num_workers=12,
        dataloader_persistent_workers=True,
        seed=42,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    
    # Run training
    try:
        run_training(model_args, data_args, training_args)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED!")
        print("="*80)
        print(f"\n📁 Model saved at: {args.output_dir}")
        print(f"📊 Logs at: {args.output_dir}/logs")
        print("\n💡 To test the model:")
        print(f"   python test.py --model {args.output_dir} --text 'Xin chào'")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

