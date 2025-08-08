#!/usr/bin/env python3
"""
Vietnamese TTS Training Script using CSV Dataset
This script trains a ChatterboxTTS model on Vietnamese data using CSV files.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from finetune_t3_local import (
    ModelArguments, 
    DataArguments, 
    CustomTrainingArguments,
    run_training
)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese TTS model with CSV dataset")
    
    # Model arguments
    parser.add_argument("--model_config", type=str, default="model_path.json",
                       help="Path to model configuration JSON file")
    parser.add_argument("--local_model_dir", type=str, default=None,
                       help="Path to local model directory")
    
    # Data arguments
    parser.add_argument("--train_csv", type=str, default="train.csv",
                       help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="val.csv",
                       help="Path to validation CSV file")
    parser.add_argument("--max_text_len", type=int, default=256,
                       help="Maximum text token length")
    parser.add_argument("--max_speech_len", type=int, default=800,
                       help="Maximum speech token length")
    parser.add_argument("--audio_prompt_duration_s", type=float, default=3.0,
                       help="Audio prompt duration in seconds")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints/vietnamese_tts",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None,
                       help="Evaluation batch size per device (defaults to same as train batch size)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=2000,
                       help="Save frequency")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=8,
                       help="Number of batches to prefetch per worker")
    parser.add_argument("--audio_cache_dir", type=str, default=None,
                       help="Directory to cache preprocessed audio (optional)")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--do_eval", action="store_true", default=True,
                       help="Run evaluation")

    # Best model tracking arguments
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                       help="Metric to use for best model selection")
    parser.add_argument("--greater_is_better", action="store_true", default=False,
                       help="Whether higher metric values are better")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True,
                       help="Load best model at the end of training")

    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate input files
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV file not found: {args.train_csv}")
    
    if args.do_eval and args.val_csv and not os.path.exists(args.val_csv):
        logger.warning(f"Validation CSV file not found: {args.val_csv}. Disabling evaluation.")
        args.do_eval = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting Vietnamese TTS training with CSV dataset")
    logger.info(f"Training CSV: {args.train_csv}")
    logger.info(f"Validation CSV: {args.val_csv if args.do_eval else 'None'}")
    logger.info(f"Output directory: {args.output_dir}")

    if args.do_eval:
        logger.info(f"Best model tracking enabled:")
        logger.info(f"  - Metric: {args.metric_for_best_model}")
        logger.info(f"  - Greater is better: {args.greater_is_better}")
        logger.info(f"  - Load best model at end: {args.load_best_model_at_end}")
        logger.info(f"  - Save total limit: {args.save_total_limit}")
        logger.info(f"  - Train batch size: {args.per_device_train_batch_size}")
        logger.info(f"  - Dataloader workers: {args.dataloader_num_workers}")
        logger.info(f"  - Prefetch factor: {args.dataloader_prefetch_factor}")
        if args.audio_cache_dir:
            logger.info(f"  - Audio cache: {args.audio_cache_dir}")
    else:
        logger.info("Evaluation disabled - will save latest checkpoints only")
    
    # Create argument objects
    model_args = ModelArguments(
        model_config=args.model_config if os.path.exists(args.model_config) else None,
        local_model_dir=args.local_model_dir,
        cache_dir=None,
        freeze_voice_encoder=True,
        freeze_s3gen=True,
        freeze_text_embeddings=None
    )
    
    data_args = DataArguments(
        train_csv=args.train_csv,
        val_csv=args.val_csv if args.do_eval else None,
        text_column_name="transcript",
        audio_column_name="audio",
        max_text_len=args.max_text_len,
        max_speech_len=args.max_speech_len,
        audio_prompt_duration_s=args.audio_prompt_duration_s,
        preprocessing_num_workers=args.dataloader_num_workers,
        ignore_verifications=True
    )
    
    # Set eval batch size to same as train batch size if not specified
    eval_batch_size = args.per_device_eval_batch_size if args.per_device_eval_batch_size is not None else args.per_device_train_batch_size

    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps if args.do_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # Best model tracking configuration
        metric_for_best_model=args.metric_for_best_model if args.do_eval else None,
        greater_is_better=args.greater_is_better if args.do_eval else None,
        load_best_model_at_end=args.load_best_model_at_end if args.do_eval else False,
        fp16=args.fp16,
        report_to="tensorboard",
        dataloader_num_workers=args.dataloader_num_workers,
        do_train=True,
        do_eval=args.do_eval,
        dataloader_pin_memory=True,
        eval_on_start=False,
        use_torch_profiler=False,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,  # Only when multiprocessing
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,  # Only when multiprocessing
        remove_unused_columns=False,  # Important for custom datasets
        label_names=["text_tokens", "speech_tokens"]  # Specify label names
    )
    
    try:
        # Run training
        run_training(model_args, data_args, training_args, is_local=True)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
