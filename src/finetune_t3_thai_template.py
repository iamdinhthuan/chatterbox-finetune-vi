#!/usr/bin/env python
"""
Fine-tune T3 model on Thai GigaSpeech2 dataset
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_t3 import main, DataArguments, ModelArguments, CustomTrainingArguments
from transformers import HfArgumentParser
from thai_dataset_adapter import load_thai_dataset_for_training


def main_thai():
    """Main function with Thai dataset integration"""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Override to use Thai dataset
    print("Loading Thai GigaSpeech2 dataset...")
    
    # For training, you might want to remove the max_samples limit
    # thai_dataset = load_thai_dataset_for_training()  # Full dataset
    thai_dataset = load_thai_dataset_for_training(max_samples=10000)  # Limited for testing
    
    # Inject the dataset into the training pipeline
    # This requires a small modification to the original main() function
    # For now, let's save the dataset and use it with the standard script
    
    # Option: Save as a local dataset that can be loaded
    # thai_dataset.save_to_disk(training_args.output_dir / "thai_dataset")
    
    # For immediate use, we'll need to modify the fine-tuning script
    # to accept pre-loaded datasets
    
    print(f"Thai dataset loaded successfully!")
    print("To use this dataset, run:")
    print(f"python finetune_t3.py --dataset_name local_thai_dataset --train_split_name train ...")


if __name__ == "__main__":
    main_thai()