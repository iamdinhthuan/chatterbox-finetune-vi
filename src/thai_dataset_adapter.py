"""
Thai GigaSpeech2 Dataset Adapter for Chatterbox Fine-tuning

This module provides an adapter to use the Thai GigaSpeech2 streaming dataset
with Chatterbox's fine-tuning scripts.
"""

from typing import Iterator, Dict, Any
import datasets
from datasets import IterableDataset, Features, Audio, Value


def create_thai_dataset_for_chatterbox(
    gigaspeech_dataset, 
    transcripts: Dict[str, str], 
    max_samples: int = None
) -> IterableDataset:
    """
    Create a HuggingFace IterableDataset compatible with Chatterbox fine-tuning
    from Thai GigaSpeech2 streaming data.
    
    Args:
        gigaspeech_dataset: The original streaming dataset from GigaSpeech2
        transcripts: Dictionary mapping segment IDs to transcripts
        max_samples: Optional limit on number of samples
        
    Returns:
        IterableDataset with 'audio' and 'text' columns as expected by Chatterbox
    """
    
    def generator():
        count = 0
        for sample in gigaspeech_dataset:
            # Extract segment ID
            segment_id = None
            key = sample.get('__key__', '')
            if key:
                parts = key.split('/')
                if parts:
                    segment_id = parts[-1]
            
            # Get transcript
            if segment_id and segment_id in transcripts:
                transcript = transcripts[segment_id]
            else:
                continue  # Skip samples without transcripts
            
            # Convert to Chatterbox expected format
            # The 'wav' field from GigaSpeech2 becomes 'audio'
            # The transcript becomes 'text'
            yield {
                'audio': sample.get('wav', {}),  # Already has 'array' and 'sampling_rate'
                'text': transcript
            }
            
            count += 1
            if max_samples and count >= max_samples:
                break
    
    # Define the features schema
    features = Features({
        'audio': Audio(sampling_rate=16000),  # GigaSpeech2 is 16kHz
        'text': Value('string')
    })
    
    # Create the IterableDataset
    return IterableDataset.from_generator(
        generator,
        features=features
    )


def load_thai_dataset_for_training(max_samples: int = None):
    """
    Complete pipeline to load Thai GigaSpeech2 for Chatterbox training.
    
    Args:
        max_samples: Optional limit on samples (useful for testing)
        
    Returns:
        IterableDataset ready for Chatterbox fine-tuning
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import csv
    
    # Load the streaming dataset
    dataset = load_dataset(
        "speechcolab/gigaspeech2",
        data_files={'train': 'data/th/train/*.tar.gz'},
        split='train',
        streaming=True
    )
    
    # Load transcripts
    try:
        tsv_path = hf_hub_download(
            repo_id="speechcolab/gigaspeech2",
            filename="data/th/train_refined.tsv",
            repo_type="dataset"
        )
        
        transcripts = {}
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    segment_id = row[0].strip()
                    transcript = row[1].strip()
                    transcripts[segment_id] = transcript
        
        print(f"Loaded {len(transcripts)} Thai transcripts")
    except Exception as e:
        print(f"Failed to load transcripts: {e}")
        transcripts = {}
    
    # Create the adapted dataset
    return create_thai_dataset_for_chatterbox(dataset, transcripts, max_samples)


# Example usage in fine-tuning script:
if __name__ == "__main__":
    # Test the adapter
    print("Testing Thai dataset adapter...")
    
    # Load a small sample
    thai_dataset = load_thai_dataset_for_training(max_samples=5)
    
    # Iterate through samples
    for i, sample in enumerate(thai_dataset):
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample['text'][:50]}...")
        print(f"  Audio shape: {sample['audio']['array'].shape}")
        print(f"  Sample rate: {sample['audio']['sampling_rate']}")