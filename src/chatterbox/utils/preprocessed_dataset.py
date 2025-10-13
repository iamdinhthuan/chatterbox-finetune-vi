"""
Dataset class for loading preprocessed .pt files (2-4x faster training)
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Union

logger = logging.getLogger(__name__)


class PreprocessedDataset(Dataset):
    """
    Dataset that loads preprocessed .pt files instead of raw audio.
    
    This is 2-4x faster because:
    - No audio loading
    - No resampling
    - No tokenization
    - No voice encoding
    - All features pre-computed offline
    """
    
    def __init__(self, preprocessed_dir: str, max_text_len: int = 256, max_speech_len: int = 4096, split: str = 'train', eval_split_size: float = 0.01):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        
        # Load metadata
        metadata_path = self.preprocessed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        all_samples = self.metadata['samples']
        
        # Split into train/val based on split parameter
        if split == 'train':
            # Use first (1 - eval_split_size) samples for training
            split_idx = int(len(all_samples) * (1 - eval_split_size))
            self.samples = all_samples[:split_idx]
            logger.info(f"Loaded {len(self.samples)}/{len(all_samples)} preprocessed samples for TRAINING from {preprocessed_dir}")
        elif split == 'val' or split == 'eval':
            # Use last eval_split_size samples for validation
            split_idx = int(len(all_samples) * (1 - eval_split_size))
            self.samples = all_samples[split_idx:]
            logger.info(f"Loaded {len(self.samples)}/{len(all_samples)} preprocessed samples for VALIDATION from {preprocessed_dir}")
        else:
            # Use all samples (backward compatibility)
            self.samples = all_samples
            logger.info(f"Loaded {len(self.samples)} preprocessed samples from {preprocessed_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, str]]]:
        """
        Load preprocessed sample from .pt file
        """
        try:
            sample_info = self.samples[idx]
            pt_file = self.preprocessed_dir / sample_info['pt_file']
            
            # Load preprocessed data
            data = torch.load(pt_file, map_location='cpu')
            
            # Validate
            if data['text_tokens'].shape[0] > self.max_text_len:
                logger.warning(f"Text too long in {pt_file}: {data['text_tokens'].shape[0]} > {self.max_text_len}")
                return None
            
            if data['speech_tokens'].shape[0] > self.max_speech_len:
                logger.warning(f"Speech too long in {pt_file}: {data['speech_tokens'].shape[0]} > {self.max_speech_len}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading preprocessed sample {idx}: {e}")
            return None


def collate_fn_preprocessed(batch):
    """
    Collate function for preprocessed dataset.
    Returns format compatible with T3 training.
    
    Creates labels for next-token prediction:
    - labels_text: text_tokens shifted by 1 (predict tokens 1..EOS from BOS..N-1)
    - labels_speech: speech_tokens shifted by 1
    - Padding positions masked with -100
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    text_token_lens = torch.stack([item['text_token_lens'] for item in batch])
    speech_tokens = torch.stack([item['speech_tokens'] for item in batch])
    speech_token_lens = torch.stack([item['speech_token_lens'] for item in batch])
    
    # Create labels for next-token prediction
    # Labels are tokens[1:] with padding masked to -100
    IGNORE_ID = -100
    
    # Text labels: predict tokens 1..EOS from inputs BOS..N-1
    labels_text = text_tokens[:, 1:].clone()  # (B, S-1)
    # Mask padding positions
    text_max_len = text_tokens.size(1) - 1
    for i in range(len(batch)):
        actual_len = text_token_lens[i].item()
        if actual_len > 1:  # Need at least BOS + 1 token
            # Mask positions after EOS
            labels_text[i, actual_len-1:] = IGNORE_ID
    
    # Speech labels: predict tokens 1..EOS from inputs BOS..N-1  
    labels_speech = speech_tokens[:, 1:].clone()  # (B, S-1)
    # Mask padding positions
    speech_max_len = speech_tokens.size(1) - 1
    for i in range(len(batch)):
        actual_len = speech_token_lens[i].item()
        if actual_len > 1:  # Need at least BOS + 1 token
            # Mask positions after EOS
            labels_speech[i, actual_len-1:] = IGNORE_ID
    
    return {
        'text_tokens': text_tokens,
        'text_token_lens': text_token_lens,
        'speech_tokens': speech_tokens,
        'speech_token_lens': speech_token_lens,
        't3_cond_speaker_emb': torch.stack([item['t3_cond_speaker_emb'] for item in batch]),
        't3_cond_prompt_speech_tokens': torch.stack([item['t3_cond_prompt_speech_tokens'] for item in batch]),
        't3_cond_emotion_adv': torch.stack([item['t3_cond_emotion_adv'] for item in batch]),
        'labels_text': labels_text,        # ✅ Added!
        'labels_speech': labels_speech,    # ✅ Added!
        'labels': labels_speech,           # For Trainer compatibility
    }
