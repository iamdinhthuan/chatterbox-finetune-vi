"""
Fast dataset loader for pre-computed embeddings and tokens

This dataset loads from .pt files created by preprocess_dataset.py
Much faster than computing embeddings on-the-fly during training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)


class PrecomputedDataset(Dataset):
    """
    Fast dataset that loads pre-computed embeddings and tokens from .pt files
    
    Args:
        preprocessed_dir: Directory containing .pt files from preprocess_dataset.py
        max_text_len: Maximum text token length (optional, for additional filtering)
        max_speech_len: Maximum speech token length (optional, for additional filtering)
    """
    
    def __init__(
        self,
        preprocessed_dir: str,
        max_text_len: Optional[int] = None,
        max_speech_len: Optional[int] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        
        # Find all .pt files
        self.data_files = sorted(list(self.preprocessed_dir.glob("*.pt")))
        
        # Filter out summary file
        self.data_files = [f for f in self.data_files if 'summary' not in f.name]
        
        if len(self.data_files) == 0:
            raise ValueError(f"No .pt files found in {preprocessed_dir}")
        
        logger.info(f"Found {len(self.data_files)} preprocessed files in {preprocessed_dir}")
        
        # Load and validate first sample
        try:
            sample = torch.load(self.data_files[0])
            required_keys = [
                'text_tokens', 'text_token_lens',
                'speech_tokens', 'speech_token_lens',
                't3_cond_speaker_emb', 't3_cond_prompt_speech_tokens',
                't3_cond_emotion_adv'
            ]
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing required key: {key}")
            logger.info(f"✅ Preprocessed data validation passed")
        except Exception as e:
            raise ValueError(f"Invalid preprocessed data format: {e}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        """
        Load pre-computed data from .pt file
        
        Returns:
            Dictionary with all required fields for training
        """
        try:
            data_file = self.data_files[idx]
            data = torch.load(data_file)
            
            # Optional: Filter by length
            if self.max_text_len and data['text_token_lens'].item() > self.max_text_len:
                return None
            
            if self.max_speech_len and data['speech_token_lens'].item() > self.max_speech_len:
                return None
            
            # Return in the format expected by training
            return {
                'text_tokens': data['text_tokens'],
                'text_token_lens': data['text_token_lens'],
                'speech_tokens': data['speech_tokens'],
                'speech_token_lens': data['speech_token_lens'],
                't3_cond_speaker_emb': data['t3_cond_speaker_emb'],
                't3_cond_prompt_speech_tokens': data['t3_cond_prompt_speech_tokens'],
                't3_cond_emotion_adv': data['t3_cond_emotion_adv'],
            }
            
        except Exception as e:
            logger.error(f"Error loading {self.data_files[idx]}: {e}")
            return None
    
    def get_metadata(self, idx):
        """Get metadata (text, audio_path) for a sample"""
        try:
            data = torch.load(self.data_files[idx])
            return {
                'text': data.get('text', ''),
                'audio_path': data.get('audio_path', ''),
            }
        except:
            return None


def collate_fn(batch):
    """
    Collate function for PrecomputedDataset
    Same as the one in finetune_t3_thai.py
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return {}
    
    # Stack all tensors
    return {
        'text_tokens': torch.nn.utils.rnn.pad_sequence(
            [item['text_tokens'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'text_token_lens': torch.stack([item['text_token_lens'] for item in batch]),
        'speech_tokens': torch.nn.utils.rnn.pad_sequence(
            [item['speech_tokens'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'speech_token_lens': torch.stack([item['speech_token_lens'] for item in batch]),
        't3_cond_speaker_emb': torch.stack([item['t3_cond_speaker_emb'] for item in batch]),
        't3_cond_prompt_speech_tokens': torch.stack([item['t3_cond_prompt_speech_tokens'] for item in batch]),
        't3_cond_emotion_adv': torch.stack([item['t3_cond_emotion_adv'] for item in batch]),
    }
