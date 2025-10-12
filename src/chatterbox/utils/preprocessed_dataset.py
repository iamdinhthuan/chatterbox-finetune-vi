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
    
    def __init__(self, preprocessed_dir: str, max_text_len: int = 256, max_speech_len: int = 4096):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        
        # Load metadata
        metadata_path = self.preprocessed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata['samples']
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
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Preprocessed data already has correct format, just stack tensors
    # No padding needed because dataloader will handle variable lengths
    
    return {
        'text_tokens': torch.stack([item['text_tokens'] for item in batch]),
        'text_token_lens': torch.stack([item['text_token_lens'] for item in batch]),
        'speech_tokens': torch.stack([item['speech_tokens'] for item in batch]),
        'speech_token_lens': torch.stack([item['speech_token_lens'] for item in batch]),
        't3_cond_speaker_emb': torch.stack([item['t3_cond_speaker_emb'] for item in batch]),
        't3_cond_prompt_speech_tokens': torch.stack([item['t3_cond_prompt_speech_tokens'] for item in batch]),
        't3_cond_emotion_adv': torch.stack([item['t3_cond_emotion_adv'] for item in batch]),
    }
