"""
Dataset with on-the-fly caching for embeddings and tokens

First epoch: Compute embeddings and save to cache
Following epochs: Load from cache (4-5x faster!)

Usage:
    from chatterbox.utils.cached_dataset import CachedSpeechFineTuningDataset
    
    train_dataset = CachedSpeechFineTuningDataset(
        data_args=data_args,
        t3_config=t3_config,
        hf_dataset=hf_dataset,
        cache_dir="./cache",  # Cache directory
        device="cuda",  # Use GPU for computing embeddings
    )
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Union
import logging
import hashlib

logger = logging.getLogger(__name__)


class CachedSpeechFineTuningDataset:
    """
    Dataset with automatic caching of embeddings and tokens
    
    - First epoch: Compute and save to cache (slow)
    - Following epochs: Load from cache (fast - 4-5x speedup!)
    - Compatible with existing SpeechFineTuningDataset interface
    """
    
    def __init__(
        self,
        data_args,
        t3_config,
        hf_dataset,
        is_hf_format: bool,
        model_dir: str,
        cache_dir: str,
        m_paths: Optional[dict] = None,
        device: str = "cuda",
    ):
        """
        Args:
            cache_dir: Directory to store cached embeddings
            device: Device for computing embeddings (cuda/cpu)
            Other args same as SpeechFineTuningDataset
        """
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from finetune_t3_thai import SpeechFineTuningDataset
        
        # Store cache config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Create underlying dataset
        self.base_dataset = SpeechFineTuningDataset(
            data_args=data_args,
            t3_config=t3_config,
            hf_dataset=hf_dataset,
            is_hf_format=is_hf_format,
            model_dir=model_dir,
            m_paths=m_paths,
            device=device,
        )
        
        # Track cache hits/misses for logging
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"📦 CachedDataset initialized")
        logger.info(f"   Cache dir: {self.cache_dir}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Dataset size: {len(self.base_dataset)}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for a given index"""
        return self.cache_dir / f"cache_{idx:06d}.pt"
    
    def _compute_and_cache(self, idx: int) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        """Compute embeddings and save to cache"""
        # Compute using base dataset
        item = self.base_dataset[idx]
        
        if item is None:
            return None
        
        # Save to cache
        cache_path = self._get_cache_path(idx)
        try:
            # Move all tensors to CPU before saving
            cache_item = {
                'text_tokens': item['text_tokens'].cpu(),
                'text_token_lens': item['text_token_lens'].cpu(),
                'speech_tokens': item['speech_tokens'].cpu(),
                'speech_token_lens': item['speech_token_lens'].cpu(),
                't3_cond_speaker_emb': item['t3_cond_speaker_emb'].cpu(),
                't3_cond_prompt_speech_tokens': item['t3_cond_prompt_speech_tokens'].cpu(),
                't3_cond_emotion_adv': item['t3_cond_emotion_adv'].cpu() if torch.is_tensor(item['t3_cond_emotion_adv']) else torch.tensor(item['t3_cond_emotion_adv']),
            }
            torch.save(cache_item, cache_path)
            self.cache_misses += 1
        except Exception as e:
            logger.warning(f"Failed to save cache for index {idx}: {e}")
        
        return item
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        """
        Get item with caching
        - If cache exists: Load from cache (fast)
        - If no cache: Compute and save to cache (slow)
        """
        cache_path = self._get_cache_path(idx)
        
        # Try to load from cache
        if cache_path.exists():
            try:
                item = torch.load(cache_path)
                self.cache_hits += 1
                return item
            except Exception as e:
                logger.warning(f"Failed to load cache for index {idx}: {e}. Recomputing...")
                # Delete corrupted cache
                cache_path.unlink(missing_ok=True)
        
        # Cache miss - compute and save
        return self._compute_and_cache(idx)
    
    def get_cache_stats(self):
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_accesses': total,
            'hit_rate': hit_rate,
        }
    
    def clear_cache(self):
        """Clear all cached files"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Cache cleared: {self.cache_dir}")


def log_cache_stats_callback(dataset):
    """
    Callback to log cache statistics after each epoch
    
    Usage in Trainer:
        from transformers import TrainerCallback
        
        class CacheStatsCallback(TrainerCallback):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def on_epoch_end(self, args, state, control, **kwargs):
                if hasattr(self.dataset, 'get_cache_stats'):
                    stats = self.dataset.get_cache_stats()
                    print(f"📊 Cache Stats: {stats['cache_hits']} hits, {stats['cache_misses']} misses, {stats['hit_rate']:.1f}% hit rate")
        
        trainer = Trainer(
            ...,
            callbacks=[CacheStatsCallback(train_dataset)]
        )
    """
    if hasattr(dataset, 'get_cache_stats'):
        stats = dataset.get_cache_stats()
        logger.info(f"📊 Cache Statistics:")
        logger.info(f"   Cache hits: {stats['cache_hits']}")
        logger.info(f"   Cache misses: {stats['cache_misses']}")
        logger.info(f"   Hit rate: {stats['hit_rate']:.1f}%")
