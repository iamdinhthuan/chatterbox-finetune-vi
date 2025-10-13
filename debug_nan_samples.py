"""
Debug NaN/Inf samples from preprocessed data

Usage:
    python debug_nan_samples.py --sample_idx 100
    python debug_nan_samples.py --audio_path "wavs/vivoice_100.wav"
"""
import argparse
import torch
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_sample(preprocessed_dir: Path, sample_idx: int):
    """Inspect a specific preprocessed sample"""
    
    # Load metadata
    metadata_path = preprocessed_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if sample_idx >= len(metadata['samples']):
        logger.error(f"Sample index {sample_idx} out of range (max: {len(metadata['samples'])-1})")
        return
    
    sample_info = metadata['samples'][sample_idx]
    pt_file = preprocessed_dir / sample_info['pt_file']
    
    # Load preprocessed data
    logger.info(f"\n{'='*60}")
    logger.info(f"📄 Sample #{sample_idx}: {sample_info['pt_file']}")
    logger.info(f"{'='*60}")
    
    data = torch.load(pt_file, map_location='cpu')
    
    # Display all fields
    logger.info(f"\n🎵 Audio Info:")
    logger.info(f"  Path: {data.get('audio_path', 'N/A')}")
    logger.info(f"  Text: {data.get('text', 'N/A')}")
    
    logger.info(f"\n📝 Text Tokens:")
    text_tokens = data.get('text_tokens')
    if text_tokens is not None:
        logger.info(f"  Shape: {text_tokens.shape}")
        logger.info(f"  Length: {data.get('text_token_lens', 'N/A')}")
        logger.info(f"  Min/Max: {text_tokens.min()}/{text_tokens.max()}")
        logger.info(f"  Has NaN: {torch.isnan(text_tokens).any()}")
        logger.info(f"  Has Inf: {torch.isinf(text_tokens).any()}")
        logger.info(f"  First 10: {text_tokens[:10].tolist()}")
        logger.info(f"  Last 10: {text_tokens[-10:].tolist()}")
    
    logger.info(f"\n🎤 Speech Tokens:")
    speech_tokens = data.get('speech_tokens')
    if speech_tokens is not None:
        logger.info(f"  Shape: {speech_tokens.shape}")
        logger.info(f"  Length: {data.get('speech_token_lens', 'N/A')}")
        logger.info(f"  Min/Max: {speech_tokens.min()}/{speech_tokens.max()}")
        logger.info(f"  Has NaN: {torch.isnan(speech_tokens).any()}")
        logger.info(f"  Has Inf: {torch.isinf(speech_tokens).any()}")
        logger.info(f"  First 10: {speech_tokens[:10].tolist()}")
        logger.info(f"  Last 10: {speech_tokens[-10:].tolist()}")
    
    logger.info(f"\n🔊 Speaker Embedding:")
    speaker_emb = data.get('t3_cond_speaker_emb')
    if speaker_emb is not None:
        logger.info(f"  Shape: {speaker_emb.shape}")
        logger.info(f"  Min/Max: {speaker_emb.min():.4f}/{speaker_emb.max():.4f}")
        logger.info(f"  Mean/Std: {speaker_emb.mean():.4f}/{speaker_emb.std():.4f}")
        logger.info(f"  Has NaN: {torch.isnan(speaker_emb).any()}")
        logger.info(f"  Has Inf: {torch.isinf(speaker_emb).any()}")
    
    logger.info(f"\n🎯 Conditioning Prompt:")
    prompt_tokens = data.get('t3_cond_prompt_speech_tokens')
    if prompt_tokens is not None:
        logger.info(f"  Shape: {prompt_tokens.shape}")
        logger.info(f"  Min/Max: {prompt_tokens.min()}/{prompt_tokens.max()}")
        logger.info(f"  Has NaN: {torch.isnan(prompt_tokens).any()}")
        logger.info(f"  Has Inf: {torch.isinf(prompt_tokens).any()}")
    
    emotion = data.get('t3_cond_emotion_adv')
    if emotion is not None:
        logger.info(f"\n😊 Emotion: {emotion}")
    
    # Check for potential issues
    logger.info(f"\n⚠️ Potential Issues:")
    issues = []
    
    if text_tokens is not None:
        if torch.isnan(text_tokens).any() or torch.isinf(text_tokens).any():
            issues.append("❌ Text tokens contain NaN/Inf")
        if text_tokens.max() > 703:
            issues.append(f"⚠️ Text tokens exceed vocab size (max: {text_tokens.max()}, expected: <=703)")
    
    if speech_tokens is not None:
        if torch.isnan(speech_tokens).any() or torch.isinf(speech_tokens).any():
            issues.append("❌ Speech tokens contain NaN/Inf")
        if speech_tokens.max() > 6562:
            issues.append(f"⚠️ Speech tokens exceed vocab size (max: {speech_tokens.max()}, expected: <=6562)")
    
    if speaker_emb is not None:
        if torch.isnan(speaker_emb).any() or torch.isinf(speaker_emb).any():
            issues.append("❌ Speaker embedding contains NaN/Inf")
        if speaker_emb.std() < 0.001:
            issues.append(f"⚠️ Speaker embedding has very low variance (std: {speaker_emb.std():.6f})")
    
    if prompt_tokens is not None:
        if torch.isnan(prompt_tokens).any() or torch.isinf(prompt_tokens).any():
            issues.append("❌ Prompt tokens contain NaN/Inf")
    
    text_len = data.get('text_token_lens', 0)
    speech_len = data.get('speech_token_lens', 0)
    if isinstance(text_len, torch.Tensor):
        text_len = text_len.item()
    if isinstance(speech_len, torch.Tensor):
        speech_len = speech_len.item()
    
    if text_len > 256:
        issues.append(f"⚠️ Text too long ({text_len} > 256)")
    if speech_len > 1200:
        issues.append(f"⚠️ Speech too long ({speech_len} > 1200)")
    if text_len < 5:
        issues.append(f"⚠️ Text too short ({text_len} < 5)")
    if speech_len < 10:
        issues.append(f"⚠️ Speech too short ({speech_len} < 10)")
    
    if issues:
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info(f"  ✅ No obvious issues detected")
    
    logger.info(f"\n{'='*60}\n")


def find_by_audio_path(preprocessed_dir: Path, audio_path: str):
    """Find sample by audio path"""
    metadata_path = preprocessed_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    for idx, sample in enumerate(metadata['samples']):
        if sample['audio_path'] == audio_path or sample['audio_path'].endswith(audio_path):
            logger.info(f"Found sample at index {idx}")
            return idx
    
    logger.error(f"Audio path not found: {audio_path}")
    return None


def batch_analyze(preprocessed_dir: Path, num_samples: int = 100):
    """Analyze random samples to find patterns"""
    import random
    
    metadata_path = preprocessed_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    total_samples = len(metadata['samples'])
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Analyzing {len(sample_indices)} random samples")
    logger.info(f"{'='*60}\n")
    
    stats = {
        'nan_text': 0,
        'nan_speech': 0,
        'nan_speaker': 0,
        'text_lens': [],
        'speech_lens': [],
        'speaker_stds': []
    }
    
    for idx in sample_indices:
        sample_info = metadata['samples'][idx]
        pt_file = preprocessed_dir / sample_info['pt_file']
        data = torch.load(pt_file, map_location='cpu')
        
        # Check for NaN
        text_tokens = data.get('text_tokens')
        if text_tokens is not None and (torch.isnan(text_tokens).any() or torch.isinf(text_tokens).any()):
            stats['nan_text'] += 1
            logger.warning(f"  Sample {idx}: NaN in text tokens")
        
        speech_tokens = data.get('speech_tokens')
        if speech_tokens is not None and (torch.isnan(speech_tokens).any() or torch.isinf(speech_tokens).any()):
            stats['nan_speech'] += 1
            logger.warning(f"  Sample {idx}: NaN in speech tokens")
        
        speaker_emb = data.get('t3_cond_speaker_emb')
        if speaker_emb is not None and (torch.isnan(speaker_emb).any() or torch.isinf(speaker_emb).any()):
            stats['nan_speaker'] += 1
            logger.warning(f"  Sample {idx}: NaN in speaker embedding")
        
        # Collect stats
        text_len = data.get('text_token_lens', 0)
        if isinstance(text_len, torch.Tensor):
            text_len = text_len.item()
        stats['text_lens'].append(text_len)
        
        speech_len = data.get('speech_token_lens', 0)
        if isinstance(speech_len, torch.Tensor):
            speech_len = speech_len.item()
        stats['speech_lens'].append(speech_len)
        
        if speaker_emb is not None:
            stats['speaker_stds'].append(speaker_emb.std().item())
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 STATISTICS:")
    logger.info(f"{'='*60}")
    logger.info(f"  Samples with NaN text: {stats['nan_text']} ({stats['nan_text']/len(sample_indices)*100:.2f}%)")
    logger.info(f"  Samples with NaN speech: {stats['nan_speech']} ({stats['nan_speech']/len(sample_indices)*100:.2f}%)")
    logger.info(f"  Samples with NaN speaker: {stats['nan_speaker']} ({stats['nan_speaker']/len(sample_indices)*100:.2f}%)")
    
    if stats['text_lens']:
        import numpy as np
        logger.info(f"\n  Text lengths:")
        logger.info(f"    Min/Max: {min(stats['text_lens'])}/{max(stats['text_lens'])}")
        logger.info(f"    Mean/Std: {np.mean(stats['text_lens']):.1f}/{np.std(stats['text_lens']):.1f}")
    
    if stats['speech_lens']:
        import numpy as np
        logger.info(f"\n  Speech lengths:")
        logger.info(f"    Min/Max: {min(stats['speech_lens'])}/{max(stats['speech_lens'])}")
        logger.info(f"    Mean/Std: {np.mean(stats['speech_lens']):.1f}/{np.std(stats['speech_lens']):.1f}")
    
    if stats['speaker_stds']:
        import numpy as np
        logger.info(f"\n  Speaker embedding std:")
        logger.info(f"    Min/Max: {min(stats['speaker_stds']):.6f}/{max(stats['speaker_stds']):.6f}")
        logger.info(f"    Mean/Std: {np.mean(stats['speaker_stds']):.6f}/{np.std(stats['speaker_stds']):.6f}")
    
    logger.info(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Debug NaN samples in preprocessed data")
    parser.add_argument("--preprocessed_dir", type=str, default="preprocessed_data",
                       help="Directory containing preprocessed .pt files")
    parser.add_argument("--sample_idx", type=int, default=None,
                       help="Sample index to inspect")
    parser.add_argument("--audio_path", type=str, default=None,
                       help="Find sample by audio path")
    parser.add_argument("--batch_analyze", type=int, default=None,
                       help="Analyze N random samples")
    
    args = parser.parse_args()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    if not preprocessed_dir.exists():
        logger.error(f"Preprocessed directory not found: {preprocessed_dir}")
        return
    
    if args.batch_analyze:
        batch_analyze(preprocessed_dir, args.batch_analyze)
    elif args.audio_path:
        idx = find_by_audio_path(preprocessed_dir, args.audio_path)
        if idx is not None:
            inspect_sample(preprocessed_dir, idx)
    elif args.sample_idx is not None:
        inspect_sample(preprocessed_dir, args.sample_idx)
    else:
        logger.error("Please specify --sample_idx, --audio_path, or --batch_analyze")
        parser.print_help()


if __name__ == "__main__":
    main()
