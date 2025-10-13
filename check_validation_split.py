"""
Check validation split for issues

Usage:
    python check_validation_split.py
"""
import json
from pathlib import Path
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    preprocessed_dir = Path("preprocessed_data")
    
    if not preprocessed_dir.exists():
        logger.error("preprocessed_data not found!")
        return
    
    # Load metadata
    metadata_path = preprocessed_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    total_samples = len(metadata['samples'])
    eval_split_size = 0.01
    split_idx = int(total_samples * (1 - eval_split_size))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Dataset Split Info:")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(f"Train samples (99%): {split_idx:,}")
    logger.info(f"Val samples (1%): {total_samples - split_idx:,}")
    logger.info(f"Val split starts at index: {split_idx}")
    
    # Check first 20 validation samples
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Checking first 20 validation samples:")
    logger.info(f"{'='*60}\n")
    
    issues_count = 0
    
    for i in range(20):
        idx = split_idx + i
        if idx >= total_samples:
            break
        
        sample_info = metadata['samples'][idx]
        pt_file = preprocessed_dir / sample_info['pt_file']
        
        # Load sample
        try:
            data = torch.load(pt_file, map_location='cpu')
        except Exception as e:
            logger.error(f"❌ Sample {idx}: Failed to load - {e}")
            issues_count += 1
            continue
        
        # Check for issues
        has_issue = False
        issues = []
        
        # Check text tokens
        text_tokens = data.get('text_tokens')
        if text_tokens is not None:
            if torch.isnan(text_tokens).any():
                issues.append("NaN in text_tokens")
                has_issue = True
            if torch.isinf(text_tokens).any():
                issues.append("Inf in text_tokens")
                has_issue = True
        
        # Check speech tokens
        speech_tokens = data.get('speech_tokens')
        if speech_tokens is not None:
            if torch.isnan(speech_tokens).any():
                issues.append("NaN in speech_tokens")
                has_issue = True
            if torch.isinf(speech_tokens).any():
                issues.append("Inf in speech_tokens")
                has_issue = True
        
        # Check speaker embedding
        speaker_emb = data.get('t3_cond_speaker_emb')
        if speaker_emb is not None:
            if torch.isnan(speaker_emb).any():
                issues.append("NaN in speaker_emb")
                has_issue = True
            if torch.isinf(speaker_emb).any():
                issues.append("Inf in speaker_emb")
                has_issue = True
            if speaker_emb.std() < 1e-6:
                issues.append("Speaker emb all zeros/constant")
                has_issue = True
        
        # Check prompt tokens
        prompt_tokens = data.get('t3_cond_prompt_speech_tokens')
        if prompt_tokens is not None:
            if torch.isnan(prompt_tokens).any():
                issues.append("NaN in prompt_tokens")
                has_issue = True
            if torch.isinf(prompt_tokens).any():
                issues.append("Inf in prompt_tokens")
                has_issue = True
        
        if has_issue:
            logger.warning(f"❌ Val sample {i} (idx {idx}):")
            logger.warning(f"   File: {sample_info['audio_path']}")
            logger.warning(f"   Text: {data.get('text', 'N/A')[:50]}...")
            logger.warning(f"   Issues: {', '.join(issues)}")
            issues_count += 1
        else:
            logger.info(f"✅ Val sample {i} (idx {idx}): OK")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 Summary:")
    logger.info(f"{'='*60}")
    logger.info(f"Samples checked: 20")
    logger.info(f"Samples with issues: {issues_count}")
    logger.info(f"Issue ratio: {issues_count/20*100:.1f}%")
    
    if issues_count > 10:
        logger.error(f"\n❌ CRITICAL: {issues_count}/20 validation samples have issues!")
        logger.error(f"This matches the ~50% NaN ratio in evaluation")
        logger.error(f"\nRECOMMENDATION: Re-preprocess the dataset!")
    elif issues_count > 0:
        logger.warning(f"\n⚠️ WARNING: {issues_count}/20 validation samples have issues")
        logger.warning(f"Consider re-preprocessing or filtering bad samples")
    else:
        logger.info(f"\n✅ All validation samples look good!")


if __name__ == "__main__":
    main()
