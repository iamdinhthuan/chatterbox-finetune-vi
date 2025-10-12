"""
Comprehensive verification script for preprocessing format
Run this AFTER preprocessing completes to verify data format is 100% correct
"""
import torch
from pathlib import Path
import sys

def verify_sample(sample_path: Path):
    """Verify one preprocessed sample has correct format"""
    
    print(f"\n{'='*80}")
    print(f"VERIFYING: {sample_path}")
    print(f"{'='*80}\n")
    
    try:
        sample = torch.load(sample_path, map_location='cpu')
    except Exception as e:
        print(f"❌ FAILED to load: {e}")
        return False
    
    errors = []
    warnings = []
    
    # Expected keys
    required_keys = [
        'text_tokens',
        'text_token_lens',
        'speech_tokens',
        'speech_token_lens',
        't3_cond_speaker_emb',
        't3_cond_prompt_speech_tokens',
        't3_cond_emotion_adv',
    ]
    
    optional_keys = ['audio_path', 'text']
    
    print("1. Checking Keys...")
    print("-" * 40)
    
    # Check all required keys present
    for key in required_keys:
        if key not in sample:
            errors.append(f"Missing required key: '{key}'")
        else:
            print(f"  ✓ {key}")
    
    # Check no extra unexpected keys
    all_expected = required_keys + optional_keys
    for key in sample.keys():
        if key not in all_expected:
            warnings.append(f"Unexpected key: '{key}'")
    
    if errors:
        print("\n❌ KEY ERRORS:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("\n2. Checking Data Types...")
    print("-" * 40)
    
    # Check text_tokens
    if not isinstance(sample['text_tokens'], torch.Tensor):
        errors.append(f"text_tokens should be torch.Tensor, got {type(sample['text_tokens'])}")
    else:
        if sample['text_tokens'].dtype != torch.long:
            errors.append(f"text_tokens should be torch.long, got {sample['text_tokens'].dtype}")
        if sample['text_tokens'].dim() != 1:
            errors.append(f"text_tokens should be 1D, got {sample['text_tokens'].dim()}D")
        print(f"  ✓ text_tokens: {sample['text_tokens'].shape}, dtype={sample['text_tokens'].dtype}")
    
    # Check text_token_lens
    if not isinstance(sample['text_token_lens'], torch.Tensor):
        errors.append(f"text_token_lens should be torch.Tensor, got {type(sample['text_token_lens'])}")
    else:
        if sample['text_token_lens'].dtype != torch.long:
            errors.append(f"text_token_lens should be torch.long, got {sample['text_token_lens'].dtype}")
        if sample['text_token_lens'].dim() != 0:
            errors.append(f"text_token_lens should be scalar (0D), got {sample['text_token_lens'].dim()}D")
        print(f"  ✓ text_token_lens: {sample['text_token_lens'].item()}, dtype={sample['text_token_lens'].dtype}")
    
    # Check speech_tokens
    if not isinstance(sample['speech_tokens'], torch.Tensor):
        errors.append(f"speech_tokens should be torch.Tensor, got {type(sample['speech_tokens'])}")
    else:
        if sample['speech_tokens'].dtype != torch.long:
            errors.append(f"speech_tokens should be torch.long, got {sample['speech_tokens'].dtype}")
        if sample['speech_tokens'].dim() != 1:
            errors.append(f"speech_tokens should be 1D, got {sample['speech_tokens'].dim()}D")
        print(f"  ✓ speech_tokens: {sample['speech_tokens'].shape}, dtype={sample['speech_tokens'].dtype}")
    
    # Check speech_token_lens
    if not isinstance(sample['speech_token_lens'], torch.Tensor):
        errors.append(f"speech_token_lens should be torch.Tensor, got {type(sample['speech_token_lens'])}")
    else:
        if sample['speech_token_lens'].dtype != torch.long:
            errors.append(f"speech_token_lens should be torch.long, got {sample['speech_token_lens'].dtype}")
        if sample['speech_token_lens'].dim() != 0:
            errors.append(f"speech_token_lens should be scalar (0D), got {sample['speech_token_lens'].dim()}D")
        print(f"  ✓ speech_token_lens: {sample['speech_token_lens'].item()}, dtype={sample['speech_token_lens'].dtype}")
    
    # Check t3_cond_speaker_emb
    if not isinstance(sample['t3_cond_speaker_emb'], torch.Tensor):
        errors.append(f"t3_cond_speaker_emb should be torch.Tensor, got {type(sample['t3_cond_speaker_emb'])}")
    else:
        if sample['t3_cond_speaker_emb'].dtype not in [torch.float, torch.float32]:
            warnings.append(f"t3_cond_speaker_emb should be torch.float, got {sample['t3_cond_speaker_emb'].dtype}")
        if sample['t3_cond_speaker_emb'].dim() != 1:
            errors.append(f"t3_cond_speaker_emb should be 1D, got {sample['t3_cond_speaker_emb'].dim()}D")
        if sample['t3_cond_speaker_emb'].shape[0] != 256:
            errors.append(f"t3_cond_speaker_emb should have 256 dims, got {sample['t3_cond_speaker_emb'].shape[0]}")
        print(f"  ✓ t3_cond_speaker_emb: {sample['t3_cond_speaker_emb'].shape}, dtype={sample['t3_cond_speaker_emb'].dtype}")
    
    # Check t3_cond_prompt_speech_tokens
    if not isinstance(sample['t3_cond_prompt_speech_tokens'], torch.Tensor):
        errors.append(f"t3_cond_prompt_speech_tokens should be torch.Tensor, got {type(sample['t3_cond_prompt_speech_tokens'])}")
    else:
        if sample['t3_cond_prompt_speech_tokens'].dtype != torch.long:
            errors.append(f"t3_cond_prompt_speech_tokens should be torch.long, got {sample['t3_cond_prompt_speech_tokens'].dtype}")
        if sample['t3_cond_prompt_speech_tokens'].dim() != 1:
            errors.append(f"t3_cond_prompt_speech_tokens should be 1D, got {sample['t3_cond_prompt_speech_tokens'].dim()}D")
        if sample['t3_cond_prompt_speech_tokens'].shape[0] != 150:
            errors.append(f"t3_cond_prompt_speech_tokens should have 150 tokens, got {sample['t3_cond_prompt_speech_tokens'].shape[0]}")
        print(f"  ✓ t3_cond_prompt_speech_tokens: {sample['t3_cond_prompt_speech_tokens'].shape}, dtype={sample['t3_cond_prompt_speech_tokens'].dtype}")
    
    # Check t3_cond_emotion_adv
    if not isinstance(sample['t3_cond_emotion_adv'], torch.Tensor):
        errors.append(f"t3_cond_emotion_adv should be torch.Tensor, got {type(sample['t3_cond_emotion_adv'])}")
    else:
        if sample['t3_cond_emotion_adv'].dtype not in [torch.float, torch.float32]:
            warnings.append(f"t3_cond_emotion_adv should be torch.float, got {sample['t3_cond_emotion_adv'].dtype}")
        if sample['t3_cond_emotion_adv'].dim() != 0:
            errors.append(f"t3_cond_emotion_adv should be scalar (0D), got {sample['t3_cond_emotion_adv'].dim()}D")
        expected_val = 0.5
        if abs(sample['t3_cond_emotion_adv'].item() - expected_val) > 0.01:
            warnings.append(f"t3_cond_emotion_adv should be ~0.5, got {sample['t3_cond_emotion_adv'].item()}")
        print(f"  ✓ t3_cond_emotion_adv: {sample['t3_cond_emotion_adv'].item()}, dtype={sample['t3_cond_emotion_adv'].dtype}")
    
    print("\n3. Checking BOS/EOS Tokens...")
    print("-" * 40)
    
    # Check text BOS/EOS
    text_tokens = sample['text_tokens']
    if text_tokens[0].item() != 255:
        errors.append(f"Text BOS token should be 255, got {text_tokens[0].item()}")
    else:
        print(f"  ✓ Text BOS (start_text_token): {text_tokens[0].item()} == 255")
    
    if text_tokens[-1].item() != 0:
        errors.append(f"Text EOS token should be 0, got {text_tokens[-1].item()}")
    else:
        print(f"  ✓ Text EOS (stop_text_token): {text_tokens[-1].item()} == 0")
    
    # Check speech BOS/EOS
    speech_tokens = sample['speech_tokens']
    if speech_tokens[0].item() != 6561:
        errors.append(f"Speech BOS token should be 6561, got {speech_tokens[0].item()}")
    else:
        print(f"  ✓ Speech BOS (start_speech_token): {speech_tokens[0].item()} == 6561")
    
    if speech_tokens[-1].item() != 6562:
        errors.append(f"Speech EOS token should be 6562, got {speech_tokens[-1].item()}")
    else:
        print(f"  ✓ Speech EOS (stop_speech_token): {speech_tokens[-1].item()} == 6562")
    
    print("\n4. Checking Lengths Match...")
    print("-" * 40)
    
    # Check lengths match actual tensor sizes
    if sample['text_token_lens'].item() != len(sample['text_tokens']):
        errors.append(f"text_token_lens ({sample['text_token_lens'].item()}) != len(text_tokens) ({len(sample['text_tokens'])})")
    else:
        print(f"  ✓ text_token_lens matches: {sample['text_token_lens'].item()}")
    
    if sample['speech_token_lens'].item() != len(sample['speech_tokens']):
        errors.append(f"speech_token_lens ({sample['speech_token_lens'].item()}) != len(speech_tokens) ({len(sample['speech_tokens'])})")
    else:
        print(f"  ✓ speech_token_lens matches: {sample['speech_token_lens'].item()}")
    
    # Print summary
    print("\n" + "="*80)
    if errors:
        print("❌ VERIFICATION FAILED!")
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("✅ VERIFICATION PASSED!")
    
    if warnings:
        print("\nWARNINGS:")
        for warn in warnings:
            print(f"  - {warn}")
    
    print("="*80)
    
    return len(errors) == 0


if __name__ == "__main__":
    preprocessed_dir = Path("./preprocessed_data")
    
    if not preprocessed_dir.exists():
        print(f"❌ Preprocessed directory not found: {preprocessed_dir}")
        print("Please run preprocessing first:")
        print("  python preprocess_dataset.py --csv metadata.csv --audio_dir wavs --add_silence")
        sys.exit(1)
    
    # Find first .pt file
    pt_files = list(preprocessed_dir.glob("sample_*.pt"))
    
    if not pt_files:
        print(f"❌ No .pt files found in {preprocessed_dir}")
        sys.exit(1)
    
    # Verify first sample
    sample_path = sorted(pt_files)[0]
    success = verify_sample(sample_path)
    
    if success:
        print("\n✅ Sample format is CORRECT!")
        print("You can proceed with training using --use_preprocessed")
    else:
        print("\n❌ Sample format has ERRORS!")
        print("Please re-run preprocessing with fixed code")
        sys.exit(1)
