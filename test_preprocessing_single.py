"""
Test preprocessing with a single audio sample before running full dataset
Verifies format is 100% correct before committing 22 hours
"""
import sys
from pathlib import Path
import torch
import csv
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess_dataset import preprocess_sample
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3.modules.t3_config import T3Config
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_sample(
    csv_path: str = "metadata.csv",
    audio_dir: str = "wavs",
    tokenizer_path: str = "VietnameseTokenizer/tokenizer.json",
    sample_idx: int = 0
):
    """
    Test preprocessing with one sample and verify format
    """
    
    print("="*80)
    print("TEST PREPROCESSING WITH SINGLE SAMPLE")
    print("="*80)
    
    # 1. Load sample from CSV
    print(f"\n1. Loading sample {sample_idx} from {csv_path}...")
    csv_path = Path(csv_path)
    audio_dir = Path(audio_dir)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        samples = list(reader)
    
    if sample_idx >= len(samples):
        print(f"❌ Sample index {sample_idx} out of range (total: {len(samples)})")
        return False
    
    sample = samples[sample_idx]
    audio_path = audio_dir / sample['audio']
    text = sample['transcript']
    
    print(f"   Audio: {audio_path}")
    print(f"   Text: {text}")
    print(f"   Exists: {audio_path.exists()}")
    
    if not audio_path.exists():
        print(f"❌ Audio file not found!")
        return False
    
    # 2. Load model components
    print(f"\n2. Loading model components...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        # Download model
        from huggingface_hub import hf_hub_download
        model_dir = Path("./cache/chatterbox_model")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        repo_id = "ResembleAI/chatterbox"
        files_to_download = ["ve.safetensors", "s3gen.safetensors", "t3_cfg.safetensors"]
        
        print(f"   Downloading model from {repo_id}...")
        for file in files_to_download:
            if not (model_dir / file).exists():
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
        
        # Try to download conds.pt (optional)
        try:
            if not (model_dir / "conds.pt").exists():
                hf_hub_download(
                    repo_id=repo_id,
                    filename="conds.pt",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
        except:
            pass
        
        # Copy custom tokenizer
        import shutil
        shutil.copy(tokenizer_path, model_dir / "tokenizer.json")
        
        # Load model
        tts = ChatterboxTTS.from_local(ckpt_dir=str(model_dir), device=device)
        
        # Get components
        text_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        speech_tokenizer = tts.s3gen.tokenizer
        voice_encoder = tts.ve
        voice_encoder.eval()
        t3_config = tts.t3.hp
        
        print(f"   ✅ Model loaded")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Preprocess sample
    print(f"\n3. Preprocessing sample...")
    try:
        preprocessed = preprocess_sample(
            audio_path=audio_path,
            text=text,
            text_tokenizer=text_tokenizer,
            speech_tokenizer=speech_tokenizer,
            voice_encoder=voice_encoder,
            t3_config=t3_config,
            max_text_len=256,
            max_speech_len=1200,
            audio_prompt_duration_s=3.0,
            add_silence=True,
            silence_padding_ms=300,
        )
        
        if preprocessed is None:
            print(f"❌ Preprocessing returned None!")
            return False
        
        print(f"   ✅ Preprocessing successful")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Verify format
    print(f"\n4. Verifying format...")
    print("="*80)
    
    errors = []
    warnings = []
    
    # Check keys
    required_keys = [
        'text_tokens', 'text_token_lens', 
        'speech_tokens', 'speech_token_lens',
        't3_cond_speaker_emb', 't3_cond_prompt_speech_tokens', 
        't3_cond_emotion_adv'
    ]
    
    print("\n📋 Keys Check:")
    for key in required_keys:
        if key not in preprocessed:
            errors.append(f"Missing key: {key}")
            print(f"   ❌ {key}")
        else:
            print(f"   ✅ {key}")
    
    if errors:
        print("\n❌ KEYS MISSING!")
        for err in errors:
            print(f"  - {err}")
        return False
    
    # Check data types and shapes
    print("\n📊 Data Types & Shapes:")
    
    # Text tokens
    text_tokens = preprocessed['text_tokens']
    print(f"\n   text_tokens:")
    print(f"     Type: {type(text_tokens)}")
    print(f"     Dtype: {text_tokens.dtype}")
    print(f"     Shape: {text_tokens.shape}")
    print(f"     Dim: {text_tokens.dim()}")
    print(f"     First 10: {text_tokens[:10].tolist()}")
    print(f"     Last 5: {text_tokens[-5:].tolist()}")
    
    if text_tokens.dtype != torch.long:
        errors.append(f"text_tokens dtype should be torch.long, got {text_tokens.dtype}")
    if text_tokens.dim() != 1:
        errors.append(f"text_tokens should be 1D, got {text_tokens.dim()}D")
    
    # Text token lens
    text_lens = preprocessed['text_token_lens']
    print(f"\n   text_token_lens:")
    print(f"     Type: {type(text_lens)}")
    print(f"     Dtype: {text_lens.dtype}")
    print(f"     Shape: {text_lens.shape}")
    print(f"     Value: {text_lens.item()}")
    
    if text_lens.dtype != torch.long:
        errors.append(f"text_token_lens dtype should be torch.long, got {text_lens.dtype}")
    if text_lens.dim() != 0:
        errors.append(f"text_token_lens should be scalar (0D), got {text_lens.dim()}D")
    
    # Speech tokens
    speech_tokens = preprocessed['speech_tokens']
    print(f"\n   speech_tokens:")
    print(f"     Type: {type(speech_tokens)}")
    print(f"     Dtype: {speech_tokens.dtype}")
    print(f"     Shape: {speech_tokens.shape}")
    print(f"     Dim: {speech_tokens.dim()}")
    print(f"     First 10: {speech_tokens[:10].tolist()}")
    print(f"     Last 5: {speech_tokens[-5:].tolist()}")
    
    if speech_tokens.dtype != torch.long:
        errors.append(f"speech_tokens dtype should be torch.long, got {speech_tokens.dtype}")
    if speech_tokens.dim() != 1:
        errors.append(f"speech_tokens should be 1D, got {speech_tokens.dim()}D")
    
    # Speech token lens
    speech_lens = preprocessed['speech_token_lens']
    print(f"\n   speech_token_lens:")
    print(f"     Type: {type(speech_lens)}")
    print(f"     Dtype: {speech_lens.dtype}")
    print(f"     Shape: {speech_lens.shape}")
    print(f"     Value: {speech_lens.item()}")
    
    if speech_lens.dtype != torch.long:
        errors.append(f"speech_token_lens dtype should be torch.long, got {speech_lens.dtype}")
    if speech_lens.dim() != 0:
        errors.append(f"speech_token_lens should be scalar (0D), got {speech_lens.dim()}D")
    
    # Speaker embedding
    speaker_emb = preprocessed['t3_cond_speaker_emb']
    print(f"\n   t3_cond_speaker_emb:")
    print(f"     Type: {type(speaker_emb)}")
    print(f"     Dtype: {speaker_emb.dtype}")
    print(f"     Shape: {speaker_emb.shape}")
    print(f"     Dim: {speaker_emb.dim()}")
    
    if speaker_emb.dtype not in [torch.float, torch.float32]:
        warnings.append(f"t3_cond_speaker_emb dtype should be torch.float, got {speaker_emb.dtype}")
    if speaker_emb.dim() != 1:
        errors.append(f"t3_cond_speaker_emb should be 1D, got {speaker_emb.dim()}D")
    if speaker_emb.shape[0] != 256:
        errors.append(f"t3_cond_speaker_emb should have 256 dims, got {speaker_emb.shape[0]}")
    
    # Conditioning prompt tokens
    cond_tokens = preprocessed['t3_cond_prompt_speech_tokens']
    print(f"\n   t3_cond_prompt_speech_tokens:")
    print(f"     Type: {type(cond_tokens)}")
    print(f"     Dtype: {cond_tokens.dtype}")
    print(f"     Shape: {cond_tokens.shape}")
    print(f"     Dim: {cond_tokens.dim()}")
    
    if cond_tokens.dtype != torch.long:
        errors.append(f"t3_cond_prompt_speech_tokens dtype should be torch.long, got {cond_tokens.dtype}")
    if cond_tokens.dim() != 1:
        errors.append(f"t3_cond_prompt_speech_tokens should be 1D, got {cond_tokens.dim()}D")
    if cond_tokens.shape[0] != 150:
        errors.append(f"t3_cond_prompt_speech_tokens should have 150 tokens, got {cond_tokens.shape[0]}")
    
    # Emotion scalar
    emotion = preprocessed['t3_cond_emotion_adv']
    print(f"\n   t3_cond_emotion_adv:")
    print(f"     Type: {type(emotion)}")
    print(f"     Dtype: {emotion.dtype}")
    print(f"     Shape: {emotion.shape}")
    print(f"     Dim: {emotion.dim()}")
    print(f"     Value: {emotion.item()}")
    
    if emotion.dtype not in [torch.float, torch.float32]:
        warnings.append(f"t3_cond_emotion_adv dtype should be torch.float, got {emotion.dtype}")
    if emotion.dim() != 0:
        errors.append(f"t3_cond_emotion_adv should be scalar (0D), got {emotion.dim()}D")
    
    # Check BOS/EOS tokens
    print("\n🔖 BOS/EOS Tokens Check:")
    
    text_bos = text_tokens[0].item()
    text_eos = text_tokens[-1].item()
    print(f"   Text BOS (first token): {text_bos} (expected: 255)")
    print(f"   Text EOS (last token): {text_eos} (expected: 0)")
    
    if text_bos != 255:
        errors.append(f"Text BOS should be 255, got {text_bos}")
    if text_eos != 0:
        errors.append(f"Text EOS should be 0, got {text_eos}")
    
    speech_bos = speech_tokens[0].item()
    speech_eos = speech_tokens[-1].item()
    print(f"   Speech BOS (first token): {speech_bos} (expected: 6561)")
    print(f"   Speech EOS (last token): {speech_eos} (expected: 6562)")
    
    if speech_bos != 6561:
        errors.append(f"Speech BOS should be 6561, got {speech_bos}")
    if speech_eos != 6562:
        errors.append(f"Speech EOS should be 6562, got {speech_eos}")
    
    # Check lengths match
    print("\n📏 Length Consistency Check:")
    
    actual_text_len = len(text_tokens)
    stored_text_len = text_lens.item()
    print(f"   Text: len(tokens)={actual_text_len}, stored_len={stored_text_len}")
    
    if actual_text_len != stored_text_len:
        errors.append(f"Text length mismatch: {actual_text_len} != {stored_text_len}")
    
    actual_speech_len = len(speech_tokens)
    stored_speech_len = speech_lens.item()
    print(f"   Speech: len(tokens)={actual_speech_len}, stored_len={stored_speech_len}")
    
    if actual_speech_len != stored_speech_len:
        errors.append(f"Speech length mismatch: {actual_speech_len} != {stored_speech_len}")
    
    # Final summary
    print("\n" + "="*80)
    
    if errors:
        print("❌ VERIFICATION FAILED!")
        print("\nERRORS:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ VERIFICATION PASSED!")
        print("\n🎉 Format is 100% correct!")
        print("   Safe to run full preprocessing on 2.6M samples")
    
    if warnings:
        print("\nWARNINGS (non-critical):")
        for warn in warnings:
            print(f"  - {warn}")
    
    print("="*80)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test preprocessing with single sample")
    parser.add_argument("--csv", type=str, default="metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--audio_dir", type=str, default="wavs", help="Audio directory")
    parser.add_argument("--tokenizer", type=str, default="VietnameseTokenizer/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to test (default: 0)")
    
    args = parser.parse_args()
    
    success = test_single_sample(
        csv_path=args.csv,
        audio_dir=args.audio_dir,
        tokenizer_path=args.tokenizer,
        sample_idx=args.sample_idx
    )
    
    sys.exit(0 if success else 1)
