"""
Test model forward pass on validation samples to find NaN source

Usage:
    python test_model_forward.py
"""
import torch
import logging
from pathlib import Path
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_sample(sample_idx=2578573):
    """Test model forward on a single validation sample"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing model forward on validation sample {sample_idx}")
    logger.info(f"{'='*60}\n")
    
    # Load preprocessed sample
    preprocessed_dir = Path("preprocessed_data")
    metadata_path = preprocessed_dir / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    sample_info = metadata['samples'][sample_idx]
    pt_file = preprocessed_dir / sample_info['pt_file']
    
    logger.info(f"Loading sample: {pt_file}")
    data = torch.load(pt_file, map_location='cpu')
    
    logger.info(f"\n📊 Sample data:")
    logger.info(f"  Audio: {data.get('audio_path', 'N/A')}")
    logger.info(f"  Text: {data.get('text', 'N/A')[:100]}...")
    logger.info(f"  Text tokens shape: {data['text_tokens'].shape}")
    logger.info(f"  Text token lens: {data['text_token_lens']}")
    logger.info(f"  Speech tokens shape: {data['speech_tokens'].shape}")
    logger.info(f"  Speech token lens: {data['speech_token_lens']}")
    logger.info(f"  Speaker emb shape: {data['t3_cond_speaker_emb'].shape}")
    logger.info(f"  Prompt tokens shape: {data['t3_cond_prompt_speech_tokens'].shape}")
    logger.info(f"  Emotion: {data['t3_cond_emotion_adv']}")
    
    # Load model
    logger.info(f"\n🔧 Loading model...")
    try:
        from src.chatterbox.models.t3.t3_for_causal_lm import ChatterboxT3ForCausalLM
        from src.chatterbox.models.t3.t3_config import T3Config
        from huggingface_hub import hf_hub_download
        
        # Download config
        config_path = hf_hub_download(
            repo_id="ResembleAI/chatterbox",
            filename="t3/config.json",
            cache_dir="./.cache"
        )
        
        # Load model
        model_dir = Path(config_path).parent
        t3_config = T3Config.from_json_file(config_path)
        
        logger.info(f"Loading model from {model_dir}")
        model = ChatterboxT3ForCausalLM.from_local(
            model_dir=str(model_dir),
            config=t3_config
        )
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Model loaded on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare inputs
    logger.info(f"\n📥 Preparing inputs...")
    
    inputs = {
        'text_tokens': data['text_tokens'].unsqueeze(0).to(device),
        'text_token_lens': torch.tensor([data['text_token_lens']]).to(device),
        'speech_tokens': data['speech_tokens'].unsqueeze(0).to(device),
        'speech_token_lens': torch.tensor([data['speech_token_lens']]).to(device),
        't3_cond_speaker_emb': data['t3_cond_speaker_emb'].unsqueeze(0).to(device),
        't3_cond_prompt_speech_tokens': data['t3_cond_prompt_speech_tokens'].unsqueeze(0).to(device),
        't3_cond_emotion_adv': torch.tensor([data['t3_cond_emotion_adv']]).to(device),
    }
    
    logger.info(f"Input shapes:")
    for key, val in inputs.items():
        if isinstance(val, torch.Tensor):
            logger.info(f"  {key}: {val.shape}, dtype={val.dtype}, device={val.device}")
            if torch.isnan(val).any():
                logger.error(f"    ❌ Contains NaN!")
            if torch.isinf(val).any():
                logger.error(f"    ❌ Contains Inf!")
    
    # Forward pass
    logger.info(f"\n🚀 Forward pass...")
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            
        logger.info(f"\n📤 Outputs:")
        if isinstance(outputs, (tuple, list)):
            loss = outputs[0]
            logits = outputs[1] if len(outputs) > 1 else None
            
            logger.info(f"  Loss: {loss}")
            logger.info(f"    Is NaN: {torch.isnan(loss).any()}")
            logger.info(f"    Is Inf: {torch.isinf(loss).any()}")
            
            if logits is not None:
                logger.info(f"  Logits shape: {logits.shape}")
                logger.info(f"    Min/Max: {logits.min():.4f}/{logits.max():.4f}")
                logger.info(f"    Mean/Std: {logits.mean():.4f}/{logits.std():.4f}")
                logger.info(f"    Has NaN: {torch.isnan(logits).any()}")
                logger.info(f"    Has Inf: {torch.isinf(logits).any()}")
        else:
            logger.info(f"  Output: {outputs}")
        
        if torch.isnan(loss):
            logger.error(f"\n❌ LOSS IS NaN!")
            logger.error(f"Need to debug model internals...")
            
            # Check model weights
            logger.info(f"\n🔍 Checking model weights...")
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
                    logger.error(f"  ❌ NaN in parameter: {name}")
            
            if not nan_params:
                logger.warning(f"  Model weights look OK, issue might be in forward computation")
        else:
            logger.info(f"\n✅ Forward pass successful! Loss = {loss.item():.4f}")
            
    except Exception as e:
        logger.error(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def test_batch_samples(num_samples=10):
    """Test model on multiple validation samples"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing {num_samples} validation samples")
    logger.info(f"{'='*60}\n")
    
    # Load model once
    logger.info(f"Loading model...")
    try:
        from src.chatterbox.models.t3.t3_for_causal_lm import ChatterboxT3ForCausalLM
        from src.chatterbox.models.t3.t3_config import T3Config
        from huggingface_hub import hf_hub_download
        
        config_path = hf_hub_download(
            repo_id="ResembleAI/chatterbox",
            filename="t3/config.json",
            cache_dir="./.cache"
        )
        
        model_dir = Path(config_path).parent
        t3_config = T3Config.from_json_file(config_path)
        
        model = ChatterboxT3ForCausalLM.from_local(
            model_dir=str(model_dir),
            config=t3_config
        )
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load preprocessed data
    preprocessed_dir = Path("preprocessed_data")
    metadata_path = preprocessed_dir / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    total_samples = len(metadata['samples'])
    split_idx = int(total_samples * 0.99)
    
    # Test samples
    nan_count = 0
    ok_count = 0
    
    for i in tqdm(range(num_samples), desc="Testing samples"):
        idx = split_idx + i
        
        sample_info = metadata['samples'][idx]
        pt_file = preprocessed_dir / sample_info['pt_file']
        
        try:
            data = torch.load(pt_file, map_location='cpu')
            
            inputs = {
                'text_tokens': data['text_tokens'].unsqueeze(0).to(device),
                'text_token_lens': torch.tensor([data['text_token_lens']]).to(device),
                'speech_tokens': data['speech_tokens'].unsqueeze(0).to(device),
                'speech_token_lens': torch.tensor([data['speech_token_lens']]).to(device),
                't3_cond_speaker_emb': data['t3_cond_speaker_emb'].unsqueeze(0).to(device),
                't3_cond_prompt_speech_tokens': data['t3_cond_prompt_speech_tokens'].unsqueeze(0).to(device),
                't3_cond_emotion_adv': torch.tensor([data['t3_cond_emotion_adv']]).to(device),
            }
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            if torch.isnan(loss):
                nan_count += 1
                logger.warning(f"  Sample {i} (idx {idx}): NaN loss")
            else:
                ok_count += 1
                
        except Exception as e:
            logger.error(f"  Sample {i} (idx {idx}): Error - {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Results:")
    logger.info(f"{'='*60}")
    logger.info(f"Samples tested: {num_samples}")
    logger.info(f"OK: {ok_count}")
    logger.info(f"NaN: {nan_count}")
    logger.info(f"NaN ratio: {nan_count/num_samples*100:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=2578573,
                       help="Validation sample index to test")
    parser.add_argument("--batch", type=int, default=None,
                       help="Test N samples")
    
    args = parser.parse_args()
    
    if args.batch:
        test_batch_samples(args.batch)
    else:
        test_single_sample(args.sample_idx)


if __name__ == "__main__":
    main()
