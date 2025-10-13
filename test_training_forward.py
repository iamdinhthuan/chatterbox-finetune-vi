"""
Test training forward pass to find NaN source

This script replicates the EXACT training setup and tests on real validation data
to pinpoint where NaN comes from.

Usage:
    python test_training_forward.py
    python test_training_forward.py --num_samples 20
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import torch
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_training_forward():
    """Test training forward pass on real validation data"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing Training Forward Pass")
    logger.info(f"{'='*60}\n")
    
    # Import training modules
    try:
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.models.t3.t3 import T3, T3Cond
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.utils.preprocessed_dataset import PreprocessedDataset, collate_fn_preprocessed
        from torch.utils.data import DataLoader
        
        logger.info("✅ Imports successful")
        
    except Exception as e:
        logger.error(f"❌ Failed to import modules: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load model
    logger.info(f"\n📥 Loading ChatterboxTTS model...")
    try:
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        t3_model = tts_model.t3
        t3_config = t3_model.hp
        logger.info(f"✅ Model loaded")
        logger.info(f"T3 config: {t3_config}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load validation dataset
    logger.info(f"\n📂 Loading validation dataset...")
    try:
        preprocessed_dir = "preprocessed_data"
        eval_split_size = 0.01
        
        val_dataset = PreprocessedDataset(
            preprocessed_dir=preprocessed_dir,
            max_text_len=256,
            max_speech_len=1200,
            split='val',
            eval_split_size=eval_split_size
        )
        
        logger.info(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create dataloader
    logger.info(f"\n🔄 Creating dataloader...")
    try:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Test one at a time first
            shuffle=False,
            collate_fn=collate_fn_preprocessed,
            num_workers=0
        )
        logger.info(f"✅ Dataloader created")
        
    except Exception as e:
        logger.error(f"❌ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass on samples
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing Forward Pass on Real Validation Data")
    logger.info(f"{'='*60}\n")
    
    num_test = 20
    nan_count = 0
    ok_count = 0
    errors = []
    
    t3_model.eval()
    
    for idx, batch in enumerate(tqdm(val_loader, total=num_test, desc="Testing")):
        if idx >= num_test:
            break
        
        if batch is None:
            logger.warning(f"  Sample {idx}: Batch is None")
            continue
        
        try:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Extract inputs
            text_tokens = batch['text_tokens']
            text_token_lens = batch['text_token_lens']
            speech_tokens = batch['speech_tokens']
            speech_token_lens = batch['speech_token_lens']
            speaker_emb = batch['t3_cond_speaker_emb']
            prompt_tokens = batch['t3_cond_prompt_speech_tokens']
            emotion = batch['t3_cond_emotion_adv']
            
            # Check for NaN in inputs
            has_nan_input = False
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    if torch.isnan(val).any():
                        logger.warning(f"  Sample {idx}: NaN in INPUT '{key}'")
                        has_nan_input = True
            
            if has_nan_input:
                nan_count += 1
                errors.append({
                    'idx': idx,
                    'type': 'NaN in inputs',
                    'batch': batch
                })
                continue
            
            # Create T3Cond
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=prompt_tokens,
                emotion_adv=emotion.unsqueeze(-1).unsqueeze(-1) if emotion.dim() == 1 else emotion,
            )
            
            # Forward pass
            with torch.no_grad():
                output = t3_model(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens,
                    training=False
                )
            
            # Check output
            speech_logits = output.speech_logits
            text_logits = output.text_logits
            
            has_nan_output = False
            if torch.isnan(speech_logits).any():
                logger.warning(f"  Sample {idx}: NaN in speech_logits")
                has_nan_output = True
            
            if torch.isnan(text_logits).any():
                logger.warning(f"  Sample {idx}: NaN in text_logits")
                has_nan_output = True
            
            if has_nan_output:
                nan_count += 1
                errors.append({
                    'idx': idx,
                    'type': 'NaN in outputs',
                    'text_len': text_token_lens.item(),
                    'speech_len': speech_token_lens.item(),
                    'logits_shape': speech_logits.shape
                })
            else:
                ok_count += 1
                
        except Exception as e:
            logger.error(f"  Sample {idx}: Error - {e}")
            errors.append({
                'idx': idx,
                'type': 'Exception',
                'error': str(e)
            })
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 FORWARD PASS TEST SUMMARY:")
    logger.info(f"{'='*60}")
    logger.info(f"Samples tested: {num_test}")
    logger.info(f"OK: {ok_count}")
    logger.info(f"NaN: {nan_count}")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"NaN ratio: {nan_count/num_test*100:.1f}%")
    
    if errors:
        logger.info(f"\n❌ Error details:")
        for err in errors[:10]:
            logger.info(f"  Sample {err['idx']}: {err['type']}")
            if 'text_len' in err:
                logger.info(f"    Text len: {err['text_len']}, Speech len: {err['speech_len']}")
    
    # Now test with TRAINING mode and LOSS computation
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 Testing LOSS COMPUTATION")
    logger.info(f"{'='*60}\n")
    
    # Get one sample
    val_loader_single = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_preprocessed,
        num_workers=0
    )
    
    batch = next(iter(val_loader_single))
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Test TRAINING forward (with loss)
    t3_model.train()
    
    try:
        logger.info("Testing forward with training=True...")
        
        text_tokens = batch['text_tokens']
        text_token_lens = batch['text_token_lens']
        speech_tokens = batch['speech_tokens']
        speech_token_lens = batch['speech_token_lens']
        speaker_emb = batch['t3_cond_speaker_emb']
        prompt_tokens = batch['t3_cond_prompt_speech_tokens']
        emotion = batch['t3_cond_emotion_adv']
        
        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=prompt_tokens,
            emotion_adv=emotion.unsqueeze(-1).unsqueeze(-1) if emotion.dim() == 1 else emotion,
        )
        
        with torch.no_grad():
            output = t3_model(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                text_token_lens=text_token_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_token_lens,
                training=True  # TRAINING MODE
            )
        
        speech_logits = output.speech_logits
        text_logits = output.text_logits
        
        logger.info(f"\n📤 Training forward output:")
        logger.info(f"  Speech logits shape: {speech_logits.shape}")
        logger.info(f"  Text logits shape: {text_logits.shape}")
        logger.info(f"  Speech logits has NaN: {torch.isnan(speech_logits).any()}")
        logger.info(f"  Text logits has NaN: {torch.isnan(text_logits).any()}")
        
        # Now test LOSS computation
        logger.info(f"\n🧮 Testing loss computation...")
        
        # Prepare labels (shift tokens for next-token prediction)
        labels_speech = speech_tokens.clone()
        labels_text = text_tokens.clone()
        
        # Compute loss manually
        import torch.nn.functional as F
        
        # Flatten for cross entropy
        B, L_speech, vocab_size = speech_logits.shape
        speech_logits_flat = speech_logits.reshape(-1, vocab_size)
        labels_speech_flat = labels_speech.reshape(-1)
        
        logger.info(f"\n  Computing speech loss...")
        logger.info(f"    Logits shape: {speech_logits_flat.shape}")
        logger.info(f"    Labels shape: {labels_speech_flat.shape}")
        logger.info(f"    Labels min/max: {labels_speech_flat.min()}/{labels_speech_flat.max()}")
        logger.info(f"    Vocab size: {vocab_size}")
        
        # Check if labels are in valid range
        invalid_labels = (labels_speech_flat < 0) | (labels_speech_flat >= vocab_size)
        if invalid_labels.any():
            logger.error(f"    ❌ Invalid labels found: {invalid_labels.sum()} out of {len(labels_speech_flat)}")
        
        loss_speech = F.cross_entropy(
            speech_logits_flat,
            labels_speech_flat,
            ignore_index=-100,
            reduction='mean'
        )
        
        logger.info(f"\n  Speech loss: {loss_speech}")
        logger.info(f"  Is NaN: {torch.isnan(loss_speech)}")
        logger.info(f"  Is Inf: {torch.isinf(loss_speech)}")
        
        if torch.isnan(loss_speech):
            logger.error(f"\n❌ LOSS IS NaN!")
            logger.error(f"This is the source of training NaN issue!")
            
            # Debug why
            logger.info(f"\n🔍 Debugging NaN loss:")
            logger.info(f"  Logits stats:")
            logger.info(f"    Min: {speech_logits_flat.min()}")
            logger.info(f"    Max: {speech_logits_flat.max()}")
            logger.info(f"    Mean: {speech_logits_flat.mean()}")
            logger.info(f"    Std: {speech_logits_flat.std()}")
            logger.info(f"    Has Inf: {torch.isinf(speech_logits_flat).any()}")
            logger.info(f"    Has NaN: {torch.isnan(speech_logits_flat).any()}")
            
            # Check softmax
            probs = F.softmax(speech_logits_flat, dim=-1)
            logger.info(f"  Softmax probs:")
            logger.info(f"    Min: {probs.min()}")
            logger.info(f"    Max: {probs.max()}")
            logger.info(f"    Has NaN: {torch.isnan(probs).any()}")
        else:
            logger.info(f"\n✅ Loss computation works!")
        
    except Exception as e:
        logger.error(f"\n❌ Training forward failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 FINAL CONCLUSION:")
    logger.info(f"{'='*60}")
    
    if nan_count == 0:
        logger.info(f"✅ Forward pass works on ALL validation samples!")
        logger.info(f"✅ Issue is likely in:")
        logger.info(f"   1. Training wrapper code (T3ForFineTuning)")
        logger.info(f"   2. Trainer.compute_loss() implementation")
        logger.info(f"   3. Labels preparation/formatting")
    else:
        logger.error(f"❌ Forward pass produces NaN on {nan_count}/{num_test} samples")
        logger.error(f"NaN comes from model forward, not loss computation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples to test")
    
    args = parser.parse_args()
    
    test_training_forward()


if __name__ == "__main__":
    main()
