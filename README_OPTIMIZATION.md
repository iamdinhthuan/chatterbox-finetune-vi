# ⚡ Training Optimization Guide

Based on: https://github.com/resemble-ai/chatterbox/issues/174

## Problem

Training with on-the-fly data processing is **SLOW**:
- Audio loading, resampling, tokenization, voice encoding happen in Dataset.__getitem__
- Major CPU bottleneck
- Iteration time: ~2.2-2.9s/it (even with optimized dataloader_num_workers)

## Solution: Offline Preprocessing

Pre-compute ALL expensive operations once, save to `.pt` files.

### Step 1: Preprocess Dataset

```bash
python preprocess_dataset.py --csv metadata.csv --add_silence
```

**What it does:**
1. Load and resample audio to 24kHz
2. Add 300ms silence padding (helps first/last word issues)
3. Tokenize text with your Vietnamese tokenizer
4. Speech tokenization (S3 tokens)
5. Voice encoding (for prompt)
6. Save everything to individual `.pt` files

**Options:**
```bash
python preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir ./audio \
  --output_dir ./preprocessed_data \
  --add_silence \
  --silence_padding_ms 300 \
  --max_text_len 256 \
  --max_speech_len 1200
```

**Output:**
- `preprocessed_data/sample_000000.pt`
- `preprocessed_data/sample_000001.pt`
- ...
- `preprocessed_data/metadata.json`

**Time:** ~10-30 minutes for 10k samples (one-time cost!)

**Size:** ~100MB for 5k samples (from 1.4GB raw audio)

### Step 2: Train with Preprocessed Data

```bash
python train.py \
  --csv metadata.csv \
  --use_preprocessed \
  --preprocessed_dir ./preprocessed_data \
  --batch_size 8 \
  --epochs 10
```

## Results

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Iteration time | 2.2-2.9s | 1.2s | **2-4x faster** |
| CPU load | High | Low | Minimal |
| GPU utilization | Bottlenecked | Maximized | VRAM becomes limit |
| Training time (10k samples) | 1-3 days | 12-24 hours | **~2x faster** |

## Additional Tips from Issue

### 1. **Silence Padding** (Important!)
```bash
python preprocess_dataset.py --csv metadata.csv --add_silence --silence_padding_ms 300
```

- Adds 300ms silence at start/end of audio
- Fixes first/last word skipping issues
- Helps model learn proper boundaries

### 2. **Epochs: 5-10 (NOT 128!)**
```bash
python train.py --epochs 10  # NOT 128!
```

- User in issue trained 128 epochs → **overfitting**
- **Best results at epoch 5**
- After epoch 10: quality degrades
- Recommended: Train 5-10 epochs, check intermediate checkpoints

### 3. **Batch Size vs VRAM**

With sequence_length=2048 (original model):
- batch_size=1 → 28GB VRAM
- batch_size=5 → faster than batch_size=1

Tips:
- Start with batch_size=4-8
- If OOM, lower to 2
- Use gradient_accumulation_steps to maintain effective batch size

### 4. **Data Quality Matters**

From user experience:
- ✅ Remove too-short samples (< 1 second)
- ✅ Clean data manually (check for errors)
- ✅ Add tricky pronunciations to training set
- ✅ 10k-12k samples for good results

### 5. **Dataloader Settings**

With preprocessing:
```python
dataloader_num_workers=2-8  # Lower is OK now (CPU not bottleneck)
dataloader_persistent_workers=True
```

Without preprocessing:
```python
dataloader_num_workers=12  # Need more to keep up
```

## Workflow Comparison

### Without Preprocessing (Slow)
```bash
# 1. Train tokenizer
python train_tokenizer_from_corpus.py metadata.csv

# 2. Train directly (SLOW!)
python train.py --csv metadata.csv --epochs 10

# Result: ~1-3 days for 10k samples
```

### With Preprocessing (Fast) ⚡
```bash
# 1. Train tokenizer
python train_tokenizer_from_corpus.py metadata.csv

# 2. Preprocess (one-time, 10-30 mins)
python preprocess_dataset.py --csv metadata.csv --add_silence

# 3. Train (FAST!)
python train.py --csv metadata.csv --use_preprocessed --epochs 10

# Result: ~12-24 hours for 10k samples (2-4x faster!)
```

## Technical Details

### What's in a .pt file?

```python
{
    "text_tokens": Tensor[T_text],      # Tokenized text
    "speech_tokens": Tensor[T_speech],  # S3 speech tokens
    "voice_emb": Tensor[D],             # Voice embedding for prompt
    "audio_path": str,                  # Original audio path
    "text": str,                        # Original text
}
```

### Why is it faster?

**Before (on-the-fly):**
```
For each batch:
  1. Load WAV from disk (I/O)
  2. Resample to 24kHz (CPU)
  3. Tokenize text (CPU)
  4. Speech tokenization (GPU)
  5. Voice encoding (GPU)
  6. Training step (GPU)
```
→ GPU waits for CPU

**After (preprocessed):**
```
For each batch:
  1. Load .pt from disk (fast!)
  2. Training step (GPU)
```
→ GPU at full utilization

## FAQ

**Q: How much disk space?**  
A: ~100MB for 5k samples (preprocessed) vs 1.4GB raw audio. Very efficient!

**Q: Can I preprocess in parallel?**  
A: Not yet in current script, but possible. Current script uses GPU sequentially.

**Q: What if I add more data?**  
A: Re-run preprocessing on new data only, append to existing preprocessed_data folder.

**Q: Does preprocessing work with data augmentation?**  
A: Not directly. For augmentation, preprocess original data, then augment during training (or preprocess augmented versions separately).

---

**Recommendation:** 

**ALWAYS use preprocessing for production training!**
- 2-4x faster
- Better GPU utilization
- One-time cost
- Much more efficient

