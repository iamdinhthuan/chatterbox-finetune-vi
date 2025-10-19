# Vietnamese TTS Fine-tuning (Chatterbox)

Fine-tune Chatterbox TTS model cho tiếng Việt với dataset của bạn.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Training Methods](#-training-methods)
  - [Simple Training](#1-simple-training-no-optimization)
  - [On-the-Fly Caching](#2-on-the-fly-caching-recommended-for-development)
  - [Pre-computed](#3-pre-computed-recommended-for-production)
- [Performance Optimization](#-performance-optimization)
- [Inference Guide](#-inference-guide)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Data

Create `metadata.csv`:

```csv
audio|transcript
vivoice_0.wav|Xin chào các bạn
vivoice_1.wav|Hôm nay trời đẹp
```

### Train (Recommended - With Caching)

```bash
python train.py \
    --csv metadata.csv \
    --audio_dir wavs \
    --use_cache \
    --cache_device cuda \
    --fp16 \
    --batch_size 8 \
    --epochs 10
```

**Expected Performance:**
- Epoch 1: 30 min (building cache)
- Epoch 2+: 6 min ⚡ (4-5x faster!)
- Total 10 epochs: ~84 min vs 300 min without cache

## 🎯 Training Methods

### 1. Simple Training (No Optimization)

```bash
python train.py \
    --csv metadata.csv \
    --audio_dir wavs \
    --batch_size 8 \
    --epochs 10
```

**Performance:**
- GPU Utilization: 40-60% (wasted!)
- Time/epoch: 30 min
- **Total 10 epochs: 300 min**

❌ **Not recommended** - GPU idle, very slow!

### 2. On-the-Fly Caching (Recommended for Development)

```bash
python train.py \
    --csv metadata.csv \
    --audio_dir wavs \
    --use_cache \
    --cache_device cuda \
    --fp16 \
    --batch_size 8 \
    --epochs 10
```

**Performance:**
- Epoch 1: 30 min (40-60% GPU) - building cache
- Epoch 2+: 6 min (95-100% GPU) ⚡ - using cache
- **Total 10 epochs: 84 min** (3.6x faster!)

**Pros:**
- ✅ Start training immediately
- ✅ Epoch 2+ very fast
- ✅ Cache persists between training sessions

**When to use:**
- Development/experimentation
- Quick iterations
- Don't want to run preprocessing separately

### 3. Pre-computed (Recommended for Production)

#### Step 1: Preprocess (once)

```bash
python preprocess_dataset.py \
    --metadata_csv metadata.csv \
    --audio_dir wavs \
    --output_dir ./preprocessed \
    --checkpoint ./vietnamese/pretrained_model_download \
    --device cuda \
    --num_workers 1
```

**Time:** ~5-10 min for 10K samples with GPU (5-10x faster than CPU!)

#### Step 2: Train with preprocessed data

Currently need to modify code:

```python
# In finetune_t3_thai.py or create new script
from chatterbox.utils.preprocessed_dataset import PrecomputedDataset

train_dataset = PrecomputedDataset(
    preprocessed_dir="./preprocessed",
    max_text_len=512,
    max_speech_len=2048,
)
```

**Performance:**
- All epochs: 6 min (95-100% GPU)
- **Total: 30 min preprocessing + 60 min training = 90 min**

**Pros:**
- ✅ All epochs fast
- ✅ Can train multiple times with same preprocessed data
- ✅ GPU preprocessing is 5-10x faster than CPU

**When to use:**
- Production/final training
- Large dataset
- Training multiple times

## 📊 Performance Comparison

| Method | Setup | Epoch 1 | Epoch 2-10 | Total (10 epochs) | GPU Util |
|--------|-------|---------|------------|-------------------|----------|
| **No optimization** | 0 | 30 min | 270 min | 300 min | 40-60% |
| **On-the-fly cache** | 0 | 30 min | 54 min | **84 min** ⚡ | 40% → 95% |
| **Pre-computed** | 30 min | 6 min | 54 min | **90 min** ⚡ | 95-100% |

**Recommendation:**
- **Development:** On-the-fly caching
- **Production:** Pre-computed with GPU preprocessing

## 🔧 Performance Optimization

### CPU Bottleneck Issue

**Problem:** GPU idle waiting for CPU to process embeddings

**Root cause:** Each sample requires 3 slow CPU operations:
1. `voice_encoder.embeds_from_wavs()` - Voice embedding
2. `speech_tokenizer.forward()` - Speech tokens  
3. `speech_tokenizer.forward()` - Conditioning tokens

**Solutions:**

#### Quick Fix: Increase Workers (CPU only)

```bash
python train.py \
    --csv metadata.csv \
    --cache_device cpu \
    --batch_size 8
    # Will automatically use num_workers=8
```

**Result:** 2-3x speedup on epoch 1

#### Better: On-the-Fly Caching with GPU

```bash
python train.py \
    --csv metadata.csv \
    --use_cache \
    --cache_device cuda \
    --fp16
    # Will automatically set num_workers=0 for CUDA compatibility
```

**Result:** 4-5x speedup from epoch 2+

#### Best: Pre-compute with GPU

```bash
python preprocess_dataset.py \
    --device cuda \
    --num_workers 1
```

**Result:** 5-10x faster preprocessing, all epochs fast

## 📝 Training Parameters

### Required
- `--csv`: Path to metadata CSV
- `--audio_dir`: Directory containing audio files

### Training
- `--batch_size`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 3)
- `--lr`: Learning rate (default: 5e-5)
- `--save_steps`: Save every N steps (default: 5000)
- `--eval_steps`: Eval every N steps (default: 5000)
- `--output_dir`: Output directory (default: ./checkpoints/vietnamese)

### Optimization
- `--use_cache`: Enable on-the-fly caching
- `--cache_dir`: Cache directory (default: ./cache)
- `--cache_device`: Device for cache - cuda or cpu (default: cuda)
- `--fp16`: Use FP16 mixed precision (faster, less VRAM)

## 🎤 Inference Guide

### Basic Inference

```bash
python infer.py \
    --checkpoint ./checkpoints/vietnamese \
    --base_model ./vietnamese/pretrained_model_download \
    --text "Xin chào các bạn" \
    --output output.wav
```

### With Custom Voice

```bash
python infer.py \
    --checkpoint ./checkpoints/vietnamese \
    --base_model ./vietnamese/pretrained_model_download \
    --voice reference.wav \
    --text "Xin chào" \
    --output output.wav
```

### Advanced Parameters

```bash
python infer.py \
    --checkpoint ./checkpoints/vietnamese \
    --text "Your text here" \
    --temperature 0.8 \
    --cfg_weight 0.5 \
    --seed 42 \
    --min_tokens 30 \
    --split_sentences \
    --fade_duration 150
```

**Parameters:**
- `--temperature`: 0.5-1.0 (higher = more natural, more variation)
- `--cfg_weight`: 0.5-1.0 (higher = follow text more closely)
- `--seed`: Random seed for reproducibility
- `--min_tokens`: Minimum tokens to prevent early stopping (default: 30)
- `--split_sentences`: Split long text by sentences (recommended for long text)
- `--fade_duration`: Cross-fade duration in ms (default: 100)

### Sentence Splitting for Long Text

For text longer than a few sentences, use `--split_sentences`:

```bash
python infer.py \
    --checkpoint ./checkpoints/vietnamese \
    --text "Câu 1. Câu 2 rất dài. Câu 3 cũng dài!" \
    --split_sentences \
    --fade_duration 150
```

**Benefits:**
- ✅ More stable for long text
- ✅ Avoid memory issues
- ✅ Can retry individual sentences
- ✅ Smooth cross-fading between sentences

## 🐛 Troubleshooting

### "Cannot re-initialize CUDA in forked subprocess"

**Solution:** Script automatically handles this by setting `num_workers=0` when using `--cache_device cuda`.

Make sure you're running:
```bash
# ✅ CORRECT
python train.py --csv ... --use_cache

# ❌ WRONG
python -c "import train; train.main()"
```

### "Expected all tensors to be on the same device"

**Fixed in latest version.** Make sure to:
```bash
git pull origin main
```

### Epoch 1 too slow?

**Normal!** Epoch 1 builds cache. Epoch 2+ will be 4-5x faster.

For faster epoch 1, use pre-computed approach:
```bash
python preprocess_dataset.py --device cuda --num_workers 1
```

### Out of GPU memory?

**Solutions:**
1. Reduce batch size:
   ```bash
   python train.py --batch_size 4
   ```

2. Use CPU cache:
   ```bash
   python train.py --cache_device cpu --batch_size 4
   ```

3. Use gradient accumulation:
   ```bash
   python train.py --batch_size 4 --gradient_accumulation_steps 4
   ```

### Out of disk space for cache?

**Cache size:** ~2-5 KB per sample
- 1K samples: ~2-5 MB
- 10K samples: ~20-50 MB
- 100K samples: ~200-500 MB

**Clear cache:**
```bash
rm -rf ./cache
```

### Cache not working?

**Check cache files:**
```bash
ls -lh ./cache | head -10
# Should see: cache_000000.pt, cache_000001.pt, ...
```

**Validate cache file:**
```bash
python -c "import torch; print(torch.load('./cache/cache_000000.pt').keys())"
```

### Audio generation too short or inconsistent?

**Use min_tokens and seed:**
```bash
python infer.py \
    --checkpoint ./checkpoints/vietnamese \
    --text "Your text" \
    --min_tokens 30 \
    --seed 42
```

**For long text, use sentence splitting:**
```bash
python infer.py \
    --text "Long text here..." \
    --split_sentences
```

## 📊 Dataset Recommendations

### Audio Quality
- **Sample rate:** 16kHz-48kHz (will be resampled to 16kHz)
- **Format:** WAV, mono
- **Duration:** 1-10 seconds per sample
- **Quality:** Clean, minimal background noise

### Dataset Size
- **Minimum:** 1,000 samples (for testing)
- **Good:** 10,000+ samples
- **Production:** 50,000+ samples

### Metadata Format

```csv
audio|transcript
wavs/audio_001.wav|Text for audio 001
wavs/audio_002.wav|Text for audio 002
subfolder/audio_003.wav|Supports subdirectories
/absolute/path/audio.wav|Or absolute paths
```

- Delimiter: `|` (pipe)
- Header line automatically skipped if contains "audio" or "transcript"
- Audio paths: relative to `--audio_dir` or absolute

## 💡 Tips & Best Practices

### Training

1. **Use FP16** for faster training:
   ```bash
   python train.py --fp16
   ```

2. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   Should see 95-100% utilization with caching!

3. **Use TensorBoard:**
   ```bash
   tensorboard --logdir ./checkpoints/vietnamese/logs
   ```

4. **Resume training:**
   ```bash
   python train.py \
       --csv metadata.csv \
       --output_dir ./checkpoints/vietnamese \
       --resume_from_checkpoint ./checkpoints/vietnamese/checkpoint-50000
   ```

### Caching

1. **Use CUDA cache for fastest:**
   ```bash
   python train.py --use_cache --cache_device cuda
   ```

2. **Cache persists - reuse it:**
   ```bash
   # First training: builds cache
   python train.py --use_cache --epochs 5
   
   # Second training: reuses cache (all epochs fast!)
   python train.py --use_cache --epochs 10
   ```

3. **Clear cache when dataset changes:**
   ```bash
   rm -rf ./cache
   ```

### Preprocessing

1. **Use GPU for preprocessing:**
   ```bash
   python preprocess_dataset.py --device cuda
   # 5-10x faster than CPU!
   ```

2. **Resume interrupted preprocessing:**
   ```bash
   # Check progress
   ls ./preprocessed/*.pt | wc -l
   
   # Resume from where it stopped
   python preprocess_dataset.py --start_idx 5000
   ```

3. **Backup preprocessed data:**
   ```bash
   tar -czf preprocessed_backup.tar.gz ./preprocessed
   ```

### Inference

1. **Reproducible generation:**
   ```bash
   python infer.py --text "..." --seed 42 --min_tokens 30
   ```

2. **For long text, always use sentence splitting:**
   ```bash
   python infer.py --text "..." --split_sentences
   ```

3. **Experiment with temperature:**
   - Low (0.5-0.7): More stable, less errors
   - Medium (0.8): Balanced (default)
   - High (0.9-1.0): More natural, more variation

## 📚 File Structure

```
chatterbox-finetuning/
├── train.py                      # Main training script
├── infer.py                      # Inference script
├── preprocess_dataset.py         # Preprocessing script
├── metadata.csv                  # Your dataset
├── wavs/                         # Audio files
├── cache/                        # On-the-fly cache (auto-created)
├── preprocessed/                 # Pre-computed data (if using)
├── checkpoints/
│   └── vietnamese/              # Trained models
├── src/
│   ├── finetune_t3_thai.py      # Core training code
│   └── chatterbox/
│       └── utils/
│           ├── cached_dataset.py           # Caching dataset
│           └── preprocessed_dataset.py     # Preprocessed dataset
└── VietnameseTokenizer/
    └── tokenizer.json           # Vietnamese tokenizer
```

## 🎓 Advanced Usage

### Custom Training Script

```python
from chatterbox.utils.cached_dataset import CachedSpeechFineTuningDataset
from transformers import Trainer, TrainingArguments

# With caching
train_dataset = CachedSpeechFineTuningDataset(
    data_args=data_args,
    t3_config=t3_config,
    hf_dataset=train_data,
    cache_dir="./cache",
    device="cuda",
)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,  # Can increase with caching!
    num_train_epochs=10,
    fp16=True,
    dataloader_num_workers=0,  # Required for CUDA caching
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Separate Train/Val CSV

```bash
python train.py \
    --train_csv train_metadata.csv \
    --val_csv val_metadata.csv \
    --audio_dir ./wavs
```

## 📈 Benchmarks

### Preprocessing Speed (10K samples)

| Device | Workers | Time |
|--------|---------|------|
| CPU | 1 | 50 min |
| CPU | 8 | 20 min |
| **GPU** | **1** | **5-10 min** ⚡ |

### Training Speed (10K samples, 10 epochs)

| Method | GPU Util | Time |
|--------|----------|------|
| No optimization | 40-60% | 300 min |
| Caching (CPU) | 40% → 95% | 105 min |
| **Caching (GPU)** | **40% → 95%** | **84 min** ⚡ |
| **Pre-computed** | **95-100%** | **90 min** ⚡ |

## 🙏 Credits

- **Chatterbox TTS:** ResembleAI
- **Vietnamese Tokenizer:** Custom-built
- **Optimization Guide:** This repo

## 📝 License

MIT License

---

**Happy Training! 🇻🇳 🚀**

For issues or questions, check [Troubleshooting](#-troubleshooting) section or open an issue on GitHub.
