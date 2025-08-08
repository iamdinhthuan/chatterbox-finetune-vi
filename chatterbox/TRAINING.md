# 🔧 Vietnamese TTS Training Guide

Complete guide for training Vietnamese TTS model from scratch.

## 📋 Overview

This guide covers the complete pipeline:
1. Dataset preparation
2. Vietnamese tokenizer creation
3. Model weight extension
4. Training process
5. Evaluation and deployment

## 🗂️ Dataset Preparation

### 1. Dataset Format

Create CSV files with the following structure:

```csv
audio,transcript
wavs/audio_001.wav,Xin chào các bạn
wavs/audio_002.wav,Hôm nay là một ngày đẹp trời
wavs/audio_003.wav,Tôi rất vui được gặp các bạn
```

### 2. Audio Requirements

- **Format**: WAV (preferred), MP3, FLAC
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Duration**: 1-10 seconds per clip
- **Quality**: Clear speech, minimal background noise
- **Total**: 10,000+ clips recommended (minimum 1,000)

### 3. Text Requirements

- **Language**: Vietnamese with proper diacritics
- **Content**: Natural speech, varied vocabulary
- **Length**: 5-50 words per sentence
- **Quality**: Accurate transcription, proper punctuation

### 4. Dataset Structure

```
your-dataset/
├── train.csv          # 90% of data
├── val.csv            # 10% of data
└── wavs/              # Audio files
    ├── audio_001.wav
    ├── audio_002.wav
    └── ...
```

## 🔤 Vietnamese Tokenizer Creation

### Step 1: Extract Text Corpus

```bash
cd chatterbox
python extract_vietnamese_text.py \
    --train_csv ../train.csv \
    --val_csv ../val.csv \
    --output vietnamese_text_corpus.txt \
    --analyze
```

**Output**: `vietnamese_text_corpus.txt` containing all Vietnamese text.

### Step 2: Create Vietnamese Tokenizer

```bash
python tokenizer_scripts/make_new_tokenizer.py \
    --text_file vietnamese_text_corpus.txt \
    --vocab_size 500 \
    --output_path tokenizer_vietnamese_new.json \
    --existing_tokenizer chatterbox-project/chatterbox_weights/tokenizer.json
```

**Parameters**:
- `--vocab_size`: Number of Vietnamese tokens (500-1000 recommended)
- `--min_frequency`: Minimum token frequency (default: 2)

### Step 3: Merge Tokenizers

```bash
python tokenizer_scripts/merge_tokenizers.py \
    --tokenizer_a chatterbox-project/chatterbox_weights/tokenizer.json \
    --tokenizer_b tokenizer_vietnamese_new.json \
    --output tokenizer_vi_merged.json
```

**Result**: Combined tokenizer with ~1200 total tokens (700 English + 500 Vietnamese).

## 🔧 Model Weight Extension

### Step 4: Extend T3 Model Weights

```bash
python tokenizer_scripts/extend_tokenizer_weights.py \
    --checkpoint_path chatterbox-project/chatterbox_weights/t3_cfg.safetensors \
    --output_path t3_cfg_vietnamese.safetensors \
    --new_text_vocab_size 1200 \
    --init_method normal
```

**Parameters**:
- `--new_text_vocab_size`: Total vocabulary size (1200)
- `--init_method`: Weight initialization (`normal`, `xavier`, `kaiming`)

### Step 5: Create Model Configuration

```bash
cat > model_path_vietnamese.json << EOF
{
  "voice_encoder_path": "chatterbox-project/chatterbox_weights/ve.safetensors",
  "t3_path": "t3_cfg_vietnamese.safetensors",
  "s3gen_path": "chatterbox-project/chatterbox_weights/s3gen.safetensors",
  "tokenizer_path": "$(pwd)/tokenizer_vi_merged.json",
  "conds_path": "chatterbox-project/chatterbox_weights/conds.pt"
}
EOF
```

## 🚀 Training Process

### Step 6: Validate Dataset

```bash
python check_dataset.py \
    --train_csv ../train.csv \
    --val_csv ../val.csv \
    --max_samples 100
```

### Step 7: Start Training

#### Basic Training:
```bash
python train_vietnamese_csv.py \
    --model_config model_path_vietnamese.json \
    --train_csv ../train.csv \
    --val_csv ../val.csv \
    --output_dir checkpoints/vietnamese_tts \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

#### Advanced Training:
```bash
python train_vietnamese_csv.py \
    --model_config model_path_vietnamese.json \
    --train_csv ../train.csv \
    --val_csv ../val.csv \
    --output_dir checkpoints/vietnamese_tts \
    --num_train_epochs 5 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_steps 2000 \
    --fp16 \
    --dataloader_num_workers 4
```

## 📊 Training Parameters

### Core Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `num_train_epochs` | Training epochs | 3-5 |
| `per_device_train_batch_size` | Batch size per GPU | 4-8 |
| `gradient_accumulation_steps` | Gradient accumulation | 2-4 |
| `learning_rate` | Learning rate | 3e-5 to 5e-5 |
| `warmup_steps` | Warmup steps | 500-1000 |

### Data Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `max_text_len` | Max text tokens | 256 |
| `max_speech_len` | Max speech tokens | 800 |
| `dataloader_num_workers` | Data loading workers | 4-8 |

### Optimization Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `fp16` | Mixed precision | True |
| `weight_decay` | Weight decay | 0.01 |
| `adam_epsilon` | Adam epsilon | 1e-8 |

## 📈 Monitoring Training

### TensorBoard

```bash
# In another terminal
tensorboard --logdir checkpoints/vietnamese_tts/runs
```

Open http://localhost:6006 to view training metrics.

### Key Metrics to Watch

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without overfitting
- **Learning Rate**: Should follow warmup schedule
- **GPU Memory**: Should be stable

### Expected Training Time

| Dataset Size | GPU | Estimated Time |
|--------------|-----|----------------|
| 10K samples | RTX 3070 | 4-6 hours |
| 50K samples | RTX 3070 | 12-18 hours |
| 100K samples | RTX 4090 | 8-12 hours |

## 🎯 Training Tips

### For Better Results

1. **Data Quality**:
   - Use high-quality audio recordings
   - Ensure accurate transcriptions
   - Balance speaker diversity

2. **Hyperparameter Tuning**:
   - Start with recommended parameters
   - Adjust batch size based on GPU memory
   - Monitor validation loss for overfitting

3. **Regularization**:
   - Use dropout if overfitting
   - Apply weight decay
   - Early stopping based on validation loss

### Common Issues

#### Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
```

#### Slow Training
```bash
# Increase workers and use mixed precision
--dataloader_num_workers 8
--fp16
```

#### Poor Quality
- Check dataset quality
- Increase training epochs
- Adjust learning rate

## ✅ Post-Training

### 1. Model Evaluation

```bash
python evaluate_model.py \
    --model_path checkpoints/vietnamese_tts/model.safetensors \
    --test_csv test.csv
```

### 2. Export Final Model

```bash
# Copy best checkpoint
cp checkpoints/vietnamese_tts/checkpoint-XXXX/model.safetensors model.safetensors
```

### 3. Test Inference

```bash
python vietnamese_tts_inference.py \
    --text "Xin chào, đây là model TTS tiếng Việt mới!"
```

## 🚀 Next Steps

1. **Fine-tuning**: Adjust for specific voices or domains
2. **Voice Cloning**: Test with reference audio
3. **Production**: Deploy with optimizations
4. **Evaluation**: Comprehensive quality assessment

---

**Need help with training?** Check our [troubleshooting guide](TROUBLESHOOTING.md) or open an issue!
