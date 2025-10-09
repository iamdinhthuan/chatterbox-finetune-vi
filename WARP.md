# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Tổng quan dự án

Đây là dự án fine-tuning mô hình **Chatterbox TTS** (ResembleAI) cho tiếng Việt. Chatterbox sử dụng kiến trúc 2-stage: T3 model (LLaMA-based, 520M params) để chuyển text thành speech tokens, và S3Gen (flow-based model) để tạo waveform.

**Đặc điểm riêng của dự án:**
- Vietnamese tokenizer tùy chỉnh (704 tokens)
- Chỉ fine-tune T3, freeze Voice Encoder và S3Gen
- Hỗ trợ cả single CSV (auto-split) hoặc separate train/val files
- PyTorch 2.6 với custom checkpoint loading để tránh lỗi serialization

## Lệnh thường dùng

### Setup ban đầu
```powershell
# Cài đặt dependencies
pip install -r requirements.txt

# Tạo Vietnamese tokenizer (bắt buộc trước khi train)
python create_vietnamese_tokenizer.py
```

### Training

**Basic training (single CSV file):**
```powershell
python train.py --csv metadata.csv --audio_dir ./
```

**Training với separate train/val files:**
```powershell
python train.py --train_csv train.csv --val_csv val.csv --audio_dir ./audio
```

**Training với custom parameters:**
```powershell
python train.py `
  --csv metadata.csv `
  --audio_dir ./audio `
  --output_dir ./checkpoints/my_model `
  --batch_size 4 `
  --epochs 10 `
  --lr 5e-5 `
  --save_steps 5000 `
  --eval_steps 5000 `
  --max_steps 100000
```

**Training với memory optimization:**
```powershell
# Giảm batch size nếu CUDA out of memory
python train.py --csv metadata.csv --audio_dir ./ --batch_size 2 --gradient_accumulation_steps 4
```

### Testing & Inference

**Test model đã train xong:**
```powershell
python test.py --model ./checkpoints/vietnamese --text "Xin chào Việt Nam"
```

**Inference từ checkpoint cụ thể:**
```powershell
python infer.py `
  --checkpoint ./checkpoints/vietnamese/checkpoint-45000 `
  --base_model ./checkpoints/vietnamese/pretrained_model_download `
  --text "Xin chào" `
  --output output.wav
```

**Batch inference:**
```powershell
python batch_infer.py `
  --checkpoint ./checkpoints/vietnamese `
  --input texts.txt `
  --output_dir ./outputs `
  --prefix audio
```

### Monitoring

**TensorBoard:**
```powershell
tensorboard --logdir ./checkpoints/vietnamese/logs
```

**Check checkpoint structure:**
```powershell
python check_checkpoint.py ./checkpoints/vietnamese/checkpoint-45000/model.safetensors
```

## Kiến trúc & Data Flow

### High-level Architecture
```
Text Input → Text Normalizer → T3 Model (trainable) → Speech Tokens → S3Gen (frozen) → Audio
                                    ↑
                            Voice Encoder (frozen)
                                    ↑
                            Reference Audio
```

### Components

**T3 Model (Text-to-Token-to-Speech)**
- 520M parameter LLaMA variant
- Input: normalized text + speaker embedding + prompt tokens
- Output: discrete speech tokens
- **Training target**: Chỉ component này được fine-tune

**S3Gen (Speech Synthesis Generator)**
- Flow-based generative model + HiFi-GAN vocoder
- Input: speech tokens + speaker embedding + prompt mel
- Output: 24kHz waveform
- **Frozen** trong quá trình training

**Voice Encoder**
- Extracts speaker embeddings từ reference audio
- **Frozen** trong quá trình training

**Vietnamese Tokenizer**
- 704 tokens total
- Character-level + BPE merges
- Bao gồm: Vietnamese characters, IPA phonemes, expressive tokens

### Training Data Processing

**Dataset format (metadata.csv):**
```csv
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Hôm nay trời đẹp
```

**Processing pipeline:**
1. Load audio → resample to 16kHz
2. Extract speaker embedding (Voice Encoder - frozen)
3. Normalize text → tokenize (Vietnamese tokenizer)
4. Extract speech tokens (S3 Tokenizer - frozen)
5. Create prompt segments (first 3s of audio)
6. Feed to T3 for training

**Sample rates:**
- S3 Tokenizer: 16kHz
- Voice Encoder: 16kHz
- Mel Extractor: 24kHz
- Final Audio: 24kHz

### File Structure

```
D:\TTS\chatterbox-finetuning\
├── train.py                    # Main training script (wrapper)
├── test.py                     # Testing script
├── infer.py                    # Inference from specific checkpoint
├── batch_infer.py              # Batch inference
├── create_vietnamese_tokenizer.py
├── metadata.csv                # Dataset file
├── VietnameseTokenizer/
│   ├── tokenizer.json          # 704 tokens
│   └── vocab_list.txt
├── src/
│   ├── finetune_t3_thai.py     # Core training logic
│   └── chatterbox/             # Chatterbox TTS library
└── checkpoints/
    └── vietnamese/
        ├── pretrained_model_download/  # Base model
        ├── checkpoint-N/               # Training checkpoints
        └── logs/                       # TensorBoard logs
```

## Core Training Code

**Main training function:** `src/finetune_t3_thai.py:run_training()`

**Key dataset classes:**
- `SpeechFineTuningDataset`: Standard dataset for normal training
- `SpeechFineTuningIterableDataset`: Streaming dataset cho large datasets

**Custom Trainer:** `SafeCheckpointTrainer`
- Handles PyTorch 2.6 checkpoint loading với numpy dtype compatibility
- Override `_load_rng_state()` để load với `weights_only=False`

**Data collator:** `SpeechDataCollator`
- Pads text và speech tokens
- Tạo labels cho text và speech prediction

## Lưu ý đặc biệt

### PyTorch 2.6 Compatibility
- Dự án có custom handling cho PyTorch 2.6 serialization issues
- `SafeCheckpointTrainer` loads checkpoints với `weights_only=False`
- Safe globals được add cho numpy dtypes

### Vietnamese Text Normalization
Text được normalize trước khi tokenize:
- Lowercase
- Unicode NFC normalization
- Remove unsupported punctuation
- Keep only: Vietnamese chars, numbers, basic punctuation (. , ! ? -)

### Memory Management
- Default batch_size=8, có thể giảm xuống 2 nếu OOM
- Sử dụng gradient_accumulation_steps để tăng effective batch size
- bf16 training enabled by default

### Checkpoint Structure
- Training checkpoints: chỉ có T3 weights với prefix "t3."
- Final model: có đầy đủ ve.safetensors, t3_cfg.safetensors, s3gen.safetensors, tokenizer.json
- Script tự động copy frozen components vào output_dir sau training

### Dataset Options
1. **Single CSV**: Provide `--csv` → auto-split với `eval_split_size` (default 0.0005)
2. **Separate files**: Provide `--train_csv` và `--val_csv` → không split

## Troubleshooting

**CUDA out of memory:**
```powershell
python train.py --csv metadata.csv --batch_size 2 --gradient_accumulation_steps 4
```

**Tokenizer not found:**
```powershell
python create_vietnamese_tokenizer.py
```

**Checkpoint loading error:**
- Dự án đã có SafeCheckpointTrainer để handle PyTorch 2.6 issues
- Nếu vẫn lỗi, check rng_state.pth file trong checkpoint

**Audio loading error:**
- Check audio format: phải là .wav
- Check đường dẫn trong CSV: relative hoặc absolute
- Check `--audio_dir` parameter

## Parameters Reference

**Training arguments quan trọng:**
- `--csv` hoặc `--train_csv`/`--val_csv`: Dataset files
- `--audio_dir`: Thư mục chứa audio (default: ".")
- `--output_dir`: Nơi lưu checkpoints (default: "./checkpoints/vietnamese")
- `--batch_size`: Batch size (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--epochs`: Number of epochs (default: 3)
- `--lr`: Learning rate (default: 5e-5)
- `--save_steps`: Save checkpoint mỗi N steps (default: 5000)
- `--eval_steps`: Evaluate mỗi N steps (default: 5000)
- `--max_steps`: Max training steps, override epochs nếu > 0 (default: -1)

**Data arguments trong finetune_t3_thai.py:**
- `max_text_len`: 256 (max text token length)
- `max_speech_len`: 1200 (max speech token length)
- `audio_prompt_duration_s`: 3.0 (duration for conditioning)
- `eval_split_size`: 0.99 (fraction for eval when using single CSV)

**Model freezing:**
- `freeze_voice_encoder`: True (always frozen)
- `freeze_s3gen`: True (always frozen)
