# Vietnamese TTS Training - Complete Setup Guide

Hướng dẫn hoàn chỉnh để setup và training model ChatterboxTTS cho tiếng Việt từ đầu.

## 🎯 Tổng quan

Quy trình này sẽ:
1. **Tạo tokenizer tiếng Việt** từ text corpus
2. **Merge với tokenizer gốc** để giữ khả năng đa ngôn ngữ
3. **Extend model weights** để phù hợp với vocabulary size mới
4. **Setup training** với dataset CSV

## 📋 Yêu cầu

### Files cần có:
```
project/
├── train.csv                 # Training data
├── val.csv                   # Validation data
├── wavs/                     # Audio files directory
└── chatterbox-project/
    └── chatterbox_weights/
        ├── tokenizer.json    # Original tokenizer
        ├── t3_cfg.safetensors # Original T3 model
        ├── ve.safetensors    # Voice encoder
        ├── s3gen.safetensors # Speech generator
        └── conds.pt          # Conditions (optional)
```

### Dependencies:
```bash
pip install pandas torch transformers tokenizers safetensors librosa tqdm
```

## 🚀 Quick Start

### Chạy setup tự động:
```bash
python run_vietnamese_setup.py
```

Script này sẽ tự động thực hiện toàn bộ quy trình và báo cáo kết quả.

## 📝 Manual Setup (từng bước)

### Bước 1: Tạo text corpus
```bash
# Trích xuất text từ CSV files
python extract_vietnamese_text.py \
    --train_csv train.csv \
    --val_csv val.csv \
    --output vietnamese_text_corpus.txt \
    --analyze
```

### Bước 2: Setup tokenizer và model
```bash
# Tạo tokenizer, merge và extend weights
python setup_vietnamese_training.py \
    --text_file vietnamese_text_corpus.txt \
    --vocab_size 500
```

### Bước 3: Kiểm tra dataset
```bash
# Validate dataset trước khi training
python check_dataset.py \
    --train_csv train.csv \
    --val_csv val.csv
```

### Bước 4: Bắt đầu training
```bash
# Start training với config mới
python train_vietnamese_csv.py \
    --model_config model_path_vietnamese.json \
    --train_csv train.csv \
    --val_csv val.csv \
    --output_dir checkpoints/vietnamese_tts
```

## 🔧 Chi tiết các bước

### 1. Text Corpus Creation
- Trích xuất text từ CSV files
- Làm sạch và chuẩn hóa text tiếng Việt
- Phân tích thống kê corpus

### 2. Vietnamese Tokenizer
- Tạo BPE tokenizer cho tiếng Việt
- Vocab size: 500 tokens (có thể điều chỉnh)
- Tránh overlap với tokenizer gốc

### 3. Tokenizer Merging
- Merge Vietnamese tokenizer với original tokenizer
- Giữ nguyên special tokens
- Tổng vocab size: ~1200+ tokens

### 4. Model Weight Extension
- Extend `text_emb.weight` và `text_head.weight`
- Initialize new tokens với normal distribution
- Backup original weights

### 5. Training Configuration
- Tạo `model_path_vietnamese.json`
- Point đến extended model và merged tokenizer
- Ready cho training

## 📊 Monitoring Training

### TensorBoard:
```bash
tensorboard --logdir checkpoints/vietnamese_tts/runs
```

### Training logs:
```bash
tail -f training.log
```

## ⚙️ Tham số tối ưu

### Tokenizer:
- `vocab_size`: 500-1000 (tùy dataset size)
- `min_frequency`: 2 (filter rare tokens)

### Training:
- `num_train_epochs`: 3-5
- `per_device_train_batch_size`: 4-8 (tùy GPU)
- `learning_rate`: 3e-5 đến 5e-5
- `warmup_steps`: 500-1000

### Model Extension:
- `init_method`: "normal" (recommended)
- `backup_original`: True (safety)

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **"tokenizer.json not found"**
   ```bash
   # Download model weights trước
   python download_hf_repo.py
   ```

2. **"Vocab size mismatch"**
   ```bash
   # Re-run weight extension
   python setup_vietnamese_training.py --vocab_size 500
   ```

3. **"Out of memory during training"**
   ```bash
   # Giảm batch size
   python train_vietnamese_csv.py --per_device_train_batch_size 2
   ```

4. **"Text encoding errors"**
   ```bash
   # Re-extract text corpus
   python extract_vietnamese_text.py --analyze
   ```

### Debug commands:

```bash
# Kiểm tra tokenizer
python tokenizer_scripts/examine_tokenizer.py tokenizer_vi_merged.json

# Kiểm tra model weights
python -c "
import torch
from safetensors import safe_open
with safe_open('t3_cfg_vietnamese.safetensors', framework='pt') as f:
    print('text_emb.weight shape:', f.get_tensor('text_emb.weight').shape)
"

# Test tokenization
python -c "
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('tokenizer_vi_merged.json')
text = 'Xin chào, tôi là trợ lý AI'
tokens = tokenizer.encode(text)
print('Tokens:', tokens.tokens)
print('IDs:', tokens.ids)
"
```

## 📈 Expected Results

### Tokenizer Stats:
- Original vocab: ~703 tokens
- Vietnamese vocab: ~500 tokens  
- Merged vocab: ~1200+ tokens
- Special tokens: 4

### Training Progress:
- Initial loss: ~8-10
- Target loss: <2.0
- Training time: 2-4 hours (tùy dataset và GPU)

## 🎯 Next Steps

Sau khi training xong:

1. **Test inference:**
   ```bash
   python inference_simple.py --model_config model_path_vietnamese.json
   ```

2. **Evaluate quality:**
   - Nghe audio output
   - Check pronunciation accuracy
   - Test với different text lengths

3. **Fine-tune parameters:**
   - Adjust learning rate
   - Increase epochs if needed
   - Try different batch sizes

## 📞 Support

Nếu gặp vấn đề:
1. Check logs trong `training.log`
2. Verify file paths trong config
3. Test với smaller dataset trước
4. Monitor GPU memory usage

Chúc bạn training thành công! 🎵🇻🇳
