# Vietnamese TTS Fine-tuning (Chatterbox)

Fine-tune Chatterbox TTS model cho tiếng Việt với dataset của bạn.

---

## 🚀 Quick Start (3 bước)

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Tạo Tokenizer từ Corpus
```bash
python train_tokenizer_from_corpus.py metadata.csv
```

**Output:**
- `VietnameseTokenizer/tokenizer.json` - Trained tokenizer
- `VietnameseTokenizer/vocab_list.txt` - Human-readable vocab

**Đặc điểm:**
- 703 tokens (49 special + 654 Vietnamese)
- Learns BPE merges từ YOUR data
- 0% OOV trên training corpus
- Preserves special tokens từ pretrained model

### 3. Train TTS Model
```bash
python train.py --csv metadata.csv --audio_dir ./
```

**Tùy chọn:**
```bash
python train.py \
  --csv metadata.csv \
  --audio_dir /path/to/audio \
  --output_dir ./checkpoints/my_model \
  --batch_size 4 \
  --epochs 10 \
  --lr 5e-5
```

### 4. Test Model
```bash
python test.py --model ./checkpoints/vietnamese --text "Xin chào"
```

---

## 📋 Yêu cầu

- Python 3.8+
- GPU với CUDA (khuyến nghị)
- Dataset: metadata.csv + audio files (.wav)

---

## 📝 Chuẩn bị Dataset

### Format metadata.csv:

```csv
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Hôm nay trời đẹp
audio_003.wav|Tôi yêu tiếng Việt
```

**Lưu ý:**
- Delimiter: `|` (pipe)
- Audio files: `.wav` format, 16kHz-48kHz, mono
- Độ dài audio: 1-10 giây/sample
- Số lượng: Tối thiểu 1,000 samples, khuyến nghị 10,000+
- Audio files cùng thư mục với CSV (hoặc dùng `--audio_dir`)

---

## 🎯 Vietnamese Tokenizer Training

### Cách hoạt động:

Script `train_tokenizer_from_corpus.py` sẽ:
1. Load texts từ metadata.csv
2. Train BPE tokenizer từ YOUR Vietnamese data
3. Preserve tất cả special tokens từ pretrained model
4. Save tokenizer tại `VietnameseTokenizer/`

### Kết quả:

- **Vocab size**: 703 tokens
- **Special tokens**: 49 (preserved at original positions)
  - 0-2: [STOP], [UNK], [SPACE]
  - 255: [START]
  - 604-639: Expressive ([giggle], [laughter], [whisper]...)
  - 695-703: Placeholders
- **Vietnamese tokens**: 654 (learned from corpus)
- **BPE merges**: ~465 merge operations
- **OOV rate**: 0% on training corpus
- **Efficiency**: 50% better than character-level

### Example:
```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("VietnameseTokenizer/tokenizer.json")

# Vietnamese text
encoding = tokenizer.encode("Tiếng Việt rất hay")
print(encoding.tokens)
# → ['T', 'iế', 'ng', 'V', 'iệt', 'rất', 'hay']  # 7 tokens

# Special tokens work
encoding = tokenizer.encode("[giggle] Xin chào [whisper]")
print(encoding.tokens)
# → ['[giggle]', 'X', 'in', 'ch', 'ào', '[whisper]']
```

---

## 📊 Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--csv` | Path to metadata CSV | **REQUIRED** |
| `--audio_dir` | Audio directory | `.` |
| `--output_dir` | Output checkpoint directory | `./checkpoints/vietnamese` |
| `--batch_size` | Batch size | `8` |
| `--gradient_accumulation_steps` | Gradient accumulation | `1` |
| `--epochs` | Number of epochs | `3` |
| `--lr` | Learning rate | `1e-5` |
| `--save_steps` | Save checkpoint every N steps | `5000` |
| `--eval_steps` | Evaluate every N steps | `5000` |
| `--max_steps` | Max training steps (-1 = full) | `-1` |

### Training Examples:

**Basic:**
```bash
python train.py --csv metadata.csv --audio_dir ./
```

**Memory optimization (low VRAM):**
```bash
python train.py --csv metadata.csv --batch_size 2 --gradient_accumulation_steps 4
```

**Long training:**
```bash
python train.py \
  --csv metadata.csv \
  --audio_dir /data/audio \
  --output_dir ./models/vietnamese_v1 \
  --batch_size 8 \
  --epochs 20 \
  --save_steps 1000
```

**Separate train/val files:**
```bash
python train.py --train_csv train.csv --val_csv val.csv --audio_dir ./audio
```

---

## 🧪 Testing & Inference

### Interactive mode:
```bash
python test.py --model ./checkpoints/vietnamese
```

### Direct text:
```bash
python test.py --model ./checkpoints/vietnamese --text "Xin chào Việt Nam"
```

### Custom output:
```bash
python test.py \
  --model ./checkpoints/vietnamese \
  --text "Hello" \
  --output hello.wav \
  --temperature 0.8 \
  --cfg_weight 0.5
```

### Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model directory | **REQUIRED** |
| `--text` | Text to synthesize | Interactive |
| `--output` | Output WAV file | `output.wav` |
| `--device` | Device (cuda/cpu/mps) | Auto-detect |
| `--temperature` | Temperature (0.5-1.0) | `0.8` |
| `--cfg_weight` | CFG weight (0.5-1.0) | `0.5` |

**Tips:**
- **Temperature cao** (0.8-1.0): Tự nhiên hơn, đa dạng hơn
- **Temperature thấp** (0.5-0.7): Ổn định hơn, ít lỗi hơn
- **CFG weight cao**: Theo text sát hơn

---

## 📁 Project Structure

```
chatterbox-finetuning/
│
├── train_tokenizer_from_corpus.py  # Train tokenizer from corpus
├── train.py                         # Main training script
├── test.py                          # Testing/inference
├── test_oov.py                      # Test OOV coverage (optional)
│
├── tokenizer.json                   # Original pretrained tokenizer (input)
├── metadata.csv                     # Your dataset (input)
│
├── VietnameseTokenizer/            # Trained tokenizer (output)
│   ├── tokenizer.json              # Use this for training
│   └── vocab_list.txt              # Human-readable vocab
│
├── checkpoints/                    # Model checkpoints (output)
│   └── vietnamese/
│       ├── checkpoint-N/
│       └── logs/                   # TensorBoard logs
│
└── src/                            # Source code
    ├── finetune_t3_thai.py         # Core training logic
    └── chatterbox/                 # Chatterbox TTS library
```

---

## 💡 Tips & Best Practices

### Dataset:
- ✅ Chất lượng audio cao, ít noise
- ✅ Giọng đọc rõ ràng, tự nhiên
- ✅ Độ dài 1-10 giây/sample
- ✅ Tối thiểu 1,000 samples, khuyến nghị 10,000+
- ✅ Text chuẩn, ít typo

### Tokenizer:
- ✅ Train tokenizer từ corpus TRƯỚC khi train model
- ✅ Test OOV coverage: `python test_oov.py`
- ✅ Nên có 0% OOV rate

### Training:
- ✅ Bắt đầu với batch_size nhỏ (2-4) nếu VRAM thấp
- ✅ Dùng gradient_accumulation để tăng effective batch size
- ✅ Monitor loss với TensorBoard: `tensorboard --logdir checkpoints/vietnamese/logs`
- ✅ Save checkpoints thường xuyên (mỗi 1000-5000 steps)
- ✅ Training time: ~1-3 ngày cho 10k samples trên 1 GPU

### Inference:
- ✅ Test nhiều temperature values để tìm giá trị tốt nhất
- ✅ Temperature = 0.8 thường cho kết quả tốt nhất
- ✅ Nếu output bị lỗi, giảm temperature xuống 0.6-0.7

---

## 🔧 Troubleshooting

### CUDA out of memory
```bash
# Giảm batch size
python train.py --csv metadata.csv --batch_size 2

# Hoặc dùng gradient accumulation
python train.py --csv metadata.csv --batch_size 2 --gradient_accumulation_steps 4
```

### Tokenizer not found
```bash
# Tạo tokenizer từ corpus
python train_tokenizer_from_corpus.py metadata.csv

# Check output
ls VietnameseTokenizer/
```

### High OOV rate
```bash
# Test OOV coverage
python test_oov.py

# Re-train tokenizer với cleaned corpus
python train_tokenizer_from_corpus.py metadata_cleaned.csv
```

### Audio loading error
- ✅ Check audio format: Phải là `.wav`
- ✅ Check đường dẫn trong CSV: Relative hoặc absolute
- ✅ Check `--audio_dir` parameter
- ✅ Test: `ls audio/*.wav | head`

### Tokenizer error: Special tokens missing
```bash
# Re-train với original tokenizer.json
python train_tokenizer_from_corpus.py metadata.csv tokenizer.json
```

### Training loss không giảm
- ✅ Check learning rate (thử 1e-5 hoặc 5e-5)
- ✅ Check dataset quality
- ✅ Tăng số epochs
- ✅ Check tokenizer coverage

---

## 📊 Monitoring Training

### TensorBoard:
```bash
tensorboard --logdir ./checkpoints/vietnamese/logs
```
Mở: http://localhost:6006

### Check checkpoints:
```bash
ls -lh checkpoints/vietnamese/checkpoint-*/
```

### Test intermediate checkpoints:
```bash
python test.py --model ./checkpoints/vietnamese/checkpoint-10000 --text "Test"
```

---

## 📚 Additional Info

### Architecture:
- **T3 Model**: LLaMA-based (520M params) - Text to Speech Tokens
- **S3Gen**: Flow-based model - Speech Tokens to Waveform
- **Fine-tuning**: Only T3, freeze Voice Encoder & S3Gen

### Tokenizer Details:
- **Type**: Byte Pair Encoding (BPE)
- **Min frequency**: 2 (only learns tokens appearing ≥2 times)
- **Pre-tokenizer**: Whitespace splitting
- **Language**: Vietnamese (vi)

### Training Strategy:
- Freeze voice encoder & S3Gen
- Only train T3 text encoder
- Use cosine learning rate schedule
- Gradient clipping (max_grad_norm=1.0)
- Weight decay = 0.01

---

## 🎉 Examples

### Complete workflow:

```bash
# 1. Prepare data
cat > metadata.csv << EOF
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Đây là tiếng Việt
audio_003.wav|Tôi yêu TTS
EOF

# 2. Train tokenizer
python train_tokenizer_from_corpus.py metadata.csv

# 3. Check tokenizer
python test_oov.py
# Expected: 0% OOV, 0 [UNK] tokens

# 4. Train model
python train.py \
  --csv metadata.csv \
  --audio_dir ./ \
  --batch_size 4 \
  --epochs 10

# 5. Monitor
tensorboard --logdir checkpoints/vietnamese/logs

# 6. Test
python test.py --model checkpoints/vietnamese --text "Xin chào Việt Nam"
```

---

## 📝 License

MIT License

## 🙏 Credits

- **Chatterbox TTS**: ResembleAI
- **Base Model**: tel4vn/chatterxbox (pretrained)
- **Vietnamese Tokenizer**: Corpus-based BPE training

---

## 📧 Support

- Issues: https://github.com/iamdinhthuan/chatterbox-finetune-vi/issues
- For development guidelines: See `WARP.md`

---

**Happy Training! 🇻🇳🎉**
