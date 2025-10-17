# Vietnamese TTS Fine-tuning (Chatterbox)

Fine-tune Chatterbox TTS model cho tiếng Việt với dataset của bạn.

## 📋 Yêu cầu

- Python 3.8+
- GPU với CUDA (khuyến nghị)
- Dataset: file CSV + audio files

## 🚀 Quick Start

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Tạo file `metadata.csv` với format:

```csv
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Hôm nay trời đẹp
audio_003.wav|Tôi yêu tiếng Việt
```

**Lưu ý:**
- Delimiter: `|` (pipe)
- Audio files: `.wav` format
- Audio files nằm cùng thư mục với `metadata.csv` (hoặc chỉ định `--audio_dir`)

### 3. Tạo Vietnamese Tokenizer

```bash
python create_vietnamese_tokenizer.py
```

**Output:**
- `VietnameseTokenizer/tokenizer.json` (704 tokens)
- `VietnameseTokenizer/vocab_list.txt`

### 4. Train

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
  --lr 5e-5 \
  --fp16
```

### 5. Test

```bash
python test.py --model ./checkpoints/vietnamese --text "Xin chào"
```

## 📊 Tham số Training

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--csv` | Đường dẫn file CSV | **BẮT BUỘC** |
| `--audio_dir` | Thư mục chứa audio | `.` |
| `--output_dir` | Thư mục lưu model | `./checkpoints/vietnamese` |
| `--batch_size` | Batch size | `4` |
| `--epochs` | Số epochs | `10` |
| `--lr` | Learning rate | `5e-5` |
| `--save_steps` | Lưu mỗi N steps | `500` |
| `--eval_steps` | Eval mỗi N steps | `500` |
| `--fp16` | Dùng FP16 (nhanh hơn) | `False` |

## 📊 Tham số Testing

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--model` | Đường dẫn model | **BẮT BUỘC** |
| `--text` | Text cần synthesize | Menu chọn |
| `--output` | File output | `output.wav` |
| `--device` | Device (cuda/cpu/mps) | Auto-detect |
| `--temperature` | Temperature | `0.8` |
| `--cfg_weight` | CFG weight | `0.5` |

## 📁 Cấu trúc Project

```
.
├── create_vietnamese_tokenizer.py  # Tạo tokenizer
├── train.py                         # Training script
├── test.py                          # Testing script
├── metadata.csv                     # Dataset CSV
├── VietnameseTokenizer/
│   ├── tokenizer.json              # Vietnamese tokenizer (704 tokens)
│   └── vocab_list.txt              # Vocab list
├── checkpoints/                     # Model checkpoints
│   └── vietnamese/
└── src/
    └── finetune_t3_thai.py         # Core training code
```

## 🎯 Vietnamese Tokenizer

**Đặc điểm:**
- **704 tokens** (tương thích pretrained model)
- **Character-level** + BPE merges
- **Expressive tokens** (36): `[giggle]`, `[laughter]`, `[sigh]`, ...
- **IPA phonemes** (55): `θ`, `ʃ`, `ɑː`, `ɓ`, `ɗ`, ...
- **Từ phổ biến**: `có`, `là`, `và`, `một`, `của`, `không`, ...
- **Phụ âm đầu**: `ng`, `nh`, `th`, `ch`, `tr`, `kh`, `ph`, ...

**Cấu trúc:**
- 0-2: Special tokens ([STOP], [UNK], [SPACE])
- 3-254: Characters + Unicode
- 255: [START]
- 256-603: BPE tokens + Unicode
- 604-639: Expressive tokens
- 640-694: IPA phonemes
- 695-703: Placeholders

## 💡 Tips

### Training
- **Batch size nhỏ** nếu hết VRAM: `--batch_size 2`
- **FP16** để train nhanh hơn: `--fp16`
- **Theo dõi training**: `tensorboard --logdir ./checkpoints/vietnamese/logs`

### Dataset
- **Chất lượng audio**: 16kHz-48kHz, mono
- **Độ dài**: 1-10 giây mỗi sample
- **Số lượng**: Tối thiểu 1000 samples, khuyến nghị 10k+

### Testing
- **Temperature cao** (0.8-1.0): Tự nhiên hơn, đa dạng hơn
- **Temperature thấp** (0.5-0.7): Ổn định hơn, ít lỗi hơn
- **CFG weight**: 0.5-1.0 (cao hơn = theo text sát hơn)

## 🔧 Troubleshooting

### CUDA out of memory
```bash
python train.py --csv metadata.csv --audio_dir ./ --batch_size 2
```

### Tokenizer error
```bash
# Tạo lại tokenizer
python create_vietnamese_tokenizer.py
```

### Audio loading error
- Kiểm tra format: `.wav`
- Kiểm tra đường dẫn trong CSV
- Kiểm tra `--audio_dir`

## 📚 Tài liệu thêm

- `docs/architecture-and-data-flow.md`: Kiến trúc model
- `docs/tokenizer_analysis.md`: Phân tích tokenizer
- `src/finetune_t3_thai.py`: Source code training

## 🎉 Ví dụ

### Training với dataset lớn
```bash
python train.py \
  --csv /data/vietnamese_tts/metadata.csv \
  --audio_dir /data/vietnamese_tts/audio \
  --output_dir ./models/vietnamese_v1 \
  --batch_size 8 \
  --epochs 20 \
  --fp16 \
  --save_steps 1000
```

### Testing
```bash
# Interactive mode
python test.py --model ./models/vietnamese_v1

# Direct text
python test.py --model ./models/vietnamese_v1 --text "Xin chào Việt Nam"

# Custom output
python test.py --model ./models/vietnamese_v1 --text "Hello" --output hello.wav
```

## 📝 License

MIT License

## 🙏 Credits

- **Chatterbox TTS**: ResembleAI
- **Vietnamese Tokenizer**: Custom-built for Vietnamese language
- **Dataset**: Your own data

---

**Chúc bạn training thành công! 🇻🇳🎉**

