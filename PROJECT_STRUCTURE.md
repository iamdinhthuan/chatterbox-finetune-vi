# Project Structure

## 📁 Cấu trúc thư mục

```
chatterbox-finetuning/
│
├── 📄 README.md                      # Hướng dẫn chính
├── 📄 QUICKSTART.md                  # Quick start guide
├── 📄 SERVER_SETUP.md                # Hướng dẫn setup server
├── 📄 PROJECT_STRUCTURE.md           # File này
│
├── 🔧 requirements.txt               # Python dependencies
├── 🔧 pyproject.toml                 # Project config
│
├── 🐍 create_vietnamese_tokenizer.py # Tạo Vietnamese tokenizer
├── 🐍 train.py                       # Training script
├── 🐍 test.py                        # Testing script
│
├── 📊 metadata.csv                   # Dataset CSV (example)
│
├── 📁 VietnameseTokenizer/           # Vietnamese tokenizer
│   ├── tokenizer.json                # Tokenizer file (704 tokens)
│   └── vocab_list.txt                # Vocabulary list
│
├── 📁 src/                           # Source code
│   ├── chatterbox/                   # Chatterbox TTS core
│   ├── finetune_t3.py                # T3 finetuning (English)
│   ├── finetune_t3_thai.py           # T3 finetuning (Thai/Vietnamese)
│   └── ...
│
├── 📁 docs/                          # Documentation
│   ├── architecture-and-data-flow.md # Model architecture
│   └── tokenizer_analysis.md        # Tokenizer analysis
│
├── 📁 cache/                         # Pretrained model cache
│   └── models--ResembleAI--chatterbox/
│
├── 📁 checkpoints/                   # Training checkpoints (tạo khi train)
│   └── vietnamese/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── logs/
│
└── 📁 notebooks/                     # Jupyter notebooks (optional)
    └── dataset.ipynb
```

## 📄 Files chính

### 1. Scripts

| File | Mô tả | Khi nào dùng |
|------|-------|--------------|
| `create_vietnamese_tokenizer.py` | Tạo Vietnamese tokenizer | 1 lần duy nhất trước khi train |
| `train.py` | Training script | Để train model |
| `test.py` | Testing script | Sau khi train xong |

### 2. Documentation

| File | Mô tả |
|------|-------|
| `README.md` | Hướng dẫn đầy đủ |
| `QUICKSTART.md` | Quick start (3 bước) |
| `SERVER_SETUP.md` | Setup trên server |
| `PROJECT_STRUCTURE.md` | Cấu trúc project |

### 3. Config

| File | Mô tả |
|------|-------|
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Project metadata |

## 📊 Dataset

### Format

File `metadata.csv`:
```csv
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Hôm nay trời đẹp
```

### Vị trí

- **CSV file**: Đặt ở root hoặc bất kỳ đâu (chỉ định với `--csv`)
- **Audio files**: Cùng thư mục với CSV hoặc chỉ định với `--audio_dir`

## 🔤 Vietnamese Tokenizer

### Files

- `VietnameseTokenizer/tokenizer.json`: Tokenizer chính (704 tokens)
- `VietnameseTokenizer/vocab_list.txt`: Danh sách vocabulary

### Cấu trúc (704 tokens)

| Range | Nội dung | Số lượng |
|-------|----------|----------|
| 0-2 | Special tokens | 3 |
| 3-254 | Characters + Unicode | 252 |
| 255 | [START] | 1 |
| 256-603 | BPE tokens + Unicode | 348 |
| 604-639 | Expressive tokens | 36 |
| 640-694 | IPA phonemes | 55 |
| 695-703 | Placeholders | 9 |

## 💾 Checkpoints

### Cấu trúc

```
checkpoints/vietnamese/
├── checkpoint-500/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
├── checkpoint-1000/
├── checkpoint-1500/
└── logs/
    └── events.out.tfevents...
```

### Quản lý

- **Auto-save**: Mỗi `--save_steps` (mặc định: 500)
- **Xóa checkpoints cũ**: Giữ lại 3-5 checkpoints mới nhất
- **Backup**: Nén và download về local

## 🔧 Source Code

### Core files

| File | Mô tả |
|------|-------|
| `src/finetune_t3_thai.py` | Core training logic |
| `src/chatterbox/` | Chatterbox TTS model |

### Không cần sửa

Các file trong `src/` đã được config sẵn, không cần sửa trừ khi muốn customize.

## 📚 Docs

| File | Nội dung |
|------|----------|
| `docs/architecture-and-data-flow.md` | Kiến trúc Chatterbox TTS |
| `docs/tokenizer_analysis.md` | Phân tích tokenizer |

## 🗑️ Files có thể xóa

### Notebooks (optional)
```bash
rm -rf notebooks/
```

### Thai-related files (nếu không dùng)
```bash
rm src/thai_dataset_adapter.py
rm src/finetune_t3_thai_template.py
```

### Docs (nếu không cần)
```bash
rm -rf docs/
```

## 📦 Minimal Setup

Nếu chỉ muốn train, chỉ cần:

```
chatterbox-finetuning/
├── create_vietnamese_tokenizer.py
├── train.py
├── test.py
├── requirements.txt
├── metadata.csv
├── audio/
├── VietnameseTokenizer/
└── src/
```

## 🚀 Workflow

```
1. create_vietnamese_tokenizer.py
   ↓
2. train.py
   ↓
3. checkpoints/vietnamese/
   ↓
4. test.py
```

## 💡 Tips

### Tổ chức dataset lớn

```
data/
├── metadata.csv
└── audio/
    ├── speaker_001/
    │   ├── audio_001.wav
    │   └── audio_002.wav
    └── speaker_002/
        ├── audio_001.wav
        └── audio_002.wav
```

### Backup checkpoints

```bash
# Nén
tar -czf vietnamese_v1.tar.gz checkpoints/vietnamese/

# Download
scp user@server:/path/to/vietnamese_v1.tar.gz ./
```

### Clean up

```bash
# Xóa checkpoints cũ (giữ 3 mới nhất)
cd checkpoints/vietnamese/
ls -t checkpoint-* | tail -n +4 | xargs rm -rf

# Xóa cache
rm -rf cache/
```

---

**Đọc thêm:**
- `README.md`: Hướng dẫn đầy đủ
- `QUICKSTART.md`: Quick start
- `SERVER_SETUP.md`: Setup server

