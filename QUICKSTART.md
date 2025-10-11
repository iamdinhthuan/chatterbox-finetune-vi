# Quick Start - Vietnamese TTS Training

## 🚀 3 Bước Đơn Giản

### Bước 1: Cài đặt
```bash
pip install -r requirements.txt
```

### Bước 2: Tạo Tokenizer
```bash
python train_tokenizer_from_corpus.py metadata.csv
```
(Learns từ YOUR data, 0% OOV)

### Bước 3: Train
```bash
python train.py --csv metadata.csv --audio_dir ./
```

---

## 📝 Format Dataset

File `metadata.csv`:
```csv
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Hôm nay trời đẹp
```

**Lưu ý:**
- Delimiter: `|` (pipe character)
- Audio: `.wav` format
- Audio files cùng thư mục với CSV

---

## 🎯 Test Model

```bash
python test.py --model ./checkpoints/vietnamese --text "Xin chào"
```

---

## ⚙️ Tùy chọn nâng cao

### Training nhanh hơn (FP16)
```bash
python train.py --csv metadata.csv --audio_dir ./ --fp16
```

### Batch size nhỏ (nếu hết VRAM)
```bash
python train.py --csv metadata.csv --audio_dir ./ --batch_size 2
```

### Train nhiều epochs
```bash
python train.py --csv metadata.csv --audio_dir ./ --epochs 20
```

### Audio ở thư mục khác
```bash
python train.py --csv metadata.csv --audio_dir /path/to/audio
```

---

## 📊 Theo dõi Training

```bash
tensorboard --logdir ./checkpoints/vietnamese/logs
```

Mở: http://localhost:6006

---

## 🔧 Troubleshooting

**CUDA out of memory?**
```bash
python train.py --csv metadata.csv --audio_dir ./ --batch_size 2
```

**Tokenizer error?**
```bash
python train_tokenizer_from_corpus.py metadata.csv
```

**Audio loading error?**
- Kiểm tra format: `.wav`
- Kiểm tra đường dẫn trong CSV
- Kiểm tra `--audio_dir`

---

**Đọc thêm:** `README.md`

