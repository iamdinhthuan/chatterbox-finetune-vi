# Server Setup - Vietnamese TTS Training

Hướng dẫn setup và train trên server với dataset lớn.

## 📋 Yêu cầu Server

- **GPU**: NVIDIA GPU với CUDA (khuyến nghị: RTX 3090, A100, V100)
- **RAM**: 32GB+ (cho dataset lớn)
- **Storage**: 100GB+ (cho dataset + checkpoints)
- **OS**: Ubuntu 20.04+ / Linux

## 🔧 Setup Server

### 1. Cài đặt CUDA & cuDNN

```bash
# Kiểm tra CUDA
nvidia-smi

# Nếu chưa có, cài CUDA 11.8+
# https://developer.nvidia.com/cuda-downloads
```

### 2. Cài đặt Python & Dependencies

```bash
# Python 3.8+
python3 --version

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Upload Dataset

**Option 1: SCP**
```bash
# Từ local machine
scp metadata.csv user@server:/path/to/project/
scp -r audio/ user@server:/path/to/project/audio/
```

**Option 2: rsync (nhanh hơn)**
```bash
rsync -avz --progress metadata.csv user@server:/path/to/project/
rsync -avz --progress audio/ user@server:/path/to/project/audio/
```

**Option 3: Wget/Curl (nếu có URL)**
```bash
wget https://your-storage.com/dataset.zip
unzip dataset.zip
```

## 🚀 Training trên Server

### 1. Tạo Tokenizer

```bash
python create_vietnamese_tokenizer.py
```

### 2. Train với Screen/Tmux (để chạy background)

**Dùng Screen:**
```bash
# Tạo session
screen -S tts_training

# Chạy training
python train.py \
  --csv metadata.csv \
  --audio_dir ./audio \
  --output_dir ./checkpoints/vietnamese_v1 \
  --batch_size 8 \
  --epochs 20 \
  --fp16 \
  --save_steps 1000

# Detach: Ctrl+A, D
# Reattach: screen -r tts_training
```

**Dùng Tmux:**
```bash
# Tạo session
tmux new -s tts_training

# Chạy training
python train.py --csv metadata.csv --audio_dir ./audio --fp16

# Detach: Ctrl+B, D
# Reattach: tmux attach -t tts_training
```

**Dùng nohup:**
```bash
nohup python train.py \
  --csv metadata.csv \
  --audio_dir ./audio \
  --fp16 \
  > training.log 2>&1 &

# Xem log
tail -f training.log
```

### 3. Theo dõi Training

**TensorBoard:**
```bash
# Trên server
tensorboard --logdir ./checkpoints/vietnamese_v1/logs --host 0.0.0.0 --port 6006

# Từ local machine, tạo SSH tunnel
ssh -L 6006:localhost:6006 user@server

# Mở browser: http://localhost:6006
```

**Xem GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Xem log:**
```bash
tail -f training.log
```

## 📊 Tối ưu cho Dataset Lớn

### Dataset > 100k samples

```bash
python train.py \
  --csv metadata.csv \
  --audio_dir ./audio \
  --batch_size 16 \
  --epochs 10 \
  --fp16 \
  --save_steps 2000 \
  --eval_steps 2000 \
  --gradient_accumulation_steps 2
```

### Multi-GPU Training

```bash
# Sử dụng DataParallel (tự động)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --csv metadata.csv \
  --audio_dir ./audio \
  --batch_size 32 \
  --fp16
```

### Tối ưu VRAM

```bash
# Batch size nhỏ + gradient accumulation
python train.py \
  --csv metadata.csv \
  --audio_dir ./audio \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --fp16
```

## 💾 Backup & Download Model

### Backup checkpoints

```bash
# Nén checkpoints
tar -czf vietnamese_v1.tar.gz checkpoints/vietnamese_v1/

# Download về local
scp user@server:/path/to/vietnamese_v1.tar.gz ./
```

### Rsync (incremental backup)

```bash
# Từ local machine
rsync -avz --progress user@server:/path/to/checkpoints/ ./checkpoints/
```

## 🔍 Monitoring

### System monitoring

```bash
# CPU, RAM, Disk
htop

# GPU
nvidia-smi -l 1

# Disk usage
df -h
du -sh checkpoints/
```

### Training progress

```bash
# Xem log
tail -f training.log

# Grep loss
grep "loss" training.log | tail -20

# Count checkpoints
ls -l checkpoints/vietnamese_v1/checkpoint-*/
```

## 🐛 Troubleshooting

### CUDA out of memory

```bash
# Giảm batch size
python train.py --csv metadata.csv --audio_dir ./ --batch_size 2 --fp16
```

### Disk full

```bash
# Xóa checkpoints cũ (giữ lại mới nhất)
cd checkpoints/vietnamese_v1/
ls -t checkpoint-* | tail -n +6 | xargs rm -rf
```

### Process killed (OOM)

```bash
# Kiểm tra RAM
free -h

# Giảm batch size hoặc tăng RAM
```

### SSH timeout

```bash
# Dùng screen/tmux thay vì SSH trực tiếp
screen -S tts_training
python train.py ...
# Ctrl+A, D để detach
```

## 📝 Example Workflow

```bash
# 1. SSH vào server
ssh user@server

# 2. Activate venv
cd /path/to/project
source venv/bin/activate

# 3. Tạo screen session
screen -S tts_training

# 4. Tạo tokenizer (1 lần duy nhất)
python create_vietnamese_tokenizer.py

# 5. Train
python train.py \
  --csv metadata.csv \
  --audio_dir /data/audio \
  --output_dir ./checkpoints/vietnamese_v1 \
  --batch_size 8 \
  --epochs 20 \
  --fp16 \
  --save_steps 1000 \
  > training.log 2>&1

# 6. Detach screen
# Ctrl+A, D

# 7. Logout
exit

# 8. Reattach sau (từ local)
ssh user@server
screen -r tts_training

# 9. Download model sau khi train xong
scp -r user@server:/path/to/checkpoints/vietnamese_v1 ./
```

## 🎯 Best Practices

1. **Luôn dùng screen/tmux** để tránh mất session
2. **Enable FP16** để train nhanh hơn
3. **Backup checkpoints** định kỳ
4. **Monitor GPU** để tối ưu batch size
5. **Save steps hợp lý** (1000-2000 cho dataset lớn)
6. **Test model** sau mỗi vài checkpoints

---

**Chúc bạn training thành công trên server! 🚀**

