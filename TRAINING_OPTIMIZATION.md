# Training Optimization Guide - Fix CPU Bottleneck

## Vấn đề hiện tại

GPU bị idle vì phải chờ CPU xử lý data. Nguyên nhân:

### Bottleneck trong `__getitem__`:
```python
# Mỗi lần load 1 sample:
1. voice_encoder.embeds_from_wavs([wav_16k]) <- CHẬM (CPU)
2. speech_tokenizer.forward([wav_16k])       <- CHẬM (CPU) 
3. speech_tokenizer.forward([cond_segment])  <- CHẬM (CPU)
```

→ Mỗi batch phải chờ 3 bước xử lý CPU này hoàn thành!

## Giải pháp

### 1. NHANH - Tăng DataLoader Workers (Khuyến nghị)

Thêm vào training command:

```bash
python src/finetune_t3_thai.py \
    --model_name_or_path /path/to/checkpoint \
    --data_dir /path/to/data \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    ...
```

**Giải thích:**
- `preprocessing_num_workers 8`: Dùng 8 CPU cores để xử lý data song song
- `dataloader_pin_memory True`: Giảm thời gian copy CPU→GPU
- Tăng batch size và gradient accumulation để GPU bận hơn

**Lưu ý:** 
- Số workers = min(số CPU cores - 2, 8)
- Nếu RAM không đủ, giảm workers xuống

### 2. TỐI ƯU HƠN - Pre-compute Embeddings (Recommended)

Tạo script preprocessing một lần, lưu embeddings:

```python
# preprocess_embeddings.py
import torch
from pathlib import Path
from tqdm import tqdm
import json

def preprocess_dataset(data_dir, output_dir):
    """Pre-compute voice embeddings and speech tokens"""
    from chatterbox.tts import ChatterboxTTS
    
    # Load models
    model = ChatterboxTTS.from_local(checkpoint_dir, device='cuda')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    with open(f"{data_dir}/metadata.csv") as f:
        lines = f.readlines()
    
    for line in tqdm(lines):
        audio_path, text = line.strip().split('|')
        
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        
        # Pre-compute
        speaker_emb = model.ve.embeds_from_wavs([wav.squeeze(0).numpy()], sample_rate=16000)
        speech_tokens, lengths = model.s3gen.tokenizer.forward([wav.squeeze(0)])
        
        # Save
        audio_id = Path(audio_path).stem
        torch.save({
            'speaker_emb': torch.from_numpy(speaker_emb[0]),
            'speech_tokens': speech_tokens.squeeze(0),
            'text': text
        }, output_dir / f"{audio_id}.pt")
        
    print(f"Preprocessed {len(lines)} samples to {output_dir}")

if __name__ == "__main__":
    preprocess_dataset(
        data_dir="/path/to/data",
        output_dir="/path/to/preprocessed"
    )
```

Sau đó sửa Dataset để load từ .pt files:

```python
def __getitem__(self, idx):
    # Fast loading from pre-computed file
    item_path = self.preprocessed_dir / f"{idx}.pt"
    data = torch.load(item_path)
    
    return {
        "speaker_emb": data['speaker_emb'],
        "speech_tokens": data['speech_tokens'],
        "text_tokens": self.tokenize_text(data['text']),
        ...
    }
```

### 3. TỐI ƯU TRAINING ARGS

```python
training_args = TrainingArguments(
    # Data loading
    dataloader_num_workers=8,           # Tăng workers
    dataloader_pin_memory=True,         # Pin memory
    dataloader_prefetch_factor=2,       # Prefetch 2 batches
    
    # Batch size
    per_device_train_batch_size=8,     # Tăng nếu RAM/VRAM đủ
    gradient_accumulation_steps=4,      # Accumulate gradients
    
    # Mixed precision
    fp16=True,                          # Nhanh hơn fp32
    
    # Gradient checkpointing
    gradient_checkpointing=True,        # Tiết kiệm VRAM
    
    # Optimizer
    optim="adamw_torch_fused",          # Faster optimizer
    
    # Logging
    logging_steps=10,
    save_steps=1000,
)
```

## Monitoring

### Kiểm tra GPU utilization:

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Train
python src/finetune_t3_thai.py ...
```

**Dấu hiệu tốt:**
- GPU Utilization: 90-100%
- GPU Memory: Gần full
- GPU-Util không nhảy 0% ↔ 100%

**Dấu hiệu CPU bottleneck:**
- GPU Utilization: 40-70% dao động
- GPU-Util: 0% → 100% → 0% (lúc có lúc không)

### Profile với PyTorch Profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in train_dataloader:
        # training step
        ...
        
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## So sánh Performance

| Method | GPU Util | Training Speed | Complexity |
|--------|----------|----------------|------------|
| Default (workers=0) | 40-60% | 1x | Low |
| + Workers=8 | 80-95% | 2-3x | Low |
| + Pin Memory | 85-98% | 2.5-3.5x | Low |
| Pre-computed | 95-100% | 4-5x | Medium |

## Checklist

- [ ] Tăng `dataloader_num_workers` lên 4-8
- [ ] Bật `dataloader_pin_memory=True`
- [ ] Tăng `per_device_train_batch_size` nếu VRAM cho phép
- [ ] Sử dụng `fp16=True`
- [ ] Monitor GPU utilization với `nvidia-smi`
- [ ] Xem xét pre-compute embeddings cho dataset lớn

## Quick Fix Command

Thêm vào training script hiện tại:

```bash
# Vietnamese
python src/finetune_t3_thai.py \
    --model_name_or_path ./vietnamese/pretrained_model_download \
    --data_dir ./data \
    --output_dir ./output \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --dataloader_pin_memory \
    --save_steps 1000 \
    --logging_steps 10
```

Điều chỉnh `preprocessing_num_workers` và `batch_size` theo:
- **CPU cores**: num_workers ≈ min(CPU_cores - 2, 8)
- **RAM**: batch_size * workers * avg_sample_size < available_RAM
- **VRAM**: batch_size * model_size < GPU_memory
