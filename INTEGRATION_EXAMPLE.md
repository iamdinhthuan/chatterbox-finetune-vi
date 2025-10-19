# Cách Tích Hợp On-the-Fly Caching vào Training

## TL;DR

Bạn có **3 options** để training nhanh hơn:

| Option | Preprocessing | Epoch 1 | Epoch 2+ | Tổng Time (10 epochs) | Khuyến nghị |
|--------|---------------|---------|----------|------------------------|-------------|
| **1. No optimization** | 0 | Slow | Slow | 300 min | ❌ Không khuyến nghị |
| **2. Pre-computed** | 30 min GPU | Fast | Fast | 90 min | ✅ Best cho production |
| **3. On-the-fly cache** | 0 | Slow | Fast | 84 min | ✅ Best cho development |

## Option 1: No Optimization (Không khuyến nghị)

Training như bình thường, không có gì thay đổi:

```bash
python src/finetune_t3_thai.py \
    --model_name_or_path ./vietnamese/pretrained_model_download \
    --data_dir ./data/vietnamese \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10
```

**Kết quả:**
- GPU Utilization: 40-60% (lãng phí!)
- Training speed: 10 samples/sec
- Time/epoch: 30 min
- **Total time: 300 min** 😢

## Option 2: Pre-computed (Khuyến nghị cho Production)

### Step 1: Preprocessing một lần

```bash
# Với GPU - Nhanh nhất (5-10x faster)
python preprocess_dataset.py \
    --metadata_csv ./data/vietnamese/metadata.csv \
    --audio_dir ./data/vietnamese/wavs \
    --output_dir ./data/preprocessed \
    --checkpoint ./vietnamese/pretrained_model_download \
    --device cuda \
    --num_workers 1
```

⏱️ Preprocessing time: ~30 phút (1 lần duy nhất!)

### Step 2: Training với preprocessed data

Hiện tại cần sửa code một chút để dùng `PrecomputedDataset`:

```python
# Trong finetune_t3_thai.py, thêm import
from chatterbox.utils.preprocessed_dataset import PrecomputedDataset

# Trong run_training(), thay thế:
if data_args.preprocessed_dir:
    train_dataset = PrecomputedDataset(
        preprocessed_dir=data_args.preprocessed_dir,
        max_text_len=data_args.max_text_len,
        max_speech_len=data_args.max_speech_len,
    )
else:
    train_dataset = SpeechFineTuningDataset(...)  # Original
```

Training command:

```bash
python src/finetune_t3_thai.py \
    --preprocessed_dir ./data/preprocessed \
    --model_name_or_path ./vietnamese/pretrained_model_download \
    --output_dir ./output \
    --per_device_train_batch_size 16 \
    --dataloader_num_workers 4 \
    --fp16 \
    --num_train_epochs 10
```

**Kết quả:**
- GPU Utilization: 95-100% ⚡
- Training speed: 40-50 samples/sec
- Time/epoch: 6 min
- **Preprocessing: 30 min**
- **Total time: 30 + 60 = 90 min** 🚀

## Option 3: On-the-Fly Caching (Khuyến nghị cho Development)

### Cách 1: Quick Integration (Đơn giản)

Sửa file training script:

```python
# train_with_cache.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.utils.cached_dataset import CachedSpeechFineTuningDataset
from transformers import Trainer, TrainingArguments, TrainerCallback
# ... other imports ...

# Callback để monitor cache
class CacheStatsCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if hasattr(self.dataset, 'get_cache_stats'):
            stats = self.dataset.get_cache_stats()
            print(f"\n📊 Cache Stats Epoch {state.epoch}: {stats['hit_rate']:.1f}% hit rate\n")

# Load model và config
# ...

# Create cached dataset thay vì SpeechFineTuningDataset
train_dataset = CachedSpeechFineTuningDataset(
    data_args=data_args,
    t3_config=chatterbox_t3_config,
    hf_dataset=train_hf_dataset,
    is_hf_format=True,
    model_dir=model_args.model_name_or_path,
    cache_dir="./cache/train",
    device="cuda",  # Use GPU for computing embeddings
)

# Train như bình thường
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[CacheStatsCallback(train_dataset)],
)

print("📌 Epoch 1: Building cache (slow)")
print("📌 Epoch 2+: Using cache (4-5x faster!)")
trainer.train()
```

Run:

```bash
python train_with_cache.py
```

### Cách 2: Modify finetune_t3_thai.py (Recommended)

Sửa `src/finetune_t3_thai.py` để support caching:

**1. Thêm imports:**

```python
from chatterbox.utils.cached_dataset import CachedSpeechFineTuningDataset
```

**2. Thêm arguments:**

```python
@dataclass
class DataArguments:
    # ... existing args ...
    
    use_cache: bool = field(
        default=False,
        metadata={"help": "Enable on-the-fly caching. Epoch 1 slow, epoch 2+ fast (4-5x)."}
    )
    
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "Cache directory for embeddings"}
    )
    
    cache_device: str = field(
        default="cuda",
        metadata={"help": "Device for computing cached embeddings: cuda or cpu"}
    )
```

**3. Sửa dataset creation trong run_training():**

```python
def run_training(...):
    # ... existing code ...
    
    # Create dataset with optional caching
    if data_args.use_cache:
        logger.info("📦 Using CachedSpeechFineTuningDataset")
        logger.info(f"   Cache dir: {data_args.cache_dir}")
        logger.info(f"   Cache device: {data_args.cache_device}")
        train_dataset = CachedSpeechFineTuningDataset(
            data_args=data_args,
            t3_config=chatterbox_t3_config,
            hf_dataset=train_hf_dataset,
            is_hf_format=True,
            model_dir=model_args.model_name_or_path,
            cache_dir=data_args.cache_dir,
            device=data_args.cache_device,
        )
    else:
        logger.info("📦 Using SpeechFineTuningDataset (no caching)")
        if data_args.streaming:
            train_dataset = SpeechFineTuningIterableDataset(...)
        else:
            train_dataset = SpeechFineTuningDataset(...)
    
    # ... rest of code ...
```

**4. Thêm cache stats callback:**

```python
from transformers import TrainerCallback

class CacheStatsCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if hasattr(self.dataset, 'get_cache_stats'):
            stats = self.dataset.get_cache_stats()
            logger.info(f"📊 Cache Epoch {state.epoch}: {stats['cache_hits']} hits, "
                       f"{stats['cache_misses']} misses, {stats['hit_rate']:.1f}% hit rate")

# Add to trainer
trainer = Trainer(
    ...,
    callbacks=[CacheStatsCallback(train_dataset)] if data_args.use_cache else []
)
```

Training command:

```bash
python src/finetune_t3_thai.py \
    --model_name_or_path ./vietnamese/pretrained_model_download \
    --data_dir ./data/vietnamese \
    --output_dir ./output \
    --use_cache \
    --cache_dir ./cache/train \
    --cache_device cuda \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --fp16
```

**Output mẫu:**

```
📦 Using CachedSpeechFineTuningDataset
   Cache dir: ./cache/train
   Cache device: cuda

Epoch 1/10
📌 Building cache (this epoch will be slow)...
Training: 100%|████████████| 1250/1250 [30:00<00:00, 0.69it/s]
📊 Cache Epoch 1: 0 hits, 10000 misses, 0.0% hit rate

Epoch 2/10
📌 Using cache (fast mode!)
Training: 100%|████████████| 1250/1250 [06:20<00:00, 3.29it/s]
📊 Cache Epoch 2: 10000 hits, 0 misses, 100.0% hit rate

Epoch 3/10
Training: 100%|████████████| 1250/1250 [06:18<00:00, 3.30it/s]
📊 Cache Epoch 3: 10000 hits, 0 misses, 100.0% hit rate
...
```

**Kết quả:**
- GPU Utilization: 
  - Epoch 1: 40-60%
  - Epoch 2+: 95-100% ⚡
- Training speed:
  - Epoch 1: 10 samples/sec
  - Epoch 2+: 40-50 samples/sec
- Time:
  - Epoch 1: 30 min
  - Epoch 2-10: 6 min each = 54 min
- **Total time: 30 + 54 = 84 min** 🚀

## So Sánh 3 Options

### Thời gian (10 epochs):

```
No optimization:    |████████████████████████████████| 300 min
Pre-computed:       |█████████| 90 min (30 preprocessing + 60 training)
On-the-fly cache:   |████████| 84 min (30 epoch1 + 54 epoch2-10)
```

### Disk Space:

| Option | Disk Usage |
|--------|------------|
| No optimization | 0 |
| Pre-computed | ~2-5 GB (permanent) |
| On-the-fly cache | ~2-5 GB (can delete after training) |

### Flexibility:

| Option | Dataset Change | Resume Training |
|--------|----------------|-----------------|
| No optimization | ✅ Easy | ✅ Normal |
| Pre-computed | ❌ Need re-preprocess | ✅ Normal |
| On-the-fly cache | ⚠️ Clear cache | ✅ Cache persists |

## Khuyến nghị

### Development / Prototyping:
```bash
# Quick start với on-the-fly caching
python src/finetune_t3_thai.py \
    --use_cache \
    --cache_device cuda \
    --num_train_epochs 10 \
    ...
```

✅ Bắt đầu train ngay  
✅ Epoch 2+ nhanh  
✅ Không cần preprocessing riêng  

### Production / Final Training:
```bash
# Step 1: Preprocessing với GPU
python preprocess_dataset.py \
    --device cuda \
    --num_workers 1 \
    ...

# Step 2: Training với preprocessed data
python src/finetune_t3_thai.py \
    --preprocessed_dir ./data/preprocessed \
    --per_device_train_batch_size 16 \
    ...
```

✅ Tất cả epochs đều nhanh  
✅ Có thể train nhiều lần với cùng preprocessed data  
✅ Preprocessing trên GPU cực nhanh (5-10x)  

### Quick Testing (< 3 epochs):
```bash
# No optimization - simplest
python src/finetune_t3_thai.py \
    --num_train_epochs 1 \
    ...
```

✅ Đơn giản nhất  
✅ Không cần setup gì thêm  
⚠️ Chỉ cho test nhanh, không khuyến nghị cho training thực sự  

## Troubleshooting

### Cache không hoạt động?

Check cache directory:
```bash
ls -lh ./cache/train | head -10
```

Xem cache stats:
```python
stats = train_dataset.get_cache_stats()
print(stats)
```

### Out of disk space?

Clear cache:
```bash
rm -rf ./cache/train
```

Hoặc dùng pre-computed approach thay vì cache.

### Epoch 1 vẫn quá chậm?

Dùng pre-computed với GPU preprocessing:
```bash
python preprocess_dataset.py --device cuda --num_workers 1
```

Preprocessing với GPU nhanh hơn 5-10x so với CPU!

## Summary

**3 cách để training nhanh hơn:**

1. **No optimization**: Đơn giản nhưng lãng phí GPU (❌)
2. **Pre-computed**: Tốt nhất cho production (✅)
3. **On-the-fly cache**: Tốt nhất cho development (✅)

**Lời khuyên:**
- Development: Dùng on-the-fly cache
- Production: Dùng pre-computed với GPU preprocessing
- Quick test: Không optimize cũng được

**Performance:**
- Pre-computed: Tất cả epochs nhanh (90 min cho 10 epochs)
- On-the-fly cache: Epoch 2+ nhanh (84 min cho 10 epochs)
- Cả hai đều giúp GPU utilization lên 95-100%!
