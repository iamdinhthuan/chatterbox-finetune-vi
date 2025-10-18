# Preprocessing Guide - Pre-compute Embeddings for Fast Training

## Tại sao cần preprocessing?

Training bị **CPU bottleneck** vì mỗi sample phải:
1. Tính voice embedding (CPU)
2. Tokenize speech (CPU)
3. Tokenize conditioning prompt (CPU)

→ GPU phải chờ CPU → GPU idle 50-60%

**Giải pháp**: Pre-compute tất cả 1 lần, lưu vào file, training chỉ cần load!

## Performance

| Method | GPU Util | Training Speed | Disk Space |
|--------|----------|----------------|------------|
| On-the-fly | 40-60% | 1x (baseline) | 0 |
| + Workers=8 | 80-95% | 2-3x | 0 |
| **Pre-computed** | **95-100%** | **4-5x** | ~2GB/1000 samples |

## Quick Start

### 1. Preprocessing Data

```bash
python preprocess_dataset.py \
    --data_dir ./data/vietnamese \
    --output_dir ./data/preprocessed \
    --checkpoint ./vietnamese/pretrained_model_download \
    --num_workers 4
```

**Parameters:**
- `--data_dir`: Thư mục chứa `metadata.csv` và audio files
- `--output_dir`: Thư mục output cho .pt files
- `--checkpoint`: Path đến pretrained model
- `--num_workers`: Số CPU cores dùng (mặc định 1, khuyến nghị 4-8)
- `--start_idx`: Bắt đầu từ index (để resume nếu bị gián đoạn)
- `--end_idx`: Kết thúc tại index (optional)

### 2. Structure của metadata.csv

```
audio_path|text
data/audio/001.wav|Xin chào các bạn.
data/audio/002.wav|Đây là bài test.
/absolute/path/003.wav|Có thể dùng absolute path.
```

- Delimiter: `|`
- Columns: `audio_path|text`
- Audio path: relative (to data_dir) hoặc absolute

### 3. Training với Preprocessed Data

#### Option A: Sử dụng PrecomputedDataset (Khuyến nghị)

Sửa file training script:

```python
from chatterbox.utils.preprocessed_dataset import PrecomputedDataset, collate_fn

# Replace dataset loading
train_dataset = PrecomputedDataset(
    preprocessed_dir="./data/preprocessed",
    max_text_len=512,
    max_speech_len=2048,
)

# Use with DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)
```

#### Option B: Sửa finetune_t3_thai.py

Thêm argument:

```python
@dataclass
class DataArguments:
    # ... existing args ...
    preprocessed_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory with preprocessed .pt files. If set, will use PrecomputedDataset."}
    )
```

Trong `run_training()`:

```python
if data_args.preprocessed_dir:
    from chatterbox.utils.preprocessed_dataset import PrecomputedDataset
    train_dataset = PrecomputedDataset(
        preprocessed_dir=data_args.preprocessed_dir,
        max_text_len=data_args.max_text_len,
        max_speech_len=data_args.max_speech_len,
    )
else:
    # Original dataset loading
    train_dataset = SpeechFineTuningDataset(...)
```

Training command:

```bash
python src/finetune_t3_thai.py \
    --preprocessed_dir ./data/preprocessed \
    --per_device_train_batch_size 16 \
    --dataloader_num_workers 4 \
    --fp16 \
    --output_dir ./output \
    --save_steps 1000
```

## Advanced Usage

### Resume Preprocessing

Nếu bị gián đoạn:

```bash
# Check how many processed
ls data/preprocessed/*.pt | wc -l

# Resume from index 500
python preprocess_dataset.py \
    --data_dir ./data/vietnamese \
    --output_dir ./data/preprocessed \
    --checkpoint ./vietnamese/pretrained_model_download \
    --start_idx 500 \
    --num_workers 4
```

### Preprocessing Large Dataset

Chia nhỏ ra:

```bash
# Process 0-10000
python preprocess_dataset.py \
    --start_idx 0 --end_idx 10000 \
    --num_workers 8 \
    ...

# Process 10000-20000
python preprocess_dataset.py \
    --start_idx 10000 --end_idx 20000 \
    --num_workers 8 \
    ...
```

### Parallel Processing

Sử dụng nhiều GPU/máy:

```bash
# Machine 1: Process first half
python preprocess_dataset.py \
    --end_idx 50000 \
    --output_dir ./preprocessed_part1 \
    ...

# Machine 2: Process second half
python preprocess_dataset.py \
    --start_idx 50000 \
    --output_dir ./preprocessed_part2 \
    ...

# Merge
cp ./preprocessed_part1/*.pt ./preprocessed_all/
cp ./preprocessed_part2/*.pt ./preprocessed_all/
```

## Output Structure

```
data/preprocessed/
├── 000000.pt          # Sample 0
├── 000001.pt          # Sample 1
├── ...
└── preprocessing_summary.json  # Summary và errors
```

Mỗi `.pt` file chứa:

```python
{
    'text_tokens': torch.Tensor,           # Shape: (seq_len,)
    'text_token_lens': torch.Tensor,       # Shape: ()
    'speech_tokens': torch.Tensor,         # Shape: (seq_len,)
    'speech_token_lens': torch.Tensor,     # Shape: ()
    't3_cond_speaker_emb': torch.Tensor,   # Shape: (embedding_dim,)
    't3_cond_prompt_speech_tokens': torch.Tensor,  # Shape: (150,)
    't3_cond_emotion_adv': torch.Tensor,   # Shape: ()
    'text': str,                           # Original text
    'audio_path': str,                     # Original audio path
}
```

## Disk Space

Ước tính:
- 1 sample ≈ 2-5 KB
- 1,000 samples ≈ 2-5 MB
- 10,000 samples ≈ 20-50 MB
- 100,000 samples ≈ 200-500 MB
- 1,000,000 samples ≈ 2-5 GB

## Troubleshooting

### "No .pt files found"

Kiểm tra:
1. `--output_dir` có đúng không?
2. Preprocessing có chạy thành công không?
3. Check `preprocessing_summary.json`

### "Invalid preprocessed data format"

Re-run preprocessing với checkpoint mới:

```bash
rm -rf ./data/preprocessed/*.pt
python preprocess_dataset.py ...
```

### "Out of memory" khi preprocessing

Giảm `--num_workers`:

```bash
python preprocess_dataset.py --num_workers 1 ...
```

### Audio files not found

Check paths trong metadata.csv:
- Dùng relative path: `audio/001.wav` (relative to data_dir)
- Hoặc absolute path: `/full/path/to/001.wav`

## Validation

Check preprocessing output:

```python
import torch
from pathlib import Path

# Load a sample
data = torch.load("./data/preprocessed/000000.pt")

print("Keys:", data.keys())
print("Text:", data['text'])
print("Text tokens shape:", data['text_tokens'].shape)
print("Speech tokens shape:", data['speech_tokens'].shape)
print("Speaker emb shape:", data['t3_cond_speaker_emb'].shape)
```

Check summary:

```bash
cat ./data/preprocessed/preprocessing_summary.json
```

## Performance Tips

1. **Optimize num_workers**: Sử dụng `num_workers = CPU_cores - 2`
2. **Use SSD**: Lưu preprocessed data trên SSD, không phải HDD
3. **Pin memory**: Bật `pin_memory=True` trong DataLoader
4. **Increase batch size**: GPU không phải chờ CPU nữa, tăng batch size!

## Complete Example

```bash
# 1. Preprocessing (1 lần, ~30 phút cho 10K samples)
python preprocess_dataset.py \
    --data_dir ./data/vietnamese \
    --output_dir ./data/preprocessed \
    --checkpoint ./vietnamese/pretrained_model_download \
    --num_workers 8

# 2. Training (4-5x nhanh hơn!)
python src/finetune_t3_thai.py \
    --preprocessed_dir ./data/preprocessed \
    --model_name_or_path ./vietnamese/pretrained_model_download \
    --output_dir ./output/run1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --fp16 \
    --num_train_epochs 10 \
    --save_steps 1000 \
    --logging_steps 10

# 3. Monitor GPU (should be 95-100%)
watch -n 1 nvidia-smi
```

## Checklist

- [ ] Đã có `metadata.csv` với format đúng
- [ ] Audio files có thể access được từ data_dir
- [ ] Checkpoint model hoạt động
- [ ] Có đủ disk space (~2GB/1000 samples)
- [ ] Đã set `--num_workers` phù hợp
- [ ] Preprocessing chạy thành công (check summary.json)
- [ ] Training script đã sửa để dùng PrecomputedDataset
- [ ] GPU utilization 95-100% khi training

## Next Steps

Sau khi preprocessing xong:
1. Training sẽ nhanh hơn 4-5x
2. GPU utilization 95-100%
3. Có thể tăng batch size
4. Training time giảm đáng kể!
