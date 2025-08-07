# Hướng dẫn Inference Model TTS Tiếng Việt

## Tổng quan
Sau khi train xong model, bạn có thể sử dụng các script inference để tạo audio từ text tiếng Việt.

## Files cần thiết

### 1. Model đã train
```
vietnamese_model_output/
├── final_vietnamese_model.pt          # Model cuối cùng
├── checkpoint_epoch_10.pt             # Checkpoint epoch 10
├── checkpoint_epoch_9.pt              # Checkpoint epoch 9
└── ...
```

### 2. Tokenizer Vietnamese
```
tokenizer_vi_expanded.json              # Tokenizer đã merge (1997 tokens)
```

## Cách sử dụng

### Option 1: Script đơn giản (Recommended)

```bash
cd chatterbox
python inference_simple.py
```

**Features:**
- ✅ Dễ sử dụng
- ✅ Demo với test texts có sẵn
- ✅ Interactive mode
- ✅ Tự động tạo thư mục output

### Option 2: Script với model architecture đầy đủ

```bash
cd chatterbox  
python inference_with_model.py
```

**Features:**
- ✅ Sử dụng chính xác model architecture đã train
- ✅ Load checkpoint với thông tin training
- ✅ Batch generation
- ✅ Interactive mode

### Option 3: Script inference nâng cao

```bash
cd chatterbox
python inference_vietnamese_tts.py --model vietnamese_model_output/final_vietnamese_model.pt --tokenizer tokenizer_vi_expanded.json --text "Xin chào các bạn"
```

**Command line options:**
```bash
# Single text
python inference_vietnamese_tts.py \
    --model vietnamese_model_output/final_vietnamese_model.pt \
    --tokenizer tokenizer_vi_expanded.json \
    --text "Xin chào, tôi là AI tiếng Việt" \
    --output hello_vietnamese.wav

# Batch từ file
echo "Xin chào các bạn" > texts.txt
echo "Hôm nay trời đẹp" >> texts.txt
python inference_vietnamese_tts.py \
    --model vietnamese_model_output/final_vietnamese_model.pt \
    --tokenizer tokenizer_vi_expanded.json \
    --text_file texts.txt \
    --output_dir batch_output

# Interactive mode
python inference_vietnamese_tts.py \
    --model vietnamese_model_output/final_vietnamese_model.pt \
    --tokenizer tokenizer_vi_expanded.json
```

## Kết quả mong đợi

### Output files
```
inference_output/
├── vietnamese_tts_001_20241206_143022.wav
├── vietnamese_tts_002_20241206_143023.wav
├── interactive_20241206_143045.wav
└── ...
```

### Console output
```
✅ Vietnamese TTS Model loaded on cuda
📝 Vocabulary size: 1997
🎯 Processing text: 'Xin chào các bạn'
📝 Text tokens shape: torch.Size([1, 17])
🎵 Mel output shape: torch.Size([1, 68, 80])
💾 Audio saved: inference_output/vietnamese_tts_001_20241206_143022.wav
```

## Lưu ý quan trọng

### ⚠️ Vocoder Limitation
Hiện tại scripts sử dụng **placeholder vocoder** (tạo audio ngẫu nhiên) vì:
- Model chỉ tạo mel spectrogram
- Cần vocoder riêng để convert mel → audio
- Vocoder thực sự cần train riêng hoặc dùng pre-trained

### 🔧 Để có audio thực sự, cần:

1. **Sử dụng pre-trained vocoder:**
```python
# Ví dụ với HiFi-GAN
from hifigan import HiFiGAN
vocoder = HiFiGAN.from_pretrained("hifigan-universal")
audio = vocoder(mel_spectrogram)
```

2. **Hoặc train vocoder riêng:**
```bash
# Train HiFi-GAN với dữ liệu tiếng Việt
python train_hifigan.py --data vietnamese_audio_data
```

## Troubleshooting

### 1. Model không load được
```bash
# Kiểm tra file tồn tại
ls -la vietnamese_model_output/
ls -la tokenizer_vi_expanded.json

# Thử với checkpoint khác
python inference_with_model.py  # Sẽ list available models
```

### 2. CUDA out of memory
```python
# Trong script, thay đổi device
tts = VietnameseTTSInference(model_path, tokenizer_path, device='cpu')
```

### 3. Tokenizer error
```bash
# Kiểm tra tokenizer format
python -c "
import json
with open('tokenizer_vi_expanded.json', 'r') as f:
    data = json.load(f)
print('Vocab size:', len(data['model']['vocab']))
"
```

## Tối ưu hóa

### 1. Tăng tốc inference
```python
# Sử dụng torch.jit.script
model = torch.jit.script(model)

# Hoặc torch.compile (PyTorch 2.0+)
model = torch.compile(model)
```

### 2. Batch processing
```python
# Process nhiều texts cùng lúc
texts = ["Text 1", "Text 2", "Text 3"]
results = tts.batch_generate(texts)
```

### 3. Memory optimization
```python
# Clear cache sau mỗi inference
torch.cuda.empty_cache()
```

## Next Steps

### 1. Cải thiện chất lượng audio
- Train vocoder riêng cho tiếng Việt
- Fine-tune với dữ liệu chất lượng cao
- Sử dụng advanced vocoder (HiFi-GAN, WaveGlow)

### 2. Tối ưu model
- Model quantization
- ONNX export cho deployment
- TensorRT optimization

### 3. Production deployment
- API server với FastAPI
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)

## Ví dụ sử dụng trong code

```python
from inference_with_model import VietnameseTTSInference

# Khởi tạo
tts = VietnameseTTSInference(
    model_path="vietnamese_model_output/final_vietnamese_model.pt",
    tokenizer_path="tokenizer_vi_expanded.json"
)

# Tạo audio
audio = tts.text_to_speech(
    text="Xin chào, tôi là trợ lý AI tiếng Việt",
    output_path="output.wav"
)

# Batch processing
texts = ["Câu 1", "Câu 2", "Câu 3"]
results = tts.batch_generate(texts, output_dir="batch_output")
```
