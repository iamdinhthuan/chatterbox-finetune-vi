# 🏃 Quick Start Guide

Get up and running with Vietnamese TTS Voice Cloning in minutes!

## 📋 Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM
- 50GB+ free disk space

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/vietnamese-tts-voice-cloning.git
cd vietnamese-tts-voice-cloning
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install ChatterboxTTS
pip install chatterbox-tts
```

## 📥 Download Models

### Option 1: Automatic Download

```bash
cd chatterbox
python download_hf_repo.py
```

### Option 2: Manual Download

```bash
# Download base models from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ResembleAI/chatterbox',
    local_dir='chatterbox/chatterbox-project/chatterbox_weights',
    local_dir_use_symlinks=False
)
"
```

## 🎯 First Inference

### 1. Test Basic TTS

```python
# test_basic.py
from chatterbox.tts import ChatterboxTTS
import torchaudio

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Generate speech
text = "Hello, this is a test."
audio = model.generate(text=text)

# Save audio
torchaudio.save("test_basic.wav", audio, 24000)
print("✅ Basic TTS test completed!")
```

### 2. Test Vietnamese TTS (if you have trained model)

```python
# test_vietnamese.py
import json
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
import torchaudio

# Load Vietnamese model config
with open("chatterbox/model_path_vietnamese.json", 'r') as f:
    config = json.load(f)

# Convert paths
current_dir = Path.cwd() / "chatterbox"
model = ChatterboxTTS.from_specified(
    voice_encoder_path=current_dir / config["voice_encoder_path"],
    t3_path=current_dir / config["t3_path"],
    s3gen_path=current_dir / config["s3gen_path"],
    tokenizer_path=Path(config["tokenizer_path"]),
    conds_path=current_dir / config["conds_path"],
    device="cuda"
)

# Test Vietnamese text
text = "Xin chào, tôi là trợ lý AI tiếng Việt."
audio = model.generate(text=text)

# Save audio
torchaudio.save("test_vietnamese.wav", audio, 24000)
print("✅ Vietnamese TTS test completed!")
```

## 🎭 Test Voice Cloning

```python
# test_voice_cloning.py
# (Assuming you have Vietnamese model and reference audio)

text = "Đây là test voice cloning với tiếng Việt."
audio = model.generate(
    text=text,
    audio_prompt_path="path/to/reference_voice.wav"
)

torchaudio.save("test_voice_clone.wav", audio, 24000)
print("✅ Voice cloning test completed!")
```

## 🌐 Launch Web Interface

```bash
cd chatterbox
python gradio_vietnamese_voice_clone.py
```

Open http://localhost:7860 in your browser.

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

#### 2. Model Not Found
```bash
# Check if models are downloaded
ls chatterbox/chatterbox-project/chatterbox_weights/
# Should see: ve.safetensors, t3_cfg.safetensors, s3gen.safetensors, etc.
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip uninstall chatterbox-tts
pip install chatterbox-tts
```

#### 4. Audio Quality Issues
- Use high-quality reference audio (16kHz+, mono)
- Keep reference audio 3-10 seconds long
- Ensure clear speech without background noise

### Getting Help

- Check [GitHub Issues](https://github.com/yourusername/vietnamese-tts-voice-cloning/issues)
- Join our [Discord community](https://discord.gg/your-invite)
- Read the [FAQ](FAQ.md)

## ✅ Verification Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] Base models downloaded
- [ ] Basic TTS test works
- [ ] Web interface launches
- [ ] (Optional) Vietnamese model works
- [ ] (Optional) Voice cloning works

## 🎯 Next Steps

1. **Training**: Follow [Training Guide](TRAINING.md) to train on your data
2. **Voice Cloning**: Read [Voice Cloning Tutorial](VOICE_CLONING.md)
3. **Production**: Check [Deployment Guide](DEPLOYMENT.md)
4. **Advanced**: Explore [Research & Development](RESEARCH.md)

---

**Need help?** Open an issue or join our community!
