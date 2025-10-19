# 🎤 Vietnamese TTS Gradio App Guide

Web interface for Vietnamese text-to-speech using fine-tuned Chatterbox model.

## 🚀 Quick Start

### Installation

```bash
# Install Gradio
pip install gradio

# Or if not already installed
pip install -r requirements.txt
```

### Run the App

```bash
# Local access only (default)
python app.py

# With public share link (Gradio tunnel)
python app.py --share

# Custom host and port
python app.py --server-name 0.0.0.0 --server-port 7860 --share
```

### Access

- **Local**: http://127.0.0.1:7860
- **Network**: http://YOUR_IP:7860 (if using --server-name 0.0.0.0)
- **Public**: Gradio will generate a public URL (if using --share)

## ✨ Features

### Input Section

1. **Text Input**
   - Enter Vietnamese text to generate
   - Supports multiple sentences
   - Automatic text normalization

2. **Reference Voice** (Optional)
   - Upload audio file for voice cloning
   - Or record with microphone
   - If not provided, uses built-in voice

### Generation Parameters

1. **Temperature** (0.1-1.5, default: 0.8)
   - Low (0.5-0.7): More stable, less variation
   - High (0.8-1.0): More natural, more variation

2. **CFG Weight** (0.0-2.0, default: 0.5)
   - Controls how closely to follow text
   - 0.5-1.0 recommended

3. **Emotion Exaggeration** (0.0-1.0, default: 0.5)
   - Voice emotion intensity
   - 0.0 = neutral, 1.0 = expressive

### Advanced Settings

1. **Random Seed** (default: 42)
   - For reproducible generation
   - Set to -1 for random

2. **Minimum Tokens** (10-100, default: 30)
   - Prevents early stopping
   - Increase if audio too short

3. **Split by Sentences** (default: ON)
   - Generate each sentence separately
   - Merge with cross-fading
   - Better for long text

4. **Cross-fade Duration** (0-500ms, default: 100ms)
   - Smooth transitions between sentences

### Model Paths

1. **Checkpoint Path**
   - Default: `./checkpoints/vietnamese`
   - Your fine-tuned model

2. **Base Model Path**
   - Default: `./vietnamese/pretrained_model_download`
   - Pre-trained base model

3. **Device**
   - auto: Auto-detect (CUDA > MPS > CPU)
   - cuda: Use NVIDIA GPU
   - mps: Use Apple Silicon GPU
   - cpu: Use CPU

## 📊 Output

### Audio Player
- Listen to generated audio
- Download as WAV file

### Status Display
- Device information
- Model loading status
- Text normalization preview
- Sentence splitting details
- Generation progress
- Audio duration and sample rate
- Error messages (if any)

## 🎯 Usage Examples

### Example 1: Simple Text

```
Input: "Xin chào các bạn."
Settings: Default
Result: Single sentence audio
```

### Example 2: Long Text with Splitting

```
Input: "Xin chào các bạn. Hôm nay là một ngày đẹp trời. Chúng ta sẽ học về AI."
Settings:
  - Split by Sentences: ON
  - Cross-fade: 100ms
Result: 3 sentences merged smoothly
```

### Example 3: Voice Cloning

```
Input: "Xin chào!"
Reference Voice: Upload your voice sample (3-10s)
Settings: Default
Result: Text spoken in your voice
```

### Example 4: Reproducible Generation

```
Input: "Test reproducibility."
Settings:
  - Seed: 42
  - Temperature: 0.7
Result: Same audio every time with these settings
```

## 💡 Tips & Best Practices

### Text Input

✅ **Good:**
- Use proper punctuation: `. ? ! ;`
- Clear sentence structure
- Vietnamese language (model is trained for Vietnamese)

❌ **Avoid:**
- Very long sentences without punctuation
- Mixed languages in same sentence
- Special characters or emojis

### Parameters

| Scenario | Temperature | CFG Weight | Split Sentences |
|----------|-------------|------------|-----------------|
| Short text (1-2 sentences) | 0.7-0.8 | 0.5 | OFF |
| Long text (3+ sentences) | 0.8 | 0.5-0.7 | ON |
| Stable/consistent | 0.5-0.6 | 0.7-1.0 | ON |
| Natural/expressive | 0.9-1.0 | 0.3-0.5 | ON |

### Reference Voice

✅ **Good reference audio:**
- 3-10 seconds duration
- Clear speech, no background noise
- Single speaker
- 16kHz or higher sample rate

❌ **Poor reference audio:**
- Too short (< 1s) or too long (> 30s)
- Multiple speakers
- Heavy background noise
- Music or sound effects

## 🐛 Troubleshooting

### App won't start

```bash
# Check if Gradio is installed
pip install gradio

# Check if other dependencies are installed
pip install torch torchaudio safetensors
```

### Model not loading

```
❌ Checkpoint not found
```

**Solution:**
1. Check checkpoint path in Model Paths section
2. Verify file exists: `./checkpoints/vietnamese`
3. Update path to your trained checkpoint

### Audio too short

```
⚠️ WARNING: Audio is very short (0.3s)
```

**Solution:**
1. Increase "Minimum Tokens" to 40-50
2. Enable "Split by Sentences"
3. Add more text or punctuation

### Inconsistent quality

**Solution:**
1. Lower temperature (0.6-0.7)
2. Enable "Split by Sentences"
3. Use reference voice
4. Set fixed seed for reproducibility

### CUDA out of memory

```
❌ CUDA out of memory
```

**Solution:**
1. Switch device to "cpu"
2. Process shorter text
3. Close other GPU applications

### Generation takes too long

**Solution:**
1. Use CUDA device (if available)
2. Reduce "Minimum Tokens"
3. Disable "Split by Sentences" for short text

## 🎨 Customization

### Change Default Paths

Edit `app.py`:

```python
# Around line 425
default_checkpoint = "./your/checkpoint/path"
default_base_model = "./your/base/model/path"
```

### Change Default Parameters

Edit `app.py`:

```python
# Around line 515-525
temperature = gr.Slider(
    value=0.8,  # Change this
    ...
)
```

### Add Custom Examples

Edit `app.py`:

```python
# Around line 625
gr.Examples(
    examples=[
        ["Your custom example text here"],
        ["Another example"],
    ],
    ...
)
```

## 🔧 Advanced Configuration

### Run on Network

```bash
# Allow access from other devices on network
python app.py --server-name 0.0.0.0 --server-port 7860
```

Access from other devices: `http://YOUR_LOCAL_IP:7860`

### Public Deployment

```bash
# Generate public URL (Gradio tunnel)
python app.py --share
```

⚠️ **Warning**: Public URL expires after 72 hours. Use for demos only.

### Production Deployment

For production, consider:
- Docker containerization
- Reverse proxy (Nginx)
- Cloud deployment (AWS, GCP, Azure)
- Load balancing for multiple users

## 📝 API Access

Gradio automatically creates an API. Access at:

```
http://127.0.0.1:7860/api/
```

### Python API Example

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")

result = client.predict(
    "Xin chào các bạn.",  # text
    "./checkpoints/vietnamese",  # checkpoint_path
    "./vietnamese/pretrained_model_download",  # base_model_path
    None,  # voice_audio
    0.8,   # temperature
    0.5,   # cfg_weight
    0.5,   # exaggeration
    42,    # seed
    30,    # min_tokens
    True,  # split_by_sentences
    100,   # fade_duration
    "auto" # device
)

audio_path = result[0]
status = result[1]
```

## 🎓 FAQ

**Q: Can I use this for commercial purposes?**
A: Check your model's license. This code is provided as-is.

**Q: How do I train my own model?**
A: See main README.md for training instructions.

**Q: Can I add more languages?**
A: You'll need to train on data for that language.

**Q: How do I improve voice quality?**
A: Use better training data, longer training, reference voice.

**Q: Can multiple users use the app simultaneously?**
A: Yes, but performance depends on your hardware. Consider queuing for production.

## 📄 License

Same as the main project license.

## 🙏 Credits

- Based on Chatterbox TTS
- Gradio for the web interface
- F5-TTS for text processing inspiration

---

**Enjoy generating Vietnamese speech! 🇻🇳🎉**

For issues or questions, check the main README.md or open an issue on GitHub.
