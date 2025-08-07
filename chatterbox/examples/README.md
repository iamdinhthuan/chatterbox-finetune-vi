# 📚 Examples

This directory contains example scripts and sample data for Vietnamese TTS Voice Cloning.

## 🚀 Quick Start

```bash
# Run all examples
python examples/basic_usage.py

# Run specific example
python -c "from examples.basic_usage import example_vietnamese_tts; example_vietnamese_tts()"
```

## 📁 Files

### Scripts
- `basic_usage.py` - Complete examples covering all features
- `voice_cloning_demo.py` - Focused voice cloning examples
- `batch_processing.py` - Batch processing utilities

### Sample Data
- `reference_voice.wav` - Sample reference audio for voice cloning
- `sample_texts.txt` - Vietnamese text samples for testing
- `outputs/` - Generated audio files (created automatically)

## 🎯 Examples Overview

### 1. Basic TTS
```python
from chatterbox.tts import ChatterboxTTS
import torchaudio

model = ChatterboxTTS.from_pretrained()
audio = model.generate("Hello, this is a test.")
torchaudio.save("output.wav", audio, 24000)
```

### 2. Vietnamese TTS
```python
# Load Vietnamese model
model = ChatterboxTTS.from_specified(
    voice_encoder_path="path/to/ve.safetensors",
    t3_path="path/to/t3_vietnamese.safetensors",
    # ... other paths
)

audio = model.generate("Xin chào, tôi là AI tiếng Việt.")
```

### 3. Voice Cloning
```python
audio = model.generate(
    text="Đây là voice cloning example.",
    audio_prompt_path="reference_voice.wav",
    exaggeration=0.5,
    temperature=0.8
)
```

### 4. Batch Processing
```python
texts = ["Câu 1", "Câu 2", "Câu 3"]
for i, text in enumerate(texts):
    audio = model.generate(text)
    torchaudio.save(f"output_{i}.wav", audio, 24000)
```

## 🔧 Requirements

- Trained Vietnamese TTS model
- Reference audio file (for voice cloning examples)
- GPU recommended for faster processing

## 📊 Expected Outputs

After running examples, you'll find:

```
examples/outputs/
├── basic_tts.wav              # Basic TTS output
├── vietnamese_tts.wav         # Vietnamese TTS output
├── voice_cloning.wav          # Voice cloning result
├── batch_YYYYMMDD_HHMMSS/     # Batch processing results
│   ├── batch_001.wav
│   ├── batch_002.wav
│   └── batch_003.wav
└── parameter_tuning/          # Different parameter settings
    ├── conservative.wav
    ├── balanced.wav
    └── creative.wav
```

## 🎭 Voice Cloning Setup

To run voice cloning examples:

1. **Prepare reference audio**:
   ```bash
   # Place your reference audio file
   cp your_voice.wav examples/reference_voice.wav
   ```

2. **Audio requirements**:
   - Duration: 3-10 seconds
   - Format: WAV (preferred)
   - Quality: Clear speech, minimal noise
   - Content: Natural speaking

3. **Run voice cloning**:
   ```bash
   python -c "from examples.basic_usage import example_voice_cloning; example_voice_cloning()"
   ```

## 🔍 Troubleshooting

### Model Not Found
```
⚠️ Vietnamese model not found. Please train the model first.
```
**Solution**: Follow the [training guide](../docs/TRAINING.md) to train your Vietnamese model.

### Reference Audio Missing
```
⚠️ Reference audio not found. Please provide reference_voice.wav
```
**Solution**: Place a reference audio file at `examples/reference_voice.wav`.

### CUDA Out of Memory
```
❌ CUDA out of memory
```
**Solution**: Use CPU or reduce batch size:
```python
model = ChatterboxTTS.from_specified(..., device="cpu")
```

### Poor Audio Quality
**Solutions**:
- Use higher quality reference audio
- Adjust parameters (lower temperature, higher cfg_weight)
- Check text for special characters

## 🎯 Next Steps

1. **Experiment with parameters** - Try different temperature and cfg_weight values
2. **Test with your voice** - Record your own reference audio
3. **Create custom scripts** - Adapt examples for your use case
4. **Share results** - Contribute improvements back to the project

## 📞 Support

- Check [documentation](../docs/)
- Open [GitHub issue](https://github.com/yourusername/vietnamese-tts-voice-cloning/issues)
- Join our [Discord community](https://discord.gg/your-invite)

---

**Happy experimenting!** 🎵
