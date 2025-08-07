# 🎭 Voice Cloning Tutorial

Learn how to clone voices using Vietnamese TTS with reference audio.

## 🎯 Overview

Voice cloning allows you to generate speech that mimics a specific person's voice using just a few seconds of reference audio. This guide covers:

1. Understanding voice cloning
2. Preparing reference audio
3. Using the Python API
4. Using the web interface
5. Advanced techniques
6. Best practices

## 🧠 How Voice Cloning Works

### Technical Process

1. **Voice Encoding**: Extract voice characteristics from reference audio
2. **Text Processing**: Convert Vietnamese text to tokens
3. **Conditional Generation**: Generate speech conditioned on voice features
4. **Audio Synthesis**: Convert tokens to high-quality audio

### Key Components

- **Voice Encoder**: Extracts speaker embeddings
- **T3 Model**: Generates speech tokens
- **S3Gen**: Converts tokens to audio waveform

## 🎵 Preparing Reference Audio

### Audio Requirements

| Requirement | Specification | Notes |
|-------------|---------------|-------|
| **Duration** | 3-10 seconds | 5-7 seconds optimal |
| **Format** | WAV, MP3, FLAC | WAV preferred |
| **Sample Rate** | 16kHz+ | 24kHz recommended |
| **Channels** | Mono | Stereo will be converted |
| **Quality** | Clear speech | Minimal background noise |
| **Content** | Natural speech | Avoid singing or shouting |

### Audio Quality Tips

#### ✅ Good Reference Audio
- Clear, natural speech
- Consistent volume
- Minimal background noise
- Single speaker
- Normal speaking pace
- Good microphone quality

#### ❌ Poor Reference Audio
- Multiple speakers
- Background music/noise
- Very quiet or loud
- Distorted/compressed
- Non-speech sounds
- Emotional extremes

### Preprocessing Reference Audio

```python
import librosa
import soundfile as sf

def preprocess_reference_audio(input_path, output_path):
    """Preprocess reference audio for optimal voice cloning"""
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=24000, mono=True)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize volume
    audio = librosa.util.normalize(audio)
    
    # Ensure optimal length (5-7 seconds)
    target_length = int(6 * sr)  # 6 seconds
    if len(audio) > target_length:
        # Take middle portion
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    
    # Save processed audio
    sf.write(output_path, audio, sr)
    print(f"✅ Processed audio saved: {output_path}")

# Usage
preprocess_reference_audio("raw_voice.wav", "reference_voice.wav")
```

## 🐍 Python API Usage

### Basic Voice Cloning

```python
from chatterbox.tts import ChatterboxTTS
import torchaudio

# Load Vietnamese TTS model
model = ChatterboxTTS.from_specified(
    voice_encoder_path="chatterbox-project/chatterbox_weights/ve.safetensors",
    t3_path="model.safetensors",
    s3gen_path="chatterbox-project/chatterbox_weights/s3gen.safetensors",
    tokenizer_path="tokenizer_vi_merged.json",
    conds_path="chatterbox-project/chatterbox_weights/conds.pt",
    device="cuda"
)

# Generate speech with voice cloning
text = "Xin chào, tôi là trợ lý AI tiếng Việt với giọng nói được nhân bản."
audio = model.generate(
    text=text,
    audio_prompt_path="reference_voice.wav"
)

# Save result
torchaudio.save("cloned_voice.wav", audio, 24000)
```

### Advanced Parameters

```python
# Fine-tune voice cloning with parameters
audio = model.generate(
    text=text,
    audio_prompt_path="reference_voice.wav",
    exaggeration=0.5,      # Emotion intensity (0.0-2.0)
    temperature=0.8,       # Creativity (0.1-1.0)
    cfg_weight=0.5,        # Quality vs diversity (0.0-1.0)
    redact=False           # Remove watermark (for research)
)
```

### Parameter Guide

| Parameter | Range | Description | Effect |
|-----------|-------|-------------|--------|
| `exaggeration` | 0.0-2.0 | Emotion intensity | Higher = more expressive |
| `temperature` | 0.1-1.0 | Sampling randomness | Higher = more varied |
| `cfg_weight` | 0.0-1.0 | Guidance strength | Higher = more faithful |

### Batch Voice Cloning

```python
def batch_voice_clone(texts, reference_audio, output_dir):
    """Clone voice for multiple texts"""
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, text in enumerate(texts, 1):
        audio = model.generate(
            text=text,
            audio_prompt_path=reference_audio
        )
        
        output_path = os.path.join(output_dir, f"clone_{i:03d}_{timestamp}.wav")
        torchaudio.save(output_path, audio, 24000)
        print(f"✅ Generated: {output_path}")

# Usage
texts = [
    "Câu thứ nhất với giọng nói được nhân bản.",
    "Câu thứ hai cũng với cùng giọng nói.",
    "Câu thứ ba để test chất lượng voice cloning."
]

batch_voice_clone(texts, "reference_voice.wav", "cloned_outputs")
```

## 🌐 Web Interface Usage

### 1. Launch Interface

```bash
cd chatterbox
python gradio_vietnamese_voice_clone.py
```

### 2. Voice Cloning Tab

1. **Enter Text**: Type Vietnamese text in the text box
2. **Upload Reference**: Click "Audio tham chiếu" to upload your reference audio
3. **Adjust Parameters** (optional):
   - **Exaggeration**: Control emotion intensity
   - **Temperature**: Control creativity
   - **CFG Weight**: Control quality
4. **Generate**: Click "Tạo giọng nói nhân bản"
5. **Listen**: Play the generated audio
6. **Download**: Right-click the audio player to save

### 3. Batch Processing Tab

1. **Enter Multiple Texts**: One sentence per line
2. **Upload Reference**: Optional reference audio for all texts
3. **Generate**: Click "Tạo tất cả"
4. **Download ZIP**: Get all generated audio files

## 🔬 Advanced Techniques

### Voice Mixing

```python
# Mix characteristics from multiple reference voices
def mix_voices(text, voice_paths, weights):
    """Mix multiple voice characteristics"""
    
    # Generate embeddings for each voice
    embeddings = []
    for voice_path in voice_paths:
        # Extract voice embedding
        embedding = model.ve.embeds_from_wavs([voice_path])
        embeddings.append(embedding)
    
    # Weighted average of embeddings
    mixed_embedding = sum(w * emb for w, emb in zip(weights, embeddings))
    
    # Generate with mixed voice
    # (This requires custom implementation)
    return mixed_embedding

# Usage
mixed_audio = mix_voices(
    text="Giọng nói kết hợp từ nhiều người",
    voice_paths=["voice1.wav", "voice2.wav"],
    weights=[0.7, 0.3]
)
```

### Voice Conversion

```python
# Convert existing speech to different voice
def voice_conversion(source_audio, target_voice):
    """Convert speech from one voice to another"""
    
    # This requires the voice conversion module
    from chatterbox.vc import VoiceConverter
    
    vc = VoiceConverter.from_specified(
        device="cuda",
        s3gen_path="chatterbox-project/chatterbox_weights/s3gen.safetensors",
        conds_path="chatterbox-project/chatterbox_weights/conds.pt"
    )
    
    # Convert voice
    converted_audio = vc.generate(
        audio=source_audio,
        target_voice_path=target_voice
    )
    
    return converted_audio
```

## 🎯 Best Practices

### For High-Quality Results

1. **Reference Audio Quality**:
   - Use studio-quality recordings when possible
   - Ensure consistent audio levels
   - Remove background noise

2. **Text Considerations**:
   - Use natural Vietnamese sentences
   - Include proper diacritics
   - Avoid very long sentences (>50 words)

3. **Parameter Tuning**:
   - Start with default parameters
   - Adjust exaggeration for emotion
   - Lower temperature for consistency

### Common Issues and Solutions

#### Issue: Poor Voice Similarity
**Solutions**:
- Use higher quality reference audio
- Increase reference audio duration (up to 10 seconds)
- Try different segments of the reference audio
- Adjust cfg_weight parameter

#### Issue: Unnatural Speech
**Solutions**:
- Lower temperature parameter
- Use more natural reference audio
- Check text for errors or unusual characters
- Reduce exaggeration parameter

#### Issue: Inconsistent Results
**Solutions**:
- Use consistent reference audio quality
- Set fixed random seed for reproducibility
- Use longer reference audio
- Increase cfg_weight for more guidance

### Quality Assessment

```python
def assess_voice_similarity(original_voice, cloned_voice):
    """Assess similarity between original and cloned voice"""
    
    # Extract embeddings
    original_emb = model.ve.embeds_from_wavs([original_voice])
    cloned_emb = model.ve.embeds_from_wavs([cloned_voice])
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(original_emb, cloned_emb)[0][0]
    
    print(f"Voice similarity: {similarity:.3f}")
    return similarity

# Usage
similarity = assess_voice_similarity("reference.wav", "cloned.wav")
```

## 🚀 Production Tips

### Optimization for Speed

```python
# Pre-load model for faster inference
class VoiceCloningService:
    def __init__(self):
        self.model = self.load_model()
    
    def clone_voice(self, text, reference_audio):
        # Fast inference with pre-loaded model
        return self.model.generate(
            text=text,
            audio_prompt_path=reference_audio,
            temperature=0.7  # Slightly lower for consistency
        )
```

### Caching Reference Voices

```python
# Cache voice embeddings for repeated use
voice_cache = {}

def get_voice_embedding(voice_path):
    if voice_path not in voice_cache:
        voice_cache[voice_path] = model.ve.embeds_from_wavs([voice_path])
    return voice_cache[voice_path]
```

## 📊 Evaluation Metrics

### Objective Metrics
- **Voice Similarity**: Cosine similarity of voice embeddings
- **Speech Quality**: MOS (Mean Opinion Score)
- **Intelligibility**: Word Error Rate (WER)

### Subjective Evaluation
- **Naturalness**: How natural does the speech sound?
- **Similarity**: How similar to the reference voice?
- **Quality**: Overall audio quality

---

**Ready to clone voices?** Start with the [Quick Start Guide](QUICK_START.md) or try the [web interface](WEB_INTERFACE.md)!
