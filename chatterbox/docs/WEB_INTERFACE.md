# 🌐 Web Interface Guide

Complete guide for using the Gradio web interface for Vietnamese TTS Voice Cloning.

## 🚀 Getting Started

### Launch Interface

```bash
cd chatterbox
python gradio_vietnamese_voice_clone.py
```

The interface will be available at:
- **Local**: http://localhost:7860
- **Network**: http://0.0.0.0:7860 (accessible from other devices)

### Interface Overview

The web interface has three main tabs:
1. **🎭 Voice Cloning**: Clone voices with reference audio
2. **🗣️ TTS thường**: Regular text-to-speech
3. **📦 Xử lý hàng loạt**: Batch processing

## 🎭 Voice Cloning Tab

### Features
- Vietnamese text input with examples
- Reference audio upload
- Advanced parameter controls
- Real-time audio generation
- Download functionality

### Step-by-Step Usage

#### 1. Enter Vietnamese Text
- Type or paste Vietnamese text in the text box
- Use proper diacritics for best results
- Keep sentences under 200 words for optimal quality

#### 2. Upload Reference Audio
- Click "🎵 Audio tham chiếu" to upload your reference voice
- **Supported formats**: WAV, MP3, FLAC
- **Optimal duration**: 3-10 seconds
- **Quality**: Clear speech, minimal background noise

#### 3. Adjust Parameters (Optional)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Cường độ cảm xúc** | 0.0-2.0 | 0.5 | Emotion intensity |
| **Temperature** | 0.1-1.0 | 0.8 | Creativity level |
| **CFG Weight** | 0.0-1.0 | 0.5 | Quality vs diversity |

#### 4. Generate Audio
- Click "🎤 Tạo giọng nói nhân bản"
- Wait for processing (usually 10-30 seconds)
- Listen to the generated audio

#### 5. Download Result
- Right-click on the audio player
- Select "Save audio as..." to download

### Example Texts

The interface provides Vietnamese example texts:
- "Xin chào, tôi là trợ lý AI tiếng Việt."
- "Hôm nay là một ngày đẹp trời."
- "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh."

Click any example to auto-fill the text box.

## 🗣️ Regular TTS Tab

### Features
- Text-to-speech without voice cloning
- Uses default Vietnamese voice
- Faster processing
- Same parameter controls

### Usage
1. Enter Vietnamese text
2. Adjust temperature and CFG weight if needed
3. Click "🎤 Tạo giọng nói"
4. Listen and download result

## 📦 Batch Processing Tab

### Features
- Process multiple texts at once
- Optional reference audio for all texts
- ZIP file download with all results
- Progress tracking

### Usage

#### 1. Prepare Text List
Enter multiple Vietnamese sentences, one per line:
```
Câu thứ nhất để tổng hợp.
Câu thứ hai với cùng giọng nói.
Câu thứ ba để test chất lượng.
```

#### 2. Upload Reference Audio (Optional)
- If provided, all texts will use the same reference voice
- If not provided, default voice will be used

#### 3. Generate Batch
- Click "🚀 Tạo tất cả"
- Wait for processing (time depends on number of texts)
- Monitor progress in the status box

#### 4. Download Results
- Download the ZIP file containing all audio files
- Files are named sequentially: `vietnamese_tts_001.wav`, etc.

### Batch Limitations
- Maximum 20 texts per batch
- Each text should be under 200 words
- Processing time: ~10-30 seconds per text

## ⚙️ Advanced Settings

### Parameter Tuning Guide

#### Exaggeration (Cường độ cảm xúc)
- **0.0**: Very neutral, monotone
- **0.5**: Natural expression (recommended)
- **1.0**: More expressive
- **2.0**: Very dramatic

#### Temperature
- **0.1**: Very consistent, less varied
- **0.5**: Balanced
- **0.8**: Natural variation (recommended)
- **1.0**: More creative, less predictable

#### CFG Weight
- **0.0**: More creative, less controlled
- **0.5**: Balanced (recommended)
- **1.0**: More faithful to training, less creative

### Quality Optimization

#### For Best Voice Similarity:
- Use high-quality reference audio
- Set CFG Weight to 0.7-0.8
- Keep temperature around 0.6-0.8

#### For Most Natural Speech:
- Use clear, natural reference audio
- Set exaggeration to 0.3-0.7
- Use temperature 0.7-0.9

#### For Consistent Results:
- Lower temperature (0.5-0.7)
- Higher CFG weight (0.6-0.8)
- Use same reference audio

## 🔧 Troubleshooting

### Common Issues

#### Interface Won't Load
```bash
# Check if port is available
netstat -an | grep 7860

# Try different port
python gradio_vietnamese_voice_clone.py --port 7861
```

#### Model Loading Errors
- Ensure all model files are downloaded
- Check `model_path_vietnamese.json` configuration
- Verify CUDA/GPU availability

#### Audio Upload Issues
- Check file format (WAV, MP3, FLAC supported)
- Ensure file size < 50MB
- Try converting to WAV format

#### Poor Audio Quality
- Use higher quality reference audio
- Check text for special characters
- Adjust parameters (lower temperature, higher CFG)

#### Slow Processing
- Reduce batch size
- Use GPU if available
- Close other applications using GPU

### Error Messages

#### "Vui lòng nhập text tiếng Việt!"
- Text box is empty
- Enter Vietnamese text before generating

#### "Vui lòng upload file audio tham chiếu!"
- No reference audio uploaded for voice cloning
- Upload audio file or use regular TTS tab

#### "Synthesis failed"
- Check model configuration
- Verify audio file format
- Try with different text or audio

## 🎯 Best Practices

### Reference Audio Tips
1. **Record in quiet environment**
2. **Speak naturally and clearly**
3. **Use consistent volume**
4. **Avoid background noise**
5. **Keep 3-10 seconds duration**

### Text Input Tips
1. **Use proper Vietnamese diacritics**
2. **Keep sentences natural length**
3. **Avoid special characters**
4. **Use proper punctuation**
5. **Test with simple sentences first**

### Performance Tips
1. **Close unnecessary browser tabs**
2. **Use Chrome or Firefox for best compatibility**
3. **Ensure stable internet connection**
4. **Allow microphone access if needed**

## 📱 Mobile Usage

### Responsive Design
- Interface adapts to mobile screens
- Touch-friendly controls
- Optimized for tablets and phones

### Mobile Limitations
- Slower processing on mobile devices
- Limited audio recording capabilities
- Smaller file upload limits

### Mobile Tips
- Use landscape mode for better layout
- Upload audio files from cloud storage
- Use shorter texts for faster processing

## 🔒 Privacy and Security

### Data Handling
- Audio files are processed locally
- No data is sent to external servers
- Temporary files are cleaned automatically

### Security Features
- Local processing only
- No user data collection
- Secure file handling

## 🚀 Production Deployment

### Server Configuration
```python
# For production deployment
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,  # Set True for public access
    auth=("username", "password"),  # Add authentication
    ssl_keyfile="path/to/key.pem",
    ssl_certfile="path/to/cert.pem"
)
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "gradio_vietnamese_voice_clone.py"]
```

### Load Balancing
- Use nginx for load balancing
- Configure multiple instances
- Implement health checks

---

**Need help?** Check our [troubleshooting guide](TROUBLESHOOTING.md) or open an issue!
