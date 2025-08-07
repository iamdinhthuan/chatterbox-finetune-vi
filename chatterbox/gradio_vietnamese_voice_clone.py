#!/usr/bin/env python3
"""
Vietnamese TTS Voice Cloning - Gradio Interface
Complete Gradio interface for Vietnamese TTS with voice cloning capabilities.
"""

import os
import json
import torch
import gradio as gr
import logging
import tempfile
import torchaudio
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_vietnamese_model():
    """Load Vietnamese TTS model"""
    global model
    
    if model is not None:
        return model
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Load config
        config_path = "model_path_vietnamese.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use trained model if available
        if os.path.exists("model.safetensors"):
            config["t3_path"] = "model.safetensors"
        
        # Convert to Path objects
        current_dir = Path.cwd()
        voice_encoder_path = current_dir / config["voice_encoder_path"]
        t3_path = current_dir / config["t3_path"]
        s3gen_path = current_dir / config["s3gen_path"]
        tokenizer_path = Path(config["tokenizer_path"])
        conds_path = current_dir / config["conds_path"] if config.get("conds_path") else None
        
        # Load model
        model = ChatterboxTTS.from_specified(
            voice_encoder_path=voice_encoder_path,
            t3_path=t3_path,
            s3gen_path=s3gen_path,
            tokenizer_path=tokenizer_path,
            conds_path=conds_path,
            device=device
        )
        
        logger.info("✅ Vietnamese TTS model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise gr.Error(f"Failed to load Vietnamese TTS model: {e}")

def synthesize_with_voice_clone(text, reference_audio, exaggeration, temperature, cfg_weight):
    """
    Synthesize Vietnamese text with voice cloning
    
    Args:
        text: Vietnamese text to synthesize
        reference_audio: Reference audio file for voice cloning
        exaggeration: Emotion exaggeration (0.0-2.0)
        temperature: Sampling temperature (0.1-1.0)
        cfg_weight: CFG weight (0.0-1.0)
    
    Returns:
        tuple: (sample_rate, audio_array) for Gradio Audio component
    """
    try:
        # Load model if not loaded
        tts_model = load_vietnamese_model()
        
        # Validate inputs
        if not text.strip():
            raise gr.Error("Vui lòng nhập text tiếng Việt!")
        
        if reference_audio is None:
            raise gr.Error("Vui lòng upload file audio tham chiếu!")
        
        # Generate audio with voice cloning
        logger.info(f"🎤 Synthesizing: '{text[:50]}...'")
        logger.info(f"🎵 Using reference audio: {reference_audio}")
        
        audio = tts_model.generate(
            text=text,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight
        )
        
        # Convert to numpy for Gradio
        if torch.is_tensor(audio):
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            audio_np = audio
        
        logger.info("✅ Synthesis completed!")
        return (24000, audio_np)  # 24kHz sample rate
        
    except Exception as e:
        logger.error(f"❌ Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {e}")

def synthesize_default_voice(text, exaggeration, temperature, cfg_weight):
    """
    Synthesize Vietnamese text with default voice (no cloning)
    """
    try:
        # Load model if not loaded
        tts_model = load_vietnamese_model()
        
        # Validate inputs
        if not text.strip():
            raise gr.Error("Vui lòng nhập text tiếng Việt!")
        
        # Generate audio with default voice
        logger.info(f"🎤 Synthesizing with default voice: '{text[:50]}...'")
        
        audio = tts_model.generate(
            text=text,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight
        )
        
        # Convert to numpy for Gradio
        if torch.is_tensor(audio):
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            audio_np = audio
        
        logger.info("✅ Synthesis completed!")
        return (24000, audio_np)
        
    except Exception as e:
        logger.error(f"❌ Synthesis failed: {e}")
        raise gr.Error(f"Synthesis failed: {e}")

def get_example_texts():
    """Get example Vietnamese texts"""
    return [
        "Xin chào, tôi là trợ lý AI tiếng Việt.",
        "Hôm nay là một ngày đẹp trời.",
        "Công nghệ trí tuệ nhân tạo đang phát triển rất nhanh.",
        "Việt Nam là một đất nước xinh đẹp với văn hóa phong phú.",
        "Cảm ơn bạn đã sử dụng hệ thống TTS tiếng Việt.",
        "Chúc bạn có một ngày tốt lành và nhiều niềm vui!",
        "Tôi có thể nói tiếng Việt rất tự nhiên và trôi chảy.",
        "Hãy thử nghiệm với nhiều câu khác nhau để test chất lượng."
    ]

def process_batch_texts(batch_texts, reference_audio, progress=gr.Progress()):
    """
    Process multiple texts in batch

    Args:
        batch_texts: Multi-line text with one sentence per line
        reference_audio: Optional reference audio for voice cloning
        progress: Gradio progress tracker

    Returns:
        tuple: (zip_file_path, status_message)
    """
    try:
        # Load model
        tts_model = load_vietnamese_model()

        # Parse texts
        texts = [line.strip() for line in batch_texts.split('\n') if line.strip()]

        if not texts:
            return None, "❌ Không có text nào để xử lý!"

        if len(texts) > 20:
            return None, "❌ Tối đa 20 câu mỗi lần xử lý!"

        # Create temporary directory
        import zipfile
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"vietnamese_tts_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

        status_messages = []
        successful = 0

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, text in enumerate(progress.tqdm(texts, desc="Đang xử lý...")):
                try:
                    # Generate audio
                    if reference_audio:
                        audio = tts_model.generate(
                            text=text,
                            audio_prompt_path=reference_audio,
                            exaggeration=0.5,
                            temperature=0.8,
                            cfg_weight=0.5
                        )
                    else:
                        audio = tts_model.generate(
                            text=text,
                            exaggeration=0.5,
                            temperature=0.8,
                            cfg_weight=0.5
                        )

                    # Save to temporary file
                    temp_audio_path = os.path.join(temp_dir, f"audio_{i+1:03d}.wav")
                    torchaudio.save(temp_audio_path, audio.cpu(), 24000)

                    # Add to zip with descriptive name
                    safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
                    zip_filename = f"{i+1:03d}_{safe_text}.wav"
                    zipf.write(temp_audio_path, zip_filename)

                    status_messages.append(f"✅ [{i+1}/{len(texts)}] {text[:50]}...")
                    successful += 1

                except Exception as e:
                    status_messages.append(f"❌ [{i+1}/{len(texts)}] Lỗi: {e}")

        # Create status message
        status = f"🎉 Hoàn thành: {successful}/{len(texts)} thành công\n\n" + "\n".join(status_messages)

        return zip_path, status

    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return None, f"❌ Lỗi xử lý batch: {e}"

# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Vietnamese TTS Voice Cloning",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .section-header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.5em;
            font-weight: bold;
            margin: 20px 0 10px 0;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            🇻🇳 Vietnamese TTS Voice Cloning 🎵
        </div>
        <div style="text-align: center; margin-bottom: 30px;">
            <p style="font-size: 1.2em; color: #666;">
                Tạo giọng nói tiếng Việt tự nhiên với công nghệ Voice Cloning
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Voice Cloning
            with gr.Tab("🎭 Voice Cloning", elem_id="voice-clone-tab"):
                gr.HTML('<div class="section-header">🎵 Nhân bản giọng nói</div>')
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="📝 Text tiếng Việt",
                            placeholder="Nhập text tiếng Việt bạn muốn tổng hợp...",
                            lines=3,
                            max_lines=5
                        )
                        
                        reference_audio = gr.Audio(
                            label="🎤 Audio tham chiếu (Voice để clone)",
                            type="filepath",
                            format="wav"
                        )
                        
                        with gr.Row():
                            exaggeration_clone = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.5,
                                step=0.1,
                                label="🎭 Cường độ cảm xúc"
                            )
                            temperature_clone = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="🌡️ Temperature"
                            )
                            cfg_weight_clone = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="⚖️ CFG Weight"
                            )
                        
                        clone_btn = gr.Button(
                            "🎵 Tạo giọng nói",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        clone_output = gr.Audio(
                            label="🔊 Kết quả Voice Cloning",
                            type="numpy"
                        )
                        
                        gr.HTML("""
                        <div style="margin-top: 20px; padding: 15px; background: #f0f8ff; border-radius: 10px;">
                            <h4>💡 Hướng dẫn sử dụng:</h4>
                            <ul>
                                <li>📝 Nhập text tiếng Việt</li>
                                <li>🎤 Upload file audio tham chiếu (3-10 giây)</li>
                                <li>⚙️ Điều chỉnh tham số nếu cần</li>
                                <li>🎵 Nhấn "Tạo giọng nói"</li>
                            </ul>
                        </div>
                        """)
            
            # Tab 2: Default Voice
            with gr.Tab("🎤 Giọng mặc định", elem_id="default-voice-tab"):
                gr.HTML('<div class="section-header">🎤 Giọng nói mặc định</div>')
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input_default = gr.Textbox(
                            label="📝 Text tiếng Việt",
                            placeholder="Nhập text tiếng Việt bạn muốn tổng hợp...",
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            exaggeration_default = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.5,
                                step=0.1,
                                label="🎭 Cường độ cảm xúc"
                            )
                            temperature_default = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="🌡️ Temperature"
                            )
                            cfg_weight_default = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.1,
                                label="⚖️ CFG Weight"
                            )
                        
                        default_btn = gr.Button(
                            "🎤 Tạo giọng nói",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        default_output = gr.Audio(
                            label="🔊 Kết quả",
                            type="numpy"
                        )
                
                # Example texts
                gr.HTML('<div class="section-header">📚 Ví dụ</div>')
                example_texts = get_example_texts()

                with gr.Row():
                    for i in range(0, len(example_texts), 2):
                        with gr.Column():
                            for j in range(2):
                                if i + j < len(example_texts):
                                    example_btn = gr.Button(
                                        example_texts[i + j],
                                        size="sm"
                                    )
                                    # Use closure to capture the text
                                    def make_click_handler(text):
                                        return lambda: text
                                    example_btn.click(
                                        make_click_handler(example_texts[i + j]),
                                        outputs=text_input_default
                                    )

            # Tab 3: Batch Processing
            with gr.Tab("📦 Xử lý hàng loạt", elem_id="batch-tab"):
                gr.HTML('<div class="section-header">📦 Tạo nhiều audio cùng lúc</div>')

                with gr.Row():
                    with gr.Column():
                        batch_texts = gr.Textbox(
                            label="📝 Danh sách text (mỗi dòng một câu)",
                            placeholder="Nhập nhiều câu tiếng Việt, mỗi dòng một câu...",
                            lines=8,
                            max_lines=15
                        )

                        batch_reference = gr.Audio(
                            label="🎤 Audio tham chiếu (tùy chọn)",
                            type="filepath",
                            format="wav"
                        )

                        batch_btn = gr.Button(
                            "📦 Tạo tất cả",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column():
                        batch_output = gr.File(
                            label="📁 Download ZIP file",
                            file_count="single"
                        )

                        batch_status = gr.Textbox(
                            label="📊 Trạng thái",
                            interactive=False,
                            lines=5
                        )
        
        # Event handlers
        clone_btn.click(
            fn=synthesize_with_voice_clone,
            inputs=[text_input, reference_audio, exaggeration_clone, temperature_clone, cfg_weight_clone],
            outputs=clone_output
        )

        default_btn.click(
            fn=synthesize_default_voice,
            inputs=[text_input_default, exaggeration_default, temperature_default, cfg_weight_default],
            outputs=default_output
        )

        batch_btn.click(
            fn=process_batch_texts,
            inputs=[batch_texts, batch_reference],
            outputs=[batch_output, batch_status]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>🎉 Vietnamese TTS Voice Cloning</h3>
            <p>Powered by ChatterboxTTS & Vietnamese Fine-tuned Model</p>
            <p style="color: #666;">
                🔧 Model: Vietnamese TTS (1200 vocab) | 
                🎵 Voice Cloning: Reference Audio Based | 
                🚀 Device: {device}
            </p>
        </div>
        """.format(device=device.upper()))
    
    return demo

if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Set to True for public sharing
        debug=True,
        show_error=True,
        quiet=False
    )
