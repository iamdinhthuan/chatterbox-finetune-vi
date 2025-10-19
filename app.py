"""
Gradio Web App for Vietnamese TTS (Chatterbox Fine-tuned)
Based on infer.py with enhanced UI
"""

import sys
import os
from pathlib import Path
import torch
import torchaudio as ta
import gradio as gr
import tempfile
import re
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.chatterbox.tts import ChatterboxTTS


# ==================== Text Processing Functions ====================

def normalize_text(text: str) -> str:
    """
    Normalize text for TTS (inspired by F5-TTS)
    - Remove multiple spaces
    - Fix punctuation spacing
    - Normalize quotes and dashes
    """
    if not text or len(text.strip()) == 0:
        return text
    
    # Remove multiple spaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Fix punctuation - remove space before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Fix punctuation - add space after punctuation if missing
    text = re.sub(r'([,.!?;:])([^\s\d])', r'\1 \2', text)
    
    # Normalize quotes and dashes (Vietnamese style)
    replacements = [
        ('"', '"'),   # Smart quote to straight quote
        ('"', '"'),   # Smart quote to straight quote  
        (''', "'"),   # Smart apostrophe
        (''', "'"),   # Smart apostrophe
        ('–', '-'),   # En dash to hyphen
        ('—', '-'),   # Em dash to hyphen
        ('…', '...'), # Ellipsis
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def split_sentences(text: str, min_length=10):
    """
    Split text into sentences by punctuation (inspired by F5-TTS)
    Supports: . ? ! ; (both English and Vietnamese punctuation)
    """
    # Split by sentence-ending punctuation
    pattern = r'([.?!。！？]+|[;；]+)'
    
    # Split while keeping delimiters
    parts = re.split(pattern, text)
    
    # Combine text with its punctuation
    sentences = []
    i = 0
    
    while i < len(parts):
        if i >= len(parts):
            break
            
        # Get text part
        sentence = parts[i].strip()
        
        # Get punctuation if exists
        if i + 1 < len(parts) and re.match(pattern, parts[i+1]):
            sentence += parts[i+1]
            i += 2
        else:
            i += 1
        
        # Only keep non-empty sentences above minimum length
        if sentence and len(sentence.strip()) >= min_length:
            sentences.append(sentence)
    
    # If no sentences found, use original text
    return sentences if sentences else [text]


def cross_fade_audio(audio_segments, sample_rate=24000, fade_duration_ms=100):
    """
    Merge audio segments with cross-fading
    """
    if len(audio_segments) == 0:
        return torch.zeros(1, 0)
    
    if len(audio_segments) == 1:
        return audio_segments[0]
    
    # Calculate fade samples
    fade_samples = int(sample_rate * fade_duration_ms / 1000)
    
    # Start with first segment
    result = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        next_segment = audio_segments[i]
        
        # Create fade out/in windows
        if result.shape[-1] >= fade_samples and next_segment.shape[-1] >= fade_samples:
            fade_out = torch.linspace(1, 0, fade_samples).to(result.device)
            fade_in = torch.linspace(0, 1, fade_samples).to(next_segment.device)
            
            # Apply fade
            result[..., -fade_samples:] = result[..., -fade_samples:] * fade_out
            next_segment[..., :fade_samples] = next_segment[..., :fade_samples] * fade_in
            
            # Overlap and merge
            overlap = result[..., -fade_samples:] + next_segment[..., :fade_samples]
            result = torch.cat([result[..., :-fade_samples], overlap, next_segment[..., fade_samples:]], dim=-1)
        else:
            # No fade, just concatenate
            result = torch.cat([result, next_segment], dim=-1)
    
    return result


# ==================== Model Management ====================

class ModelManager:
    def __init__(self):
        self.model = None
        self.device = None
        self.checkpoint_path = None
        self.base_model_path = None
        
    def find_latest_checkpoint(self, checkpoint_path):
        """Find the latest checkpoint file in directory"""
        checkpoint_path = Path(checkpoint_path)
        
        # If it's a file, return it
        if checkpoint_path.is_file():
            return checkpoint_path
        
        # If it's a directory, find latest checkpoint
        if checkpoint_path.is_dir():
            # Look for checkpoint subdirectories (checkpoint-XXXXX)
            checkpoint_dirs = sorted(
                [d for d in checkpoint_path.glob("checkpoint-*") if d.is_dir()],
                key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
                reverse=True
            )
            
            if checkpoint_dirs:
                return checkpoint_dirs[0]
            
            # No checkpoint subdirs, use the directory itself
            return checkpoint_path
        
        return checkpoint_path
    
    def load_model(self, checkpoint_path, base_model_path, device):
        """Load or reload model if paths changed (following infer.py logic)"""
        if (self.model is None or 
            self.checkpoint_path != checkpoint_path or 
            self.base_model_path != base_model_path or
            self.device != device):
            
            from safetensors.torch import load_file
            from src.chatterbox.models.t3 import T3
            from src.chatterbox.models.s3gen import S3Gen
            from src.chatterbox.models.voice_encoder import VoiceEncoder
            from src.chatterbox.models.tokenizers import EnTokenizer
            
            print(f"🔄 Loading model...")
            
            checkpoint_path = Path(checkpoint_path)
            base_model_path = Path(base_model_path)
            
            # Find latest checkpoint
            checkpoint_path = self.find_latest_checkpoint(checkpoint_path)
            print(f"📁 Using checkpoint: {checkpoint_path}")
            
            # Check if checkpoint has full model files
            has_full_model = (checkpoint_path / "ve.safetensors").exists()
            
            if has_full_model:
                # Checkpoint has all files, load directly
                print("   ✓ Found complete model in checkpoint")
                self.model = ChatterboxTTS.from_local(str(checkpoint_path), device=device)
            else:
                # Checkpoint only has T3, need to combine with pretrained
                print("   ✓ Loading finetuned T3 from checkpoint")
                print("   ✓ Loading pretrained VE/S3Gen from base model")
                
                # Load pretrained components
                ve = VoiceEncoder()
                ve.load_state_dict(load_file(base_model_path / "ve.safetensors"))
                ve.to(device).eval()
                
                s3gen = S3Gen()
                s3gen.load_state_dict(load_file(base_model_path / "s3gen.safetensors"), strict=False)
                s3gen.to(device).eval()
                
                # Load finetuned T3
                t3 = T3()
                
                # Try different checkpoint filenames
                checkpoint_file = None
                for filename in ["model.safetensors", "pytorch_model.safetensors", "t3_cfg.safetensors"]:
                    if (checkpoint_path / filename).exists():
                        checkpoint_file = checkpoint_path / filename
                        break
                
                if checkpoint_file is None:
                    raise FileNotFoundError(f"No model checkpoint found in {checkpoint_path}")
                
                print(f"   Loading from: {checkpoint_file.name}")
                t3_checkpoint = load_file(checkpoint_file)
                
                # Extract T3 state dict - checkpoint has "t3." prefix
                t3_state = {}
                for key, value in t3_checkpoint.items():
                    if key.startswith("t3."):
                        new_key = key.replace("t3.", "", 1)
                        t3_state[new_key] = value
                
                if not t3_state:
                    # No "t3." prefix, use as is
                    t3_state = t3_checkpoint
                
                print(f"   Loaded {len(t3_state)} T3 parameters")
                t3.load_state_dict(t3_state)
                t3.to(device).eval()
                
                # Load tokenizer
                tokenizer_path = checkpoint_path / "tokenizer.json"
                if not tokenizer_path.exists():
                    tokenizer_path = base_model_path / "tokenizer.json"
                tokenizer = EnTokenizer(str(tokenizer_path))
                
                # Load conds if available
                conds = None
                conds_path = checkpoint_path / "conds.pt"
                if not conds_path.exists():
                    conds_path = base_model_path / "conds.pt"
                if conds_path.exists():
                    from src.chatterbox.tts import Conditionals
                    map_location = torch.device('cpu') if device in ["cpu", "mps"] else None
                    conds = Conditionals.load(str(conds_path), map_location=map_location).to(device)
                
                self.model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
            
            self.checkpoint_path = checkpoint_path
            self.base_model_path = base_model_path
            self.device = device
            
            print(f"✅ Model loaded successfully on {device}")
        
        return self.model


# Global model manager
model_manager = ModelManager()


# ==================== TTS Generation Function ====================

def generate_tts(
    text,
    checkpoint_path,
    base_model_path,
    voice_audio,
    temperature,
    cfg_weight,
    exaggeration,
    seed,
    min_tokens,
    split_by_sentences,
    fade_duration,
    device
):
    """Main TTS generation function"""
    
    try:
        # Validate inputs
        if not text or not text.strip():
            return None, "❌ Please enter text to generate!"
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None, f"❌ Checkpoint not found: {checkpoint_path}"
        
        if not base_model_path or not Path(base_model_path).exists():
            return None, f"❌ Base model not found: {base_model_path}"
        
        # Set random seed
        if seed >= 0:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Auto-detect device if not specified
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        status_lines = []
        status_lines.append(f"🖥️  Device: {device}")
        status_lines.append(f"📁 Checkpoint: {Path(checkpoint_path).name}")
        
        # Load model
        model = model_manager.load_model(checkpoint_path, base_model_path, device)
        status_lines.append("✅ Model loaded")
        
        # Normalize text
        normalized_text = normalize_text(text)
        status_lines.append(f"📝 Original text: {text}")
        if normalized_text != text:
            status_lines.append(f"📝 Normalized: {normalized_text}")
        
        # Prepare voice conditioning
        if voice_audio is not None:
            status_lines.append(f"🎤 Loading reference voice...")
            model.prepare_conditionals(voice_audio, exaggeration=exaggeration)
            status_lines.append("✅ Voice conditioning prepared")
        else:
            if model.conds is None:
                status_lines.append("⚠️  Using random conditioning")
                dummy_wav = np.random.randn(16000 * 3).astype(np.float32) * 0.01
                model.prepare_conditionals(dummy_wav, exaggeration=exaggeration)
            else:
                status_lines.append("✅ Using built-in voice from checkpoint")
        
        # Generate speech
        status_lines.append(f"\n🎵 Generating speech...")
        
        if split_by_sentences:
            # Split text into sentences
            sentences = split_sentences(normalized_text)
            status_lines.append(f"📝 Split into {len(sentences)} sentences:")
            for i, sent in enumerate(sentences, 1):
                preview = sent[:60] + "..." if len(sent) > 60 else sent
                status_lines.append(f"   {i}. {preview}")
            
            # Generate audio for each sentence
            audio_segments = []
            for i, sentence in enumerate(sentences, 1):
                status_lines.append(f"\n🎙️  Generating sentence {i}/{len(sentences)}...")
                
                wav_seg = model.generate(
                    sentence,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    min_tokens=min_tokens,
                )
                audio_segments.append(wav_seg)
                
                seg_duration = wav_seg.shape[-1] / model.sr
                status_lines.append(f"   ✓ Generated {seg_duration:.2f}s")
            
            # Merge audio segments with cross-fading
            status_lines.append(f"\n🔗 Merging {len(audio_segments)} segments with {fade_duration}ms cross-fade...")
            wav = cross_fade_audio(audio_segments, sample_rate=model.sr, fade_duration_ms=fade_duration)
            status_lines.append(f"   ✓ Merged successfully")
        else:
            # Generate full text at once
            wav = model.generate(
                normalized_text,
                temperature=temperature,
                cfg_weight=cfg_weight,
                min_tokens=min_tokens,
            )
        
        # Check audio quality
        duration = wav.shape[-1] / model.sr
        status_lines.append(f"\n✅ Generation complete!")
        status_lines.append(f"⏱️  Duration: {duration:.2f}s")
        status_lines.append(f"🔊 Sample rate: {model.sr}Hz")
        
        if duration < 0.5:
            status_lines.append(f"⚠️  WARNING: Audio is very short ({duration:.2f}s)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
            ta.save(output_path, wav.cpu(), model.sr)
        
        return output_path, "\n".join(status_lines)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


# ==================== Gradio Interface ====================

def create_interface():
    """Create Gradio interface"""
    
    # Default paths
    default_checkpoint = "./checkpoints/vietnamese"
    default_base_model = "./vietnamese/pretrained_model_download"
    
    with gr.Blocks(
        title="Vietnamese TTS - Chatterbox Fine-tuned",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🎤 Vietnamese TTS - Chatterbox Fine-tuned
        
        Fine-tuned Vietnamese text-to-speech model with advanced features:
        - ✨ Text normalization (spaces, punctuation, quotes)
        - 📝 Smart sentence splitting (by . ? ! ;)
        - 🔊 Cross-fade merging for smooth audio
        - 🎯 Reproducible generation with seed
        - 🎛️ Adjustable temperature, CFG weight, emotion
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")
                
                text_input = gr.Textbox(
                    label="Text to Generate",
                    placeholder="Enter Vietnamese text here...\nExample: Xin chào các bạn. Hôm nay trời đẹp quá!",
                    lines=6,
                    value="Xin chào các bạn. Hôm nay là một ngày đẹp trời!"
                )
                
                voice_input = gr.Audio(
                    label="Reference Voice (Optional)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                gr.Markdown("### ⚙️ Settings")
                
                with gr.Accordion("🎛️ Generation Parameters", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more natural but varied (0.5-1.0 recommended)"
                    )
                    
                    cfg_weight = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="CFG Weight",
                        info="Classifier-free guidance strength (0.5-1.0 recommended)"
                    )
                    
                    exaggeration = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Emotion Exaggeration",
                        info="Voice emotion intensity"
                    )
                
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        info="For reproducible generation (-1 for random)"
                    )
                    
                    min_tokens = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        label="Minimum Tokens",
                        info="Prevent early stopping"
                    )
                    
                    split_by_sentences = gr.Checkbox(
                        value=True,
                        label="Split by Sentences",
                        info="Generate each sentence separately and merge"
                    )
                    
                    fade_duration = gr.Slider(
                        minimum=0,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Cross-fade Duration (ms)",
                        info="For sentence merging"
                    )
                
                with gr.Accordion("📂 Model Paths", open=False):
                    checkpoint_path = gr.Textbox(
                        value=default_checkpoint,
                        label="Checkpoint Path",
                        placeholder="./checkpoints/vietnamese"
                    )
                    
                    base_model_path = gr.Textbox(
                        value=default_base_model,
                        label="Base Model Path",
                        placeholder="/checkpoints/vietnamese/pretrained_model_download"
                    )
                    
                    device = gr.Radio(
                        choices=["auto", "cuda", "cpu", "mps"],
                        value="auto",
                        label="Device",
                        info="auto = auto-detect"
                    )
                
                generate_btn = gr.Button(
                    "🎵 Generate Speech",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 Output")
                
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    interactive=False
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=20,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Examples
        gr.Markdown("### 📚 Examples")
        gr.Examples(
            examples=[
                ["Xin chào các bạn. Hôm nay là một ngày đẹp trời!"],
                ["Tôi rất vui được gặp bạn. Bạn có khỏe không?"],
                ["Chào buổi sáng! Hôm nay chúng ta sẽ học về trí tuệ nhân tạo."],
                ["Cảm ơn bạn đã lắng nghe. Hẹn gặp lại!"],
            ],
            inputs=[text_input]
        )
        
        # Usage guide
        with gr.Accordion("📖 Usage Guide", open=False):
            gr.Markdown("""
            ### 🚀 How to Use:
            
            1. **Enter text** in the input box (Vietnamese text recommended)
            2. **Upload reference voice** (optional) - the model will clone this voice
            3. **Adjust parameters**:
               - **Temperature**: 0.5-0.7 = stable, 0.8-1.0 = more natural
               - **CFG Weight**: 0.5-1.0 recommended for quality
               - **Emotion**: 0.0 = neutral, 1.0 = expressive
            4. **Click Generate** and wait for the audio
            
            ### ✨ Features:
            
            - **Text Normalization**: Automatically fixes spaces, punctuation, quotes
            - **Sentence Splitting**: Long text is split by . ? ! ; and merged smoothly
            - **Reproducible**: Same seed = same output
            - **Cross-fade**: Smooth transitions between sentences
            
            ### 💡 Tips:
            
            - Use proper punctuation (. ? ! ;) for better results
            - Split by sentences for long text (> 2-3 sentences)
            - Higher temperature = more variation but may be less stable
            - Use reference voice for voice cloning
            - Set seed ≥ 0 for reproducible results
            
            ### 🐛 Troubleshooting:
            
            - **Audio too short**: Increase min_tokens
            - **Inconsistent quality**: Lower temperature, enable sentence splitting
            - **Model not loading**: Check checkpoint and base model paths
            """)
        
        # Event handler
        generate_btn.click(
            fn=generate_tts,
            inputs=[
                text_input,
                checkpoint_path,
                base_model_path,
                voice_input,
                temperature,
                cfg_weight,
                exaggeration,
                seed,
                min_tokens,
                split_by_sentences,
                fade_duration,
                device
            ],
            outputs=[audio_output, status_output]
        )
    
    return app


# ==================== Main ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese TTS Gradio App")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    
    args = parser.parse_args()
    
    app = create_interface()
    
    print("=" * 80)
    print("🎤 Vietnamese TTS Gradio App")
    print("=" * 80)
    print(f"🌐 Starting server on {args.server_name}:{args.server_port}")
    if args.share:
        print("🔗 Public share link will be generated")
    print("=" * 80)
    
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_api=False
    )
