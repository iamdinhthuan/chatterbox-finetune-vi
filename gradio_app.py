"""Gradio interface for Vietnamese TTS inference."""

import functools
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import gradio as gr

# Reuse helper utilities from the CLI inference script
from infer import load_finetuned_model, normalize_vietnamese


# Ensure src directory is available when the module is executed directly
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _detect_device(requested: Optional[str] = None) -> str:
    if requested and requested.lower() != "auto":
        return requested.lower()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=2)
def _load_cached_model(checkpoint: str, base_model: str, device: str):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise gr.Error(f"Checkpoint not found: {checkpoint_path}")

    base_model_path = Path(base_model)
    if not base_model_path.exists():
        raise gr.Error(
            "Base model not found. Ensure the pretrained model is available or update the path."
        )

    return load_finetuned_model(checkpoint_path, base_model_path, device)


def _prepare_conditionals(model, voice_path: Optional[str], exaggeration: float):
    if voice_path:
        voice_file = Path(voice_path)
        if not voice_file.exists():
            raise gr.Error(f"Voice file not found: {voice_file}")
        model.prepare_conditionals(str(voice_file), exaggeration=exaggeration)
        return "Đã sử dụng giọng tham chiếu tải lên."

    if model.conds is None:
        dummy_seconds = 3
        rng = np.random.default_rng()
        dummy_wav = rng.normal(0.0, 0.01, 16000 * dummy_seconds).astype(np.float32)
        model.prepare_conditionals(dummy_wav, exaggeration=exaggeration)
        return "Không có giọng được cung cấp, dùng giọng ngẫu nhiên."  # noqa: E501

    model.prepare_conditionals(None, exaggeration=exaggeration)
    return "Sử dụng giọng có sẵn trong checkpoint."


def synthesize(
    checkpoint: str,
    base_model: str,
    text: str,
    temperature: float,
    cfg_weight: float,
    exaggeration: float,
    voice_file: Optional[str],
    voice_upload: Optional[str],
    device_choice: str,
):
    if not text or not text.strip():
        raise gr.Error("Vui lòng nhập văn bản cần chuyển giọng.")

    device = _detect_device(device_choice)
    model = _load_cached_model(checkpoint, base_model, device)

    normalized_text = normalize_vietnamese(text)
    sentences = [s.strip() for s in re.split(r"(?<=[\.\?])\s+", normalized_text) if s.strip()]
    if not sentences:
        sentences = [normalized_text.strip()]

    voice_source = voice_file or voice_upload
    if not voice_source:
        raise gr.Error("Vui lòng tải hoặc chọn một file giọng tham chiếu.")
    prep_message_base = _prepare_conditionals(model, voice_source, exaggeration)

    segments = []
    sr = getattr(model, "sr", 24000)

    with torch.inference_mode():
        for sentence in sentences:
            segment = model.generate(
                sentence,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            if isinstance(segment, torch.Tensor):
                segment_tensor = segment.detach().cpu().float()
            else:
                segment_tensor = torch.as_tensor(segment, dtype=torch.float32)

            if segment_tensor.ndim > 1:
                segment_tensor = segment_tensor.squeeze(0)

            segments.append(segment_tensor)

    if len(segments) == 1:
        wav_tensor = segments[0]
    else:
        wav_tensor = torch.cat(segments, dim=-1)

    wav_np = wav_tensor.numpy().astype(np.float32)
    prep_message = f"{prep_message_base} Tổng số câu: {len(sentences)}."

    return " ".join(sentences), prep_message, (sr, wav_np)


def build_interface() -> gr.Blocks:
    description = (
        "Nhập đường dẫn checkpoint đã fine-tune và văn bản tiếng Việt để tạo giọng nói."
    )

    with gr.Blocks(title="Vietnamese TTS Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""# Vietnamese TTS Inference
Sử dụng mô hình đã fine-tune để tổng hợp giọng nói tiếng Việt trực tiếp trên trình duyệt.""")

        with gr.Row():
            checkpoint_input = gr.Textbox(
                value="/home/huy/chatterbox-finetuning/checkpoint-120000/",
                label="Checkpoint",
                placeholder="Ví dụ: ./checkpoints/vietnamese/checkpoint-45000",
            )
            base_model_input = gr.Textbox(
                value="/home/huy/chatterbox-finetuning/chatterbox/",
                label="Base model",
            )

        text_input = gr.Textbox(
            label="Văn bản",
            lines=4,
            value=(
                "Dù là sinh viên xuất sắc, nhưng Linh luôn cảm thấy tự ti mỗi khi có cơ hội "
                "được tiếp xúc với người nước ngoài hoặc khi phải đọc tài liệu chuyên ngành bằng tiếng Anh."
            ),
            placeholder="Nhập văn bản tiếng Việt...",
        )

        with gr.Accordion("Tùy chỉnh", open=False):
            with gr.Row():
                temperature_input = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                cfg_input = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="CFG weight")
            with gr.Row():
                exaggeration_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Exaggeration",
                )
                device_input = gr.Radio(
                    choices=["auto", "cpu", "cuda", "mps"],
                    value="auto",
                    label="Thiết bị",
                    interactive=True,
                )
            voice_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Giọng tham chiếu (upload)",
            )
            voice_file_picker = gr.File(
                label="Chọn file giọng từ máy (bắt buộc)",
                file_types=["audio"],
                type="filepath",
            )

        synth_button = gr.Button("Tạo giọng nói")

        normalized_output = gr.Textbox(label="Văn bản sau chuẩn hoá", interactive=False)
        prep_output = gr.Textbox(label="Thông tin giọng nói", interactive=False)
        audio_output = gr.Audio(type="numpy", label="Kết quả audio")

        synth_button.click(
            synthesize,
            inputs=[
                checkpoint_input,
                base_model_input,
                text_input,
                temperature_input,
                cfg_input,
                exaggeration_input,
                voice_file_picker,
                voice_input,
                device_input,
            ],
            outputs=[normalized_output, prep_output, audio_output],
        )

        gr.Markdown(description)

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(share=True)
