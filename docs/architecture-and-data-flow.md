# Chatterbox TTS Architecture and Data Flow

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Key Components](#key-components)
5. [Fine-tuning Strategies](#fine-tuning-strategies)
6. [Implementation Details](#implementation-details)

## Overview

Chatterbox TTS is a production-grade open-source text-to-speech system from Resemble AI. It uses a two-stage generation pipeline with a 0.5B parameter LLaMA-based model (T3) for text-to-token conversion and a flow-based acoustic model (S3Gen) for token-to-speech synthesis.

### Key Features
- 🎭 **Emotion Exaggeration Control**: Unique feature for controlling speech expressiveness
- 🌏 **Multi-language Support**: Extensible to different languages (Thai support added)
- 🔒 **Built-in Watermarking**: Perth watermarking for responsible AI
- 🎯 **Zero-shot Voice Cloning**: Generate speech in any voice with just a reference audio

## System Architecture

### High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Text Input    │────▶│    T3 Model     │────▶│   S3Gen Model   │────▶ Audio Output
└─────────────────┘     │  (Text→Tokens)  │     │ (Tokens→Speech) │     
                        └─────────────────┘     └─────────────────┘
                               ▲                        ▲
                               │                        │
                        ┌──────┴────────┐      ┌───────┴────────┐
                        │ Text Features │      │ Audio Features │
                        │ - Tokenization│      │ - Mel Spectra  │
                        │ - Embeddings  │      │ - Speaker Emb  │
                        └───────────────┘      └────────────────┘
```

### Component Overview

1. **T3 (Text-to-Token-to-Speech)**
   - 520M parameter LLaMA-based model
   - Converts text to discrete speech tokens
   - Handles linguistic and prosodic features

2. **S3Gen (Speech Synthesis Generator)**
   - Flow-based generative model
   - Converts speech tokens to high-quality waveforms
   - Includes HiFi-GAN vocoder for audio synthesis

3. **Supporting Components**
   - **Voice Encoder**: Extracts speaker embeddings for voice cloning
   - **S3 Tokenizer**: Converts audio to discrete tokens (16kHz)
   - **Text Tokenizer**: Processes input text
   - **Perth Watermarker**: Adds imperceptible watermarks

## Data Flow Diagrams

### TTS Inference Pipeline

```
┌─────────────────┐                          ┌──────────────────┐
│   Input Text    │                          │ Reference Audio  │
└────────┬────────┘                          └────────┬─────────┘
         │                                            │
         ▼                                            ├─────────────┬──────────────┐
┌─────────────────┐                                   ▼             ▼              ▼
│ Text Normalizer │                          ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
│   (punc_norm)   │                          │ Voice Encoder  │ │ S3 Tokenizer │ │ Mel Extractor│
└────────┬────────┘                          │ (VoiceEncoder) │ │  @ 16kHz     │ │  @ 24kHz     │
         │                                   └────────┬───────┘ └──────┬───────┘ └──────┬───────┘
         ▼                                            │                │                │
┌─────────────────┐                                   ▼                ▼                ▼
│ Text Tokenizer  │                          ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
│  (EnTokenizer)  │                          │    Speaker     │ │   Prompt     │ │  Prompt Mel  │
└────────┬────────┘                          │   Embedding    │ │   Tokens     │ │              │
         │                                   └────────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                                            │                │                │
         │                                            └────────────────┴────────────────┘
         │                                                             │
         ▼                                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    T3 Model (0.5B LLaMA)                              │
│                              Text → Speech Token Generation                           │
└──────────────────────────────────────────────┬───────────────────────────────────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  Speech Tokens   │
                                      └────────┬─────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         │                                     ▼                                     │
┌────────┴────────────────────────────────────────────────────────────────┐          │
│                        S3Gen Flow Model (CausalMaskedDiff)               │◄─────────┘
│                         Token → Mel Spectrogram Generation               │
└──────────────────────────────────────────────┬───────────────────────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ Mel Spectrogram  │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ HiFi-GAN Vocoder│
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ Audio @ 24kHz   │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │Perth Watermarker │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │   Final Audio   │
                                      └──────────────────┘
```

### T3 Fine-tuning Data Flow

```
┌───────────────────────┐
│ Dataset Audio/Text    │
│       Pairs           │
└──────────┬────────────┘
           │
           ├────────────────────┬─────────────────────┐
           ▼                    │                     ▼
┌──────────────────────┐        │          ┌──────────────────────┐
│   Load & Resample    │        │          │  Text Normalization  │
│   to 16kHz           │        │          │     (punc_norm)      │
└──────────┬───────────┘        │          └──────────┬───────────┘
           │                    │                     │
           ▼                    │                     ▼
┌──────────────────────┐        │          ┌──────────────────────┐
│    Audio @ 16kHz     │        │          │    Text Tokenizer    │
└──────────┬───────────┘        │          │    (EnTokenizer)     │
           │                    │          └──────────┬───────────┘
           ├──────────┬─────────┘                     │
           ▼          ▼                               ▼
┌──────────────┐ ┌──────────────┐          ┌──────────────────────┐
│Voice Encoder │ │ S3 Tokenizer │          │    Text Tokens       │
│   [FROZEN]   │ │   [FROZEN]   │          └──────────┬───────────┘
└──────┬───────┘ └──────┬───────┘                     │
       │                │                              │
       ▼                ▼                              │
┌──────────────┐ ┌──────────────┐                     │
│   Speaker    │ │Speech Tokens │                     │
│  Embeddings  │ │ + Prompts    │                     │
└──────┬───────┘ └──────┬───────┘                     │
       │                │                              │
       └────────────────┴──────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │      T3 Model Training           │
        │        [TRAINABLE]               │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │       Loss Calculation           │
        │    (Cross-Entropy Loss)          │
        └───────────────────────────────────┘
```

### S3Gen Fine-tuning Data Flow

```
┌───────────────────────┐
│    Dataset Audio      │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Dual Resampling     │
└──────────┬────────────┘
           │
           ├────────────────────┬─────────────────────┐
           ▼                    ▼                     ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Audio @ 16kHz      │ │  Audio @ 24kHz   │ │  Audio @ 16kHz   │
└──────────┬───────────┘ └──────────┬───────┘ └──────────┬───────┘
           │                         │                     │
           ▼                         ▼                     ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   S3 Tokenizer       │ │   Mel Extractor  │ │ CAMPPlus Encoder │
│     [FROZEN]         │ │                  │ │    [FROZEN]      │
└──────────┬───────────┘ └──────────┬───────┘ └──────────┬───────┘
           │                         │                     │
           ▼                         ▼                     ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Target S3 Tokens    │ │   Target Mel     │ │ Speaker Embedding│
│  + Prompt Tokens     │ │  + Prompt Mel    │ │                  │
└──────────┬───────────┘ └──────────┬───────┘ └──────────┬───────┘
           │                         │                     │
           └─────────────────────────┴─────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │   Flow Model Training         │
                    │   (CausalMaskedDiff)          │
                    │      [TRAINABLE]              │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      CFM Loss Calculation     │
                    │ (Conditional Flow Matching)   │
                    └────────────────────────────────┘
                    
                    ┌────────────────────────────────┐
                    │   HiFi-GAN Vocoder            │
                    │      [FROZEN]                 │
                    └────────────────────────────────┘
```

## Key Components

### 1. Text Processing Pipeline

**Location**: `src/chatterbox/tts.py:22-61`

The `punc_norm()` function handles text normalization:
- Capitalizes first letter
- Removes multiple spaces
- Replaces uncommon punctuation
- Ensures proper sentence endings

### 2. ChatterboxTTS Class

**Location**: `src/chatterbox/tts.py:106-249`

Main interface orchestrating the TTS process:

```python
class ChatterboxTTS:
    def __init__(self, t3, s3gen, ve, tokenizer, device, conds=None):
        # Initialize all components
        
    def generate(self, text, audio_prompt_path=None, ...):
        # Main synthesis pipeline
        1. Prepare conditionals from audio
        2. Normalize and tokenize text
        3. Generate speech tokens (T3)
        4. Convert to speech (S3Gen)
        5. Apply watermarking
```

### 3. Training Dataset Classes

#### T3 Dataset (`src/finetune_t3.py:108-200`)
- Handles audio/text pairs
- Extracts speaker embeddings
- Creates prompt segments
- Prepares training batches

#### S3Gen Dataset (`src/finetune_s3gen.py:77-273`)
- Dual sample rate processing (16kHz/24kHz)
- Mel spectrogram extraction
- Prompt/target pair creation

### 4. Model Components

#### T3 Model
- **Architecture**: 520M parameter LLaMA variant
- **Vocab Size**: 704 text tokens, 8194 speech tokens
- **Special Features**: Emotion adversarial training

#### S3Gen Model
- **Flow Model**: CausalMaskedDiffWithXvec
- **Vocoder**: HiFi-GAN
- **Speaker Encoder**: CAMPPlus (x-vector based)

## Fine-tuning Strategies

### Frozen Component Strategy

```
T3 Fine-tuning:
├── ❄️ Frozen: Voice Encoder
├── ❄️ Frozen: S3Gen (entire model)
└── 🔥 Trainable: T3 Model

S3Gen Fine-tuning:
├── ❄️ Frozen: HiFi-GAN Vocoder
├── ❄️ Frozen: Speaker Encoder (optional)
├── ❄️ Frozen: S3 Tokenizer (optional)
└── 🔥 Trainable: Flow Model
```

### Training Parameters

#### T3 Fine-tuning
```bash
python finetune_t3.py \
  --output_dir ./checkpoints/chatterbox_finetuned \
  --model_name_or_path ResembleAI/chatterbox \
  --dataset_name <your_dataset> \
  --max_text_len 256 \
  --max_speech_len 800 \
  --audio_prompt_duration_s 3.0 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5
```

#### S3Gen Fine-tuning
```bash
python finetune_s3gen.py \
  --output_dir ./checkpoints/s3gen_finetuned \
  --model_name_or_path ResembleAI/chatterbox \
  --dataset_name <your_dataset> \
  --max_speech_token_len 750 \
  --max_mel_len 1500 \
  --prompt_audio_duration_s 3.0
```

## Implementation Details

### Sample Rate Management

| Component | Sample Rate | Purpose |
|-----------|------------|---------|
| S3 Tokenizer | 16 kHz | Speech tokenization |
| Voice Encoder | 16 kHz | Speaker embeddings |
| Mel Extractor | 24 kHz | Acoustic features |
| Final Audio | 24 kHz | Output waveform |

### Multi-Modal Conditioning

The system uses multiple conditioning signals:

1. **Linguistic**: Text tokens provide content
2. **Speaker**: Voice embeddings capture timbre
3. **Prosodic**: Prompt tokens/mel provide style
4. **Emotion**: Exaggeration parameter controls expressiveness

### Training Data Processing

Key steps in dataset preparation:

```python
# From finetune_t3.py:173-200
def __getitem__(self, idx):
    # 1. Load and resample audio to 16kHz
    wav_16k, text = self._load_audio_text_from_item(idx)
    
    # 2. Extract speaker embedding
    speaker_emb = self.voice_encoder.embeds_from_wavs([wav_16k])
    
    # 3. Normalize and tokenize text
    normalized_text = punc_norm(text)
    text_tokens = self.text_tokenizer.text_to_tokens(normalized_text)
    
    # 4. Generate speech tokens
    speech_tokens = self.speech_tokenizer.forward([wav_16k])
    
    # 5. Create prompt segments
    prompt_tokens = speech_tokens[:prompt_len]
    
    # 6. Return training batch
    return {
        "text_tokens": text_tokens,
        "speech_tokens": speech_tokens,
        "speaker_emb": speaker_emb,
        "prompt_tokens": prompt_tokens
    }
```

### Loss Functions

- **T3**: Cross-entropy loss for token prediction
- **S3Gen**: Conditional Flow Matching (CFM) loss for continuous generation

### Memory Optimization

- Gradient accumulation for larger effective batch sizes
- Mixed precision training (fp16) support
- Efficient data loading with HuggingFace datasets

## Best Practices

1. **Data Preparation**
   - Ensure high-quality audio recordings
   - Consistent text normalization
   - Balanced speaker distribution

2. **Fine-tuning Tips**
   - Start with small learning rates (5e-5)
   - Monitor validation loss for overfitting
   - Use early stopping when available

3. **Inference Optimization**
   - Adjust `cfg_weight` for quality/speed trade-off
   - Lower `temperature` for more consistent output
   - Cache speaker embeddings for same-voice synthesis

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use fp16 training

2. **Poor Audio Quality**
   - Check input audio sample rate
   - Verify text normalization
   - Adjust exaggeration parameter

3. **Slow Training**
   - Enable multi-GPU training
   - Increase number of workers
   - Use cached datasets

## References

- [Chatterbox GitHub Repository](https://github.com/resemble-ai/chatterbox)
- [Model on Hugging Face](https://huggingface.co/ResembleAI/chatterbox)
- [Perth Watermarking](https://github.com/resemble-ai/perth)