# Vietnamese TTS Fine-tuning with Chatterbox

Fine-tune [ResembleAI/Chatterbox](https://github.com/resemble-ai/chatterbox) for Vietnamese text-to-speech.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

**Format:** CSV with pipe delimiter
```csv
audio|transcript
wavs/audio_001.wav|Xin chào các bạn
wavs/audio_002.wav|Hôm nay trời đẹp
```

### 3. Train Vietnamese Tokenizer

```bash
python train_tokenizer_from_corpus.py metadata.csv
```

**Output:** `VietnameseTokenizer/tokenizer.json` (704 tokens, preserves 49 special tokens)

### 4. Preprocess Dataset (Optional but Recommended)

**Why?** 2-4x faster training by pre-computing audio features offline.

**Test first (10 seconds):**
```bash
python test_preprocessing_single.py
```

**Single-threaded (22 hours for 2.6M samples):**
```bash
python preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --add_silence
```

**Multi-threaded (3-6 hours, **recommended**):**
```bash
python preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --add_silence \
  --num_workers 8
```

**Workers:** Use `nproc` to check CPU cores. Recommend: 4-8 workers.

**Verify after preprocessing:**
```bash
python verify_preprocessing_format.py
```

### 5. Train

**Without preprocessing:**
```bash
python train.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --epochs 5 \
  --batch_size 4
```

**With preprocessing (2-4x faster):**
```bash
python train.py \
  --csv metadata.csv \
  --use_preprocessed \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5
```

### 6. Inference

```bash
python infer.py \
  --checkpoint ./checkpoints/vietnamese/checkpoint-XXXXX \
  --text "Xin chào, đây là bản demo tiếng Việt" \
  --reference_audio ./sample.wav \
  --output ./output.wav
```

---

## 📊 Dataset Requirements

- **Format:** Pipe-delimited CSV (`audio|transcript`)
- **Audio:** 16kHz or higher, mono, WAV format
- **Text:** Vietnamese text (normalized)
- **Size:** Minimum 1 hour, recommended 10+ hours

**Example structure:**
```
project/
├── metadata.csv
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
```

---

## 🔧 Training Configuration

### Recommended Settings

| Dataset Size | Epochs | Batch Size | Learning Rate | Training Time* |
|--------------|--------|------------|---------------|----------------|
| 1-5 hours | 5-10 | 4 | 1e-5 | 1-3 days |
| 5-20 hours | 5-8 | 8 | 1e-5 | 3-7 days |
| 20+ hours | 3-5 | 8-16 | 1e-5 | 1-2 weeks |

*Without preprocessing. With preprocessing: 2-4x faster.

### Advanced Options

```bash
python train.py \
  --csv metadata.csv \
  --use_preprocessed \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5 \
  --gradient_accumulation_steps 2 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --max_steps 100000
```

---

## ⚡ Preprocessing Performance

### Speed Comparison

| Mode | Workers | Speed | Time (2.6M samples) | Speedup |
|------|---------|-------|---------------------|---------|
| **None** | - | - | Training: ~5-10 days | 1x |
| **Preprocessed** | 1 | 32 it/s | 22 hours + Training: ~2 days | 2-3x |
| **Preprocessed** | 4 | 120 it/s | 6 hours + Training: ~2 days | 2-3x |
| **Preprocessed** | 8 | 220 it/s | **3.3 hours** + Training: ~2 days | **2-3x** |

### What Gets Preprocessed?

Preprocessing saves:
- ✅ Text tokens (with BOS/EOS)
- ✅ Speech tokens (with BOS/EOS)
- ✅ Speaker embeddings (256 dims)
- ✅ Conditioning prompt tokens (150 tokens)
- ✅ Emotion scalar (0.5)

**NOT saved:** Raw audio (use original files for inference)

**Storage:** ~20-25 GB for 2.6M samples (~8 KB per sample)

---

## 🐛 Troubleshooting

### Common Issues

**1. "S3Tokenizer object has no attribute 'encode'"**
- **Fixed in latest version.** Pull latest code: `git pull origin main`

**2. "input.size(-1) must be equal to input_size. Expected 40, got 48000"**
- **Fixed in latest version.** Uses `embeds_from_wavs()` instead of direct call.

**3. Data format mismatch errors during training**
- **Solution:** Re-run preprocessing with latest code. Old .pt files are incompatible.
- Delete old data: `rm -rf preprocessed_data/`
- Re-preprocess: `python preprocess_dataset.py --csv metadata.csv --audio_dir wavs --add_silence`

**4. Out of memory (OOM)**
- Reduce batch size: `--batch_size 2`
- Enable gradient accumulation: `--gradient_accumulation_steps 4`
- Reduce max lengths: `--max_text_len 128 --max_speech_len 800`

**5. Training very slow**
- Use preprocessing: `python preprocess_dataset.py ... --num_workers 8`
- Enable bf16: Already enabled by default
- Increase batch size if memory allows: `--batch_size 16`

---

## 📝 Tokenizer Details

### Vietnamese Tokenizer Stats

- **Total tokens:** 704 (matches pretrained model)
- **Vietnamese vocab:** 655 tokens
- **Special tokens:** 49 preserved from pretrained model
  - Text: BOS=255, EOS=0
  - Speech: BOS=6561, EOS=6562
- **Training method:** BPE (Byte Pair Encoding) on 2.6M samples
- **Coverage:** 100% on Vietnamese corpus

### Special Token Positions

```
Positions 0-2: [0, 1, 2]
Position 255: [255]
Positions 604-639: [604, 605, ..., 639]
Positions 695-703: [695, 696, ..., 703]
```

**Why preserve?** Pretrained model expects these exact token IDs. Changing them breaks the model.

---

## 🧪 Verification Scripts

### Test Single Sample (10 seconds)

```bash
python test_preprocessing_single.py --sample_idx 0
```

**Verifies:**
- ✅ Audio loading
- ✅ Model loading
- ✅ Preprocessing format
- ✅ BOS/EOS tokens (255, 0, 6561, 6562)
- ✅ All required fields present

**Output:** "VERIFICATION PASSED" = safe to run full preprocessing

### Verify Preprocessed Data

```bash
python verify_preprocessing_format.py
```

**Checks:**
- Keys present
- Data types correct
- Tensor shapes correct
- BOS/EOS tokens correct
- Lengths match

---

## 📂 File Structure

```
chatterbox-finetuning/
├── README.md                          # This file
├── WARP.md                            # Developer guide
├── metadata.csv                       # Your training data
├── wavs/                              # Audio files
├── VietnameseTokenizer/
│   └── tokenizer.json                 # Trained tokenizer
├── preprocessed_data/                 # Preprocessed features (optional)
│   ├── sample_000000.pt
│   ├── sample_000001.pt
│   └── metadata.json
├── checkpoints/                       # Training checkpoints
│   └── vietnamese/
├── train_tokenizer_from_corpus.py    # Train tokenizer
├── preprocess_dataset.py              # Preprocess dataset
├── train.py                           # Training script
├── infer.py                           # Inference script
├── test_preprocessing_single.py       # Test preprocessing
└── verify_preprocessing_format.py     # Verify format
```

---

## 🔬 Technical Details

### Data Format (Preprocessed)

Each `.pt` file contains:

```python
{
    "text_tokens": torch.Tensor([255, 45, 67, ..., 0]),         # [seq_len], dtype=long
    "text_token_lens": torch.Tensor(152),                        # scalar, dtype=long
    "speech_tokens": torch.Tensor([6561, 1234, ..., 6562]),     # [seq_len], dtype=long
    "speech_token_lens": torch.Tensor(805),                      # scalar, dtype=long
    "t3_cond_speaker_emb": torch.Tensor([0.1, 0.2, ...]),      # [256], dtype=float
    "t3_cond_prompt_speech_tokens": torch.Tensor([...]),        # [150], dtype=long
    "t3_cond_emotion_adv": torch.Tensor(0.5),                   # scalar, dtype=float
    "audio_path": "wavs/vivoice_0.wav",
    "text": "Xin chào các bạn"
}
```

### Model Architecture

- **Base model:** ResembleAI/Chatterbox
- **Frozen:** Voice Encoder, S3Gen (speech-to-waveform)
- **Trainable:** T3 (text-to-speech tokens)
- **Tokenizers:**
  - Text: Custom Vietnamese BPE (704 tokens)
  - Speech: S3Tokenizer (6561 tokens)

### Training Process

1. **Text → Text Tokens** (Vietnamese tokenizer)
2. **Audio → Speech Tokens** (S3Tokenizer, frozen)
3. **Audio → Speaker Embedding** (Voice Encoder, frozen)
4. **Text Tokens → Speech Tokens** (T3 model, **trainable**)
5. **Speech Tokens → Waveform** (S3Gen, frozen)

**Only T3 is fine-tuned.** Other components remain frozen.

---

## 📊 Memory Requirements

### Training

| Batch Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| 4 | ~12 GB | RTX 3080 (12GB) |
| 8 | ~20 GB | RTX 3090 (24GB) |
| 16 | ~40 GB | A100 (40GB) |

### Preprocessing

| Workers | RAM | GPU Memory |
|---------|-----|------------|
| 1 | ~8 GB | ~6 GB |
| 4 | ~20 GB | ~6 GB (shared) |
| 8 | ~35 GB | ~6 GB (shared) |

**Tip:** Each worker loads its own model copy in memory.

---

## 🎯 Expected Results

### Training Metrics

- **Loss:** Should decrease from ~8-10 to ~2-4
- **Evaluation:** Check every 5000 steps
- **Overfitting:** If eval loss increases, reduce epochs

### Inference Quality

- **Intelligibility:** Should be clear after 5+ epochs
- **Naturalness:** Improves with more data (10+ hours)
- **Speaker similarity:** Depends on reference audio quality

---

## 🤝 Contributing

See [WARP.md](WARP.md) for development guidelines.

---

## 📄 License

Same as [Chatterbox](https://github.com/resemble-ai/chatterbox).

---

## 🙏 Credits

- Base model: [ResembleAI/Chatterbox](https://github.com/resemble-ai/chatterbox)
- Preprocessing optimization: [Issue #174](https://github.com/resemble-ai/chatterbox/issues/174)

---

## 📚 Additional Resources

- [Chatterbox Paper](https://arxiv.org/abs/your-paper-link)
- [Model Card](https://huggingface.co/ResembleAI/chatterbox)
- [Demo](https://resemble.ai)

---

## ⚠️ Important Notes

### Data Format Fix (Dec 2024)

**If you preprocessed data before Dec 12, 2024, you MUST re-preprocess!**

**Old format (missing):**
- ❌ No BOS/EOS tokens
- ❌ Missing token lengths
- ❌ Missing conditioning prompts

**New format (correct):**
- ✅ BOS/EOS tokens (255, 0, 6561, 6562)
- ✅ Token lengths included
- ✅ All conditioning fields

**How to check:**
```bash
python verify_preprocessing_format.py
```

**If fails:** Delete old data and re-preprocess with latest code.

### Multiprocessing (Dec 2024)

**New in latest version:** Multi-threaded preprocessing for 4-10x speedup.

**Usage:**
```bash
python preprocess_dataset.py \
  --csv metadata.csv \
  --audio_dir wavs \
  --add_silence \
  --num_workers 8  # 4-10x faster!
```

---

## 🆘 Getting Help

1. Check troubleshooting section above
2. Run verification scripts (`test_preprocessing_single.py`, `verify_preprocessing_format.py`)
3. Check [WARP.md](WARP.md) for developer details
4. Open issue on GitHub with:
   - Command used
   - Error message
   - System info (`nvidia-smi`, `python --version`)

---

## 📈 Changelog

### Latest (Dec 2024)
- ✅ Fixed data format mismatch (BOS/EOS, lengths, conditioning)
- ✅ Added multiprocessing support (4-10x faster preprocessing)
- ✅ Added verification scripts
- ✅ Improved documentation

### Previous
- Added preprocessing optimization (2-4x training speedup)
- Vietnamese tokenizer training
- Initial release

---

**Ready to start? Run:**

```bash
# 1. Train tokenizer
python train_tokenizer_from_corpus.py metadata.csv

# 2. Test preprocessing (10 seconds)
python test_preprocessing_single.py

# 3. Preprocess (3-6 hours with 8 workers)
python preprocess_dataset.py --csv metadata.csv --audio_dir wavs --add_silence --num_workers 8

# 4. Train (2-3 days with preprocessing)
python train.py --csv metadata.csv --use_preprocessed --epochs 10 --batch_size 8
```

**Questions? Check WARP.md or open an issue!** 🚀
