# Critical Data Format Fix

## ❌ Problem Discovered

Preprocessing và training code **KHÔNG MATCH** về data format!

### Before Fix:

**Preprocessing saved:**
```python
{
    "text_tokens": [45, 67, 89, ...],              # ❌ No BOS/EOS
    "speech_tokens": [1234, 5678, ...],            # ❌ No BOS/EOS
    "voice_emb": [0.1, 0.2, ...],                  # ❌ Wrong key name
    "audio_path": "wavs/vivoice_0.wav",
    "text": "Xin chào..."
}
```

**Training expected:**
```python
{
    "text_tokens": [255, 45, 67, ..., 0],          # ✓ BOS=255, EOS=0
    "text_token_lens": 150,                         # ❌ MISSING!
    "speech_tokens": [6561, 1234, ..., 6562],      # ✓ BOS=6561, EOS=6562
    "speech_token_lens": 800,                       # ❌ MISSING!
    "t3_cond_speaker_emb": [0.1, 0.2, ...],       # ❌ Key name wrong!
    "t3_cond_prompt_speech_tokens": [...],        # ❌ MISSING!
    "t3_cond_emotion_adv": 0.5,                    # ❌ MISSING!
}
```

---

## ⚠️ Impact

**Without this fix:**
- Training would CRASH immediately
- Data loader would return wrong format
- T3 model would receive invalid inputs
- Complete failure, não training possible!

**Source of truth:** `src/chatterbox/utils/t3dataset.py:126-160`

---

## ✅ Fix Applied (Commit `9fd94a8`)

### 1. **Add BOS/EOS Tokens**

**Text tokens:**
```python
# Before
text_tokens = [45, 67, 89, ...]

# After  
text_tokens = [255, 45, 67, 89, ..., 0]  # BOS=255 at start, EOS=0 at end
```

**Speech tokens:**
```python
# Before
speech_tokens = [1234, 5678, ...]

# After
speech_tokens = [6561, 1234, 5678, ..., 6562]  # BOS=6561 at start, EOS=6562 at end
```

**Source:** `src/chatterbox/models/t3/modules/t3_config.py`:
```python
class T3Config:
    start_text_token = 255
    stop_text_token = 0
    start_speech_token = 6561
    stop_speech_token = 6562
```

### 2. **Add Token Lengths**

```python
text_token_len = len(text_tokens)  # Track actual length (before padding)
speech_token_len = len(speech_tokens)
```

**Why needed:** Training code uses lengths for loss calculation and attention masks.

### 3. **Extract Conditioning Prompt Tokens**

```python
# Extract from beginning of audio (3 seconds)
cond_audio_segment = wav[:cond_audio_samples]

# Tokenize conditioning prompt
cond_prompt_tokens_batch, _ = speech_tokenizer.forward(
    [cond_audio_segment], 
    max_len=t3_config.speech_cond_prompt_len  # 150 tokens
)

# Pad/truncate to exact length
cond_prompt_speech_tokens = ensure_length(cond_prompt_tokens, 150)
```

**Why needed:** T3 model uses conditioning prompts for voice consistency.

### 4. **Rename Keys for Compatibility**

```python
# Before
"voice_emb": speaker_emb

# After
"t3_cond_speaker_emb": speaker_emb
```

**Why:** Training code specifically looks for `t3_cond_speaker_emb` key.

### 5. **Add Emotion Scalar**

```python
"t3_cond_emotion_adv": torch.tensor(0.5, dtype=torch.float)
```

**Why:** T3 training uses emotion adversarial loss (default value = 0.5).

---

## 📝 Complete Fixed Format

```python
{
    # Text (with BOS/EOS)
    "text_tokens": torch.Tensor([255, 45, 67, ..., 0]),    # [seq_len]
    "text_token_lens": torch.Tensor(150),                   # scalar
    
    # Speech (with BOS/EOS)
    "speech_tokens": torch.Tensor([6561, 1234, ..., 6562]),  # [seq_len]
    "speech_token_lens": torch.Tensor(800),                   # scalar
    
    # Conditioning
    "t3_cond_speaker_emb": torch.Tensor([0.1, 0.2, ...]),    # [256]
    "t3_cond_prompt_speech_tokens": torch.Tensor([...]),     # [150]
    "t3_cond_emotion_adv": torch.Tensor(0.5),                # scalar
    
    # Metadata
    "audio_path": "wavs/vivoice_0.wav",
    "text": "Xin chào các bạn"
}
```

---

## 🔧 Code Changes

### preprocess_dataset.py

**1. Import T3Config:**
```python
from chatterbox.models.t3.modules.t3_config import T3Config
```

**2. Load T3 config:**
```python
t3_config = tts.t3.hp  # Access T3 hyperparameters
```

**3. Update preprocess_sample():**
```python
def preprocess_sample(
    ...,
    t3_config: T3Config,  # ← New parameter
    ...
):
    # Text with BOS/EOS
    raw_text_tokens = text_tokenizer.encode(text_normalized).ids
    text_tokens = [t3_config.start_text_token] + raw_text_tokens + [t3_config.stop_text_token]
    text_token_len = len(text_tokens)
    
    # Speech with BOS/EOS
    raw_speech_tokens = speech_tokenizer.forward([wav])[0]
    speech_tokens = torch.cat([
        torch.tensor([t3_config.start_speech_token]),
        raw_speech_tokens,
        torch.tensor([t3_config.stop_speech_token])
    ])
    speech_token_len = len(speech_tokens)
    
    # Conditioning prompt
    cond_audio_segment = wav[:cond_audio_samples]
    cond_prompt_tokens = speech_tokenizer.forward(
        [cond_audio_segment], 
        max_len=t3_config.speech_cond_prompt_len
    )[0]
    
    # Return updated format
    return {
        "text_tokens": text_tensor,
        "text_token_lens": torch.tensor(text_token_len),
        "speech_tokens": speech_tokens,
        "speech_token_lens": torch.tensor(speech_token_len),
        "t3_cond_speaker_emb": speaker_emb,
        "t3_cond_prompt_speech_tokens": cond_prompt_tokens,
        "t3_cond_emotion_adv": torch.tensor(0.5),
        "audio_path": str(audio_path),
        "text": text,
    }
```

### src/chatterbox/utils/preprocessed_dataset.py

**Updated collate_fn:**
```python
def collate_fn_preprocessed(batch):
    return {
        'text_tokens': torch.stack([item['text_tokens'] for item in batch]),
        'text_token_lens': torch.stack([item['text_token_lens'] for item in batch]),
        'speech_tokens': torch.stack([item['speech_tokens'] for item in batch]),
        'speech_token_lens': torch.stack([item['speech_token_lens'] for item in batch]),
        't3_cond_speaker_emb': torch.stack([item['t3_cond_speaker_emb'] for item in batch]),
        't3_cond_prompt_speech_tokens': torch.stack([item['t3_cond_prompt_speech_tokens'] for item in batch]),
        't3_cond_emotion_adv': torch.stack([item['t3_cond_emotion_adv'] for item in batch]),
    }
```

---

## ⚠️ IMPORTANT: Re-run Preprocessing!

**Old preprocessed data (before commit `9fd94a8`) is INVALID!**

```bash
# Stop current preprocessing if running
Ctrl+C

# Delete old preprocessed data
rm -rf preprocessed_data/

# Re-run with fixed code
git pull origin main
python preprocess_dataset.py --csv metadata.csv --audio_dir wavs --add_silence
```

**Why:** Old .pt files have wrong format and will cause training to crash.

---

## 🧪 How to Verify

After re-preprocessing, check one .pt file:

```python
import torch

# Load one sample
sample = torch.load('preprocessed_data/sample_000000.pt')

# Check keys
print("Keys:", sample.keys())
# Should have: text_tokens, text_token_lens, speech_tokens, speech_token_lens,
#              t3_cond_speaker_emb, t3_cond_prompt_speech_tokens, t3_cond_emotion_adv

# Check text tokens
print("Text tokens:", sample['text_tokens'][:10])
# First should be 255 (BOS), last should be 0 (EOS)

# Check speech tokens  
print("Speech tokens:", sample['speech_tokens'][:5], "...", sample['speech_tokens'][-5:])
# First should be 6561 (BOS), last should be 6562 (EOS)

# Check shapes
print("Text len:", sample['text_token_lens'])
print("Speech len:", sample['speech_token_lens'])
print("Speaker emb shape:", sample['t3_cond_speaker_emb'].shape)  # Should be [256]
print("Prompt tokens shape:", sample['t3_cond_prompt_speech_tokens'].shape)  # Should be [150]
print("Emotion adv:", sample['t3_cond_emotion_adv'])  # Should be 0.5
```

**Expected output:**
```
Keys: dict_keys(['text_tokens', 'text_token_lens', 'speech_tokens', 'speech_token_lens', 't3_cond_speaker_emb', 't3_cond_prompt_speech_tokens', 't3_cond_emotion_adv', 'audio_path', 'text'])

Text tokens: tensor([255,  45,  67,  89, 123, 156, ...])
Speech tokens: tensor([6561, 1234, 5678, ...]) ... tensor([..., 9876, 5432, 6562])

Text len: tensor(152)
Speech len: tensor(805)
Speaker emb shape: torch.Size([256])
Prompt tokens shape: torch.Size([150])
Emotion adv: tensor(0.5000)
```

---

## 📚 References

**Training data format source:**
- `src/chatterbox/utils/t3dataset.py:126-160` - Original SpeechFineTuningDataset.__getitem__()
- `src/finetune_t3_thai.py:334-379` - SpeechFineTuningIterableDataset.__getitem__()

**T3 Config:**
- `src/chatterbox/models/t3/modules/t3_config.py` - BOS/EOS token IDs

**Model expectations:**
- `src/chatterbox/models/t3/t3.py:31-34` - Asserts BOS/EOS presence
- `src/finetune_t3_thai.py:1360-1361` - Data collator expects these keys

---

## ✅ Verification Checklist

- [x] BOS/EOS tokens added to text
- [x] BOS/EOS tokens added to speech
- [x] Token lengths included
- [x] Conditioning prompt tokens extracted
- [x] Key names match training expectations
- [x] Emotion scalar included
- [x] Collate function updated
- [ ] **Re-run preprocessing with fixed code**
- [ ] Verify .pt file format
- [ ] Test training with preprocessed data

---

## 🎉 After Re-preprocessing

Training should work with preprocessed data:

```bash
python train.py \
  --csv metadata.csv \
  --use_preprocessed \
  --epochs 10 \
  --batch_size 8
```

**Expected:** No data format errors, training starts smoothly, 2-4x faster than without preprocessing!
