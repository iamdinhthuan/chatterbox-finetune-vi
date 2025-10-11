# Vietnamese Tokenizer Training

Train Vietnamese BPE tokenizer from your corpus while preserving special tokens from pretrained model.

## Quick Start

### 1. Prepare your data

Create `metadata.csv` with format:
```
audio_path|transcript
audio/file1.wav|Xin chào các bạn
audio/file2.wav|Đây là tiếng Việt
```

### 2. Train tokenizer

```bash
python train_tokenizer_from_corpus.py metadata.csv tokenizer.json
```

**Arguments:**
- `metadata.csv` - Your corpus file (required)
- `tokenizer.json` - Original pretrained tokenizer (default: `tokenizer.json`)
- `VietnameseTokenizer` - Output directory (default: `VietnameseTokenizer`)

**Output:**
```
VietnameseTokenizer/
├── tokenizer.json      # Trained tokenizer (use this for training)
└── vocab_list.txt      # Human-readable vocabulary
```

### 3. Use for training

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("VietnameseTokenizer/tokenizer.json")
encoding = tokenizer.encode("Xin chào")
print(encoding.tokens)  # ['X', 'in', 'ch', 'ào']
```

## Features

✅ **Corpus-based**: Learns optimal BPE merges from your Vietnamese data  
✅ **Special tokens preserved**: All expressive tokens ([giggle], [whisper]...) keep their positions  
✅ **No OOV**: 0% out-of-vocabulary on training corpus  
✅ **Efficient**: ~50% fewer tokens than character-level tokenization

## How it works

1. **Extract special tokens** from original pretrained tokenizer (positions 0-2, 255, 604-639, 695-703)
2. **Train BPE** on your Vietnamese corpus with remaining positions
3. **Merge vocabularies** while preserving special token positions
4. **Verify** all special tokens are at correct positions

## Test OOV Coverage

After training, check coverage on your dataset:

```bash
python test_oov.py
```

This shows:
- Out-of-vocabulary characters
- [UNK] token count
- Tokenization examples

## Requirements

```bash
pip install tokenizers
```

## Example Results

**Before (Original English tokenizer):**
```
Text: "Tiếng Việt rất hay"
Tokens: 14 → ['T', 'i', '[UNK]', 'n', 'g', 'V', 'i', '[UNK]', 't', ...]
```

**After (Vietnamese tokenizer):**
```
Text: "Tiếng Việt rất hay"
Tokens: 7 → ['T', 'iế', 'ng', 'V', 'iệt', 'rất', 'hay']
```

**Special tokens work:**
```
Text: "[giggle] Xin chào [whisper]"
Tokens: ['[giggle]', 'X', 'in', 'ch', 'ào', '[whisper]']
```

## Technical Details

- **Vocab size**: 704 tokens (same as original)
- **Special tokens**: 49 preserved at original positions
- **Vietnamese tokens**: ~655 learned from corpus
- **BPE merges**: ~460-470 merge operations
- **Min frequency**: 2 (only learns tokens that appear ≥2 times)

## Troubleshooting

**Script fails with "file not found":**
- Check `metadata.csv` exists
- Check `tokenizer.json` exists (original pretrained tokenizer)

**High OOV rate after training:**
- Ensure metadata.csv uses `|` delimiter
- Check transcript column is named "transcript"
- Verify corpus has enough samples (recommended: >10k sentences)

**[UNK] tokens in output:**
- Run `test_oov.py` to identify problematic characters
- Check if your corpus has unusual characters not in Unicode Vietnamese range

## Files

- `train_tokenizer_from_corpus.py` - Main training script
- `test_oov.py` - OOV testing script (optional)
- `tokenizer.json` - Original pretrained tokenizer (input)
- `metadata.csv` - Your corpus (input)
- `VietnameseTokenizer/` - Output directory

## Best Practices

1. **Use large corpus**: More data → better BPE merges (recommended: >100k sentences)
2. **Clean data**: Remove corrupted text, normalize Unicode
3. **Keep special tokens**: Never modify positions of [STOP], [START], expressive tokens
4. **Test before training**: Run `test_oov.py` to verify 0% OOV rate

## License

Same as Chatterbox TTS model.
