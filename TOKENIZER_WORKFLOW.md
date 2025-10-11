# Vietnamese Tokenizer - Complete Workflow

## ✅ Final Setup

### Files Structure

```
chatterbox-finetuning/
├── train_tokenizer_from_corpus.py    # Main training script
├── test_oov.py                        # OOV testing (optional)
├── TOKENIZER_README.md                # Detailed documentation
├── tokenizer.json                     # Original pretrained tokenizer (input)
├── metadata.csv                       # Your Vietnamese corpus (input)
└── VietnameseTokenizer/              # Trained tokenizer (output)
    ├── tokenizer.json                # ← Use this for TTS training
    └── vocab_list.txt                # Human-readable vocab
```

---

## 🚀 Quick Start

### Step 1: Train Tokenizer

```bash
python train_tokenizer_from_corpus.py metadata.csv
```

**Output:**
```
✅ Loaded 2,604,620 sentences
✅ Training completed: 655 tokens
✅ Final vocab: 703 tokens
✅ BPE merges: 465
✅ All 49 special tokens preserved!
```

### Step 2: Verify (Optional)

```bash
python test_oov.py
```

Should show:
- 0% OOV rate
- 0 [UNK] tokens
- Perfect coverage

### Step 3: Use for Training

Update your training script to use the new tokenizer:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("VietnameseTokenizer/tokenizer.json")
```

---

## 📊 Results

### Trained Tokenizer Stats

- **Vocab size**: 703 tokens
- **Special tokens**: 49 (preserved from original)
- **Vietnamese tokens**: 654 (learned from corpus)
- **BPE merges**: 465
- **Training corpus**: 2.6M Vietnamese sentences

### Performance Comparison

| Metric | Original (EN) | Vietnamese | Improvement |
|--------|--------------|------------|-------------|
| Tokens for "Tiếng Việt rất hay" | 14 | 7 | **50% fewer** |
| [UNK] tokens | Yes | **0** | **100% better** |
| Special tokens | ✅ | ✅ | **Preserved** |

### Example Tokenizations

```python
# Vietnamese text
tokenizer.encode("Tiếng Việt rất hay")
# → ['T', 'iế', 'ng', 'V', 'iệt', 'rất', 'hay']  # 7 tokens

tokenizer.encode("Tôi đang học TTS")
# → ['Tôi', 'đang', 'học', 'T', 'T', 'S']  # 6 tokens

# Special tokens work
tokenizer.encode("[giggle] Xin chào [whisper]")
# → ['[giggle]', 'X', 'in', 'ch', 'ào', '[whisper]']
```

---

## 🔧 Technical Details

### Special Tokens Preserved

All special tokens from pretrained model are preserved at exact positions:

- **Core**: [STOP] (0), [UNK] (1), [SPACE] (2), [START] (255)
- **Expressive**: [UH], [UM], [giggle], [laughter], [whisper], [groan]... (604-639)
- **Placeholder**: [PLACEHOLDER55-63] (695-703)

### BPE Training Config

- **Min frequency**: 2 (only learns tokens appearing ≥2 times)
- **Pre-tokenizer**: Whitespace splitting
- **Model**: Byte Pair Encoding (BPE)
- **Dropout**: None
- **Language**: Vietnamese (vi)

### Vocabulary Allocation

```
Position Range    | Content
------------------|------------------------------------------
0-2               | Core special tokens
3-254             | Vietnamese characters, bigrams, words
255               | [START] token
256-603           | Vietnamese BPE tokens
604-639           | Expressive TTS tokens
640-694           | Vietnamese tokens (extended)
695-703           | Placeholder tokens
```

---

## 📝 Best Practices

### ✅ DO

- Use large corpus (>100k sentences recommended)
- Clean your text data before training
- Test with `test_oov.py` after training
- Keep `tokenizer.json` (original) as backup

### ❌ DON'T

- Don't modify special token positions manually
- Don't train with corpus <10k sentences (may overfit)
- Don't use corrupted or mixed-language text
- Don't remove expressive tokens

---

## 🐛 Troubleshooting

### Issue: High [UNK] rate after training

**Solution:**
```bash
python test_oov.py  # Check which characters are missing
# Re-train with cleaned corpus
```

### Issue: Special tokens not working

**Check:**
```python
import json
vocab = json.load(open('VietnameseTokenizer/tokenizer.json'))['model']['vocab']
print(vocab['[giggle]'])  # Should be 606
print(vocab['[whisper]'])  # Should be 622
```

### Issue: Training too slow

**For large corpus:**
- Use SSD for faster I/O
- Reduce corpus size for initial testing
- Training time: ~3-5 minutes for 2.6M sentences on modern CPU

---

## 📚 Additional Resources

- **Detailed docs**: See `TOKENIZER_README.md`
- **OOV testing**: Run `test_oov.py`
- **Training script**: `train_tokenizer_from_corpus.py` (well-commented)

---

## ✨ Summary

**What you have now:**
- ✅ Optimized Vietnamese BPE tokenizer
- ✅ Preserved special tokens from pretrained model
- ✅ 0% OOV on your corpus
- ✅ ~50% more efficient than character-level
- ✅ Ready for TTS training

**Next steps:**
1. Use `VietnameseTokenizer/tokenizer.json` in your TTS training
2. Monitor training metrics
3. Enjoy better Vietnamese TTS quality! 🎉

---

**Version**: 1.0 Final  
**Date**: Trained from 2.6M Vietnamese sentences  
**Status**: ✅ Production Ready
