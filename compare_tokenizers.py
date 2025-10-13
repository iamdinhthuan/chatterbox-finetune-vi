"""Compare char tokenizer vs BPE tokenizer"""
import json

print("="*60)
print("TOKENIZER COMPARISON")
print("="*60)

# Load tokenizers
with open('tokenizer_char.json', 'r', encoding='utf-8') as f:
    char_tok = json.load(f)

with open('VietnameseTokenizer/tokenizer.json', 'r', encoding='utf-8') as f:
    bpe_tok = json.load(f)

# Stats
print("\n📊 STATISTICS:")
print("-"*60)

char_vocab = char_tok['model']['vocab']
char_merges = char_tok['model'].get('merges', [])

bpe_vocab = bpe_tok['model']['vocab']
bpe_merges = bpe_tok['model'].get('merges', [])

print(f"Char Tokenizer:")
print(f"  Vocab size: {len(char_vocab)}")
print(f"  Merges: {len(char_merges)}")
print(f"  File size: 13,792 bytes")

print(f"\nBPE Tokenizer:")
print(f"  Vocab size: {len(bpe_vocab)}")
print(f"  Merges: {len(bpe_merges)}")
print(f"  File size: 46,311 bytes")

# Sample vocab
print("\n📝 SAMPLE VOCAB (First 30):")
print("-"*60)

print("\nChar Tokenizer:")
for i, (token, idx) in enumerate(list(char_vocab.items())[:30]):
    print(f"  {idx:3d}: {repr(token)}")

print("\nBPE Tokenizer:")
for i, (token, idx) in enumerate(list(bpe_vocab.items())[:30]):
    print(f"  {idx:3d}: {repr(token)}")

# Check special tokens
print("\n🔖 SPECIAL TOKENS:")
print("-"*60)

char_special = [t for t in char_tok.get('added_tokens', []) if t['special']]
bpe_special = [t for t in bpe_tok.get('added_tokens', []) if t['special']]

print(f"\nChar Tokenizer ({len(char_special)} special tokens):")
for t in char_special[:10]:
    print(f"  ID {t['id']:3d}: {t['content']}")

print(f"\nBPE Tokenizer ({len(bpe_special)} special tokens):")
for t in bpe_special[:10]:
    print(f"  ID {t['id']:3d}: {t['content']}")

# Test tokenization
print("\n🧪 TOKENIZATION TEST:")
print("-"*60)

test_texts = [
    "Xin chào các bạn",
    "Hôm nay trời đẹp",
    "Tôi là một người Việt Nam"
]

from tokenizers import Tokenizer

char_tokenizer = Tokenizer.from_file('tokenizer_char.json')
bpe_tokenizer = Tokenizer.from_file('VietnameseTokenizer/tokenizer.json')

for text in test_texts:
    char_enc = char_tokenizer.encode(text)
    bpe_enc = bpe_tokenizer.encode(text)
    
    print(f"\nText: '{text}'")
    print(f"  Char: {len(char_enc.ids):2d} tokens - {char_enc.ids}")
    print(f"  BPE:  {len(bpe_enc.ids):2d} tokens - {bpe_enc.ids}")
    print(f"  Efficiency: BPE uses {len(char_enc.ids) - len(bpe_enc.ids)} fewer tokens")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

avg_char = sum(len(char_tokenizer.encode(t).ids) for t in test_texts) / len(test_texts)
avg_bpe = sum(len(bpe_tokenizer.encode(t).ids) for t in test_texts) / len(test_texts)

print(f"\nAverage tokens per sentence:")
print(f"  Char: {avg_char:.1f}")
print(f"  BPE:  {avg_bpe:.1f}")
print(f"  BPE is {(1 - avg_bpe/avg_char)*100:.1f}% more efficient")

print(f"\nTRAINING SPEED:")
print(f"  Char: 0.15 epochs → clear Vietnamese ✅")
print(f"  BPE:  0.40 epochs → not clear yet ❌")

print(f"\nREASON:")
print(f"  Char tokenizer is CLOSER to pretrained English tokenizer")
print(f"  → Text embeddings partially match")
print(f"  → Faster convergence")
print(f"")
print(f"  BPE tokenizer is DIFFERENT from pretrained")
print(f"  → Text embeddings must learn from scratch")
print(f"  → Slower convergence")

print("\n⚠️ RECOMMENDATION:")
print("  Use CHAR TOKENIZER for faster training!")
print("  Trade-off: slightly less efficient tokenization")
print("  But: 3x faster convergence (0.15 vs 0.4+ epochs)")
