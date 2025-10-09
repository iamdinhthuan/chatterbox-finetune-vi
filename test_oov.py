"""
Test OOV (Out-Of-Vocabulary) với metadata.csv
Kiểm tra xem có bao nhiêu ký tự/từ trong dataset không có trong tokenizer
"""

import json
import csv
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer


def test_oov_with_metadata(csv_path, tokenizer_path):
    """
    Test OOV với metadata.csv
    """
    
    print("="*80)
    print("TEST OOV WITH METADATA.CSV")
    print("="*80)
    
    # Load tokenizer
    print(f"\n📖 Loading tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = json.load(open(tokenizer_path, 'r', encoding='utf-8'))['model']['vocab']
    
    print(f"✅ Tokenizer loaded: {len(vocab)} tokens")
    
    # Load metadata.csv
    print(f"\n📊 Loading dataset: {csv_path}")
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    transcripts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            if 'transcript' in row:
                transcripts.append(row['transcript'])
    
    print(f"✅ Loaded {len(transcripts)} transcripts")
    
    # Analyze characters
    print("\n" + "="*80)
    print("CHARACTER ANALYSIS")
    print("="*80)
    
    all_chars = Counter()
    for text in transcripts:
        all_chars.update(text)
    
    print(f"\n📊 Total unique characters in dataset: {len(all_chars)}")
    print(f"📊 Total character occurrences: {sum(all_chars.values())}")
    
    # Check OOV characters
    oov_chars = {}
    for char, count in all_chars.items():
        if char not in vocab:
            oov_chars[char] = count
    
    if oov_chars:
        print(f"\n❌ OOV Characters: {len(oov_chars)}")
        print("\nTop 20 OOV characters:")
        for char, count in sorted(oov_chars.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  '{char}' (U+{ord(char):04X}): {count:,} occurrences")
    else:
        print(f"\n✅ NO OOV CHARACTERS! All characters covered!")
    
    # Test tokenization
    print("\n" + "="*80)
    print("TOKENIZATION TEST")
    print("="*80)
    
    total_unk = 0
    samples_with_unk = 0
    
    for text in transcripts:
        encoding = tokenizer.encode(text)
        unk_count = encoding.tokens.count('[UNK]')
        if unk_count > 0:
            total_unk += unk_count
            samples_with_unk += 1
    
    print(f"\n📊 Total samples: {len(transcripts)}")
    print(f"📊 Samples with [UNK]: {samples_with_unk} ({samples_with_unk/len(transcripts)*100:.2f}%)")
    print(f"📊 Total [UNK] tokens: {total_unk}")
    
    if samples_with_unk > 0:
        print(f"\n❌ WARNING: {samples_with_unk} samples contain [UNK] tokens!")
        print("\nExamples with [UNK]:")
        count = 0
        for text in transcripts:
            encoding = tokenizer.encode(text)
            if '[UNK]' in encoding.tokens:
                print(f"\n  Text: {text}")
                print(f"  Tokens: {encoding.tokens}")
                count += 1
                if count >= 5:
                    break
    else:
        print(f"\n✅ NO [UNK] TOKENS! Perfect tokenization!")
    
    # Sample tokenization
    print("\n" + "="*80)
    print("SAMPLE TOKENIZATION (First 10)")
    print("="*80)
    
    for i, text in enumerate(transcripts[:10]):
        encoding = tokenizer.encode(text)
        print(f"\n{i+1}. '{text}'")
        print(f"   Tokens ({len(encoding.tokens)}): {encoding.tokens[:20]}")
        if len(encoding.tokens) > 20:
            print(f"   ... (truncated, total {len(encoding.tokens)} tokens)")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    token_lengths = []
    for text in transcripts:
        encoding = tokenizer.encode(text)
        token_lengths.append(len(encoding.tokens))
    
    print(f"\n📊 Token length statistics:")
    print(f"   Min: {min(token_lengths)}")
    print(f"   Max: {max(token_lengths)}")
    print(f"   Mean: {sum(token_lengths)/len(token_lengths):.2f}")
    print(f"   Median: {sorted(token_lengths)[len(token_lengths)//2]}")
    
    # Character coverage
    print("\n" + "="*80)
    print("COVERAGE SUMMARY")
    print("="*80)
    
    coverage = (len(all_chars) - len(oov_chars)) / len(all_chars) * 100
    print(f"\n✅ Character coverage: {coverage:.2f}%")
    print(f"   - Total unique chars: {len(all_chars)}")
    print(f"   - Covered chars: {len(all_chars) - len(oov_chars)}")
    print(f"   - OOV chars: {len(oov_chars)}")
    
    if samples_with_unk == 0 and len(oov_chars) == 0:
        print("\n" + "="*80)
        print("🎉 PERFECT! TOKENIZER COVERS 100% OF DATASET!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  TOKENIZER NEEDS IMPROVEMENT")
        print("="*80)
        if len(oov_chars) > 0:
            print(f"\n💡 Add these {len(oov_chars)} OOV characters to tokenizer:")
            for char, count in sorted(oov_chars.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   '{char}' (U+{ord(char):04X})")


if __name__ == "__main__":
    csv_path = "metadata.csv"
    tokenizer_path = "VietnameseTokenizer/tokenizer.json"
    
    test_oov_with_metadata(csv_path, tokenizer_path)

