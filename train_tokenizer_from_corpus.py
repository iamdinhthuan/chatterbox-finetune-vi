"""
Train Vietnamese BPE tokenizer from corpus
Preserves special token positions from original pretrained model
"""

import json
import csv
import shutil
from pathlib import Path
from collections import OrderedDict
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def load_corpus(csv_path: str) -> list[str]:
    """Load texts from metadata CSV"""
    texts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            if 'transcript' in row and row['transcript'].strip():
                texts.append(row['transcript'].strip())
    return texts


def extract_special_tokens(tokenizer_json: dict) -> dict[int, str]:
    """Extract special tokens and their positions from tokenizer"""
    vocab = tokenizer_json['model']['vocab']
    special_tokens = {}
    for token, token_id in vocab.items():
        if token.startswith('[') and token.endswith(']'):
            special_tokens[token_id] = token
    return special_tokens


def train_bpe_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    """Train BPE tokenizer on corpus"""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]"],
        min_frequency=2,
        show_progress=True,
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def build_final_vocab(
    temp_tokenizer: Tokenizer,
    special_tokens_map: dict[int, str],
    vocab_size: int
) -> tuple[OrderedDict, list]:
    """Build final vocabulary with preserved special token positions"""
    
    # Get available positions (excluding special tokens)
    reserved_positions = set(special_tokens_map.keys())
    if 303 in reserved_positions:  # '[' bracket token
        reserved_positions.remove(303)
    available_positions = sorted(set(range(vocab_size)) - reserved_positions)
    
    # Save temp tokenizer to extract merges
    temp_dir = Path("temp_tokenizer")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "temp.json"
    temp_tokenizer.save(str(temp_file))
    
    with open(temp_file, 'r', encoding='utf-8') as f:
        temp_json = json.load(f)
    
    merges = temp_json['model'].get('merges', [])
    
    # Map Vietnamese tokens to available positions
    temp_vocab = temp_tokenizer.get_vocab()
    if "[UNK]" in temp_vocab:
        del temp_vocab["[UNK]"]
    
    temp_vocab_items = sorted(temp_vocab.items(), key=lambda x: x[1])
    vietnamese_mapping = {
        token: available_positions[idx]
        for idx, (token, _) in enumerate(temp_vocab_items)
        if idx < len(available_positions)
    }
    
    # Combine special tokens + Vietnamese tokens
    all_tokens = {}
    for tid, token in special_tokens_map.items():
        all_tokens[tid] = token
    for token, tid in vietnamese_mapping.items():
        all_tokens[tid] = token
    
    # Sort by ID
    final_vocab = OrderedDict()
    for tid in sorted(all_tokens.keys()):
        final_vocab[all_tokens[tid]] = tid
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return final_vocab, merges


def save_tokenizer(
    vocab: OrderedDict,
    merges: list,
    special_tokens_map: dict[int, str],
    original_tokenizer_json: dict,
    output_dir: Path
):
    """Save tokenizer JSON and vocab list"""
    
    # Build added_tokens list
    added_tokens = [
        {
            "id": tid,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        }
        for tid, token in special_tokens_map.items()
    ]
    
    # Create tokenizer JSON
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": original_tokenizer_json.get("normalizer"),
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": original_tokenizer_json.get("post_processor"),
        "decoder": original_tokenizer_json.get("decoder"),
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
            "language": "vi"
        }
    }
    
    # Save tokenizer.json
    output_dir.mkdir(exist_ok=True)
    tokenizer_file = output_dir / "tokenizer.json"
    with open(tokenizer_file, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    
    # Save vocab_list.txt
    vocab_list_file = output_dir / "vocab_list.txt"
    with open(vocab_list_file, "w", encoding="utf-8") as f:
        for token, tid in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{tid}\t{token}\n")
    
    return tokenizer_file


def verify_special_tokens(
    final_vocab: OrderedDict,
    special_tokens_map: dict[int, str]
) -> bool:
    """Verify all special tokens are preserved"""
    preserved = 0
    for tid, token in special_tokens_map.items():
        if token in final_vocab and final_vocab[token] == tid:
            preserved += 1
    return preserved == len(special_tokens_map)


def train_tokenizer(
    corpus_csv: str,
    original_tokenizer_path: str,
    output_dir: str = "VietnameseTokenizer",
    vocab_size: int = 704
):
    """
    Train Vietnamese tokenizer from corpus with preserved special tokens
    
    Args:
        corpus_csv: Path to metadata.csv (format: audio_path|transcript)
        original_tokenizer_path: Path to original tokenizer.json
        output_dir: Output directory (default: VietnameseTokenizer)
        vocab_size: Target vocabulary size (default: 704)
    """
    
    print("="*80)
    print("TRAIN VIETNAMESE TOKENIZER FROM CORPUS")
    print("="*80)
    
    # Load original tokenizer
    print(f"\n📖 Loading original tokenizer: {original_tokenizer_path}")
    with open(original_tokenizer_path, 'r', encoding='utf-8') as f:
        original_tokenizer_json = json.load(f)
    
    special_tokens_map = extract_special_tokens(original_tokenizer_json)
    print(f"✅ Found {len(special_tokens_map)} special tokens")
    
    # Load corpus
    print(f"\n📖 Loading corpus: {corpus_csv}")
    texts = load_corpus(corpus_csv)
    print(f"✅ Loaded {len(texts):,} sentences")
    print(f"📊 Total characters: {sum(len(t) for t in texts):,}")
    
    # Calculate vocab allocation
    reserved_count = len(special_tokens_map) + (1 if 303 in special_tokens_map else 0)
    available_for_vietnamese = vocab_size - reserved_count
    print(f"\n📊 Vocabulary allocation:")
    print(f"   Total: {vocab_size}")
    print(f"   Special tokens: {len(special_tokens_map)}")
    print(f"   Vietnamese: {available_for_vietnamese}")
    
    # Train BPE tokenizer
    print(f"\n🏋️ Training BPE tokenizer...")
    temp_tokenizer = train_bpe_tokenizer(texts, available_for_vietnamese)
    print(f"✅ Training completed: {temp_tokenizer.get_vocab_size()} tokens")
    
    # Build final vocabulary with preserved positions
    print(f"\n🔧 Building final vocabulary...")
    final_vocab, merges = build_final_vocab(
        temp_tokenizer, special_tokens_map, vocab_size
    )
    print(f"✅ Final vocab: {len(final_vocab)} tokens")
    print(f"✅ BPE merges: {len(merges)}")
    
    # Save tokenizer
    print(f"\n💾 Saving tokenizer...")
    output_path = Path(output_dir)
    tokenizer_file = save_tokenizer(
        final_vocab, merges, special_tokens_map,
        original_tokenizer_json, output_path
    )
    print(f"✅ Saved: {tokenizer_file}")
    print(f"✅ Saved: {output_path / 'vocab_list.txt'}")
    
    # Verify
    print(f"\n🔍 Verifying special tokens...")
    if verify_special_tokens(final_vocab, special_tokens_map):
        print(f"✅ All {len(special_tokens_map)} special tokens preserved!")
    else:
        print(f"⚠️  Some special tokens may be missing")
    
    # Test
    print(f"\n🧪 Testing tokenizer...")
    test_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    
    test_samples = texts[:3] if len(texts) >= 3 else texts
    for i, text in enumerate(test_samples, 1):
        encoding = test_tokenizer.encode(text)
        tokens_preview = encoding.tokens[:10]
        if len(encoding.tokens) > 10:
            tokens_preview.append("...")
        print(f"\n{i}. '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   → {len(encoding.tokens)} tokens: {tokens_preview}")
    
    # Test special tokens
    special_test = "[giggle] Xin chào [whisper]"
    encoding = test_tokenizer.encode(special_test)
    print(f"\nSpecial tokens test: '{special_test}'")
    print(f"   → {encoding.tokens}")
    
    print("\n" + "="*80)
    print("✅ DONE! Tokenizer ready for training")
    print("="*80)
    print(f"\nUsage:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tokenizer = Tokenizer.from_file('{output_dir}/tokenizer.json')")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python train_tokenizer_from_corpus.py <corpus.csv> [original_tokenizer.json] [output_dir]")
        print("\nExample:")
        print("  python train_tokenizer_from_corpus.py metadata.csv tokenizer.json VietnameseTokenizer")
        sys.exit(1)
    
    corpus_csv = sys.argv[1]
    original_tokenizer = sys.argv[2] if len(sys.argv) > 2 else "tokenizer.json"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "VietnameseTokenizer"
    
    # Validate inputs
    if not Path(corpus_csv).exists():
        print(f"❌ Corpus file not found: {corpus_csv}")
        sys.exit(1)
    
    if not Path(original_tokenizer).exists():
        print(f"❌ Original tokenizer not found: {original_tokenizer}")
        sys.exit(1)
    
    # Train tokenizer
    train_tokenizer(
        corpus_csv=corpus_csv,
        original_tokenizer_path=original_tokenizer,
        output_dir=output_dir,
        vocab_size=704
    )
