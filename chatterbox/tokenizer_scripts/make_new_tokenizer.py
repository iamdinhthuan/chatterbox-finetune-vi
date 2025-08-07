#!/usr/bin/env python3

import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC

def load_existing_vocab(existing_tokenizer_path: str) -> set:
    try:
        with open(existing_tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        existing_vocab = set()
        if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
            existing_vocab = set(tokenizer_data['model']['vocab'].keys())
        
        if 'added_tokens' in tokenizer_data:
            for token_info in tokenizer_data['added_tokens']:
                if 'content' in token_info:
                    existing_vocab.add(token_info['content'])
        
        return existing_vocab
    
    except Exception:
        return set()

def create_japanese_tokenizer(text_file: str, vocab_size: int = 500, output_path: str = "japanese_tokenizer.json", 
                             existing_tokenizer_path: str = None) -> str:
    existing_vocab = set()
    if existing_tokenizer_path:
        existing_vocab = load_existing_vocab(existing_tokenizer_path)
    
    special_tokens = ["[UNK]", "[START]", "[STOP]", "[SPACE]"]
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    initial_vocab_size = vocab_size + len(existing_vocab) + 100
    trainer = BpeTrainer(
        vocab_size=initial_vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=False
    )
    
    tokenizer.train([text_file], trainer)
    
    if existing_vocab:
        tokenizer = filter_existing_tokens(tokenizer, existing_vocab, vocab_size, special_tokens)
    
    tokenizer.save(output_path)
    return output_path

def derive_merges_from_vocab(vocab_dict):
    tokens_by_id = sorted(vocab_dict.items(), key=lambda x: x[1])
    
    base_chars = set()
    compound_tokens = []
    
    for token, token_id in tokens_by_id:
        if token.startswith('[') and token.endswith(']'):
            continue
        elif len(token) == 1:
            base_chars.add(token)
        elif len(token) > 1:
            compound_tokens.append((token, token_id))
    
    merges = []
    available_tokens = base_chars.copy()
    
    for token, token_id in compound_tokens:
        best_split = find_optimal_split(token, available_tokens)
        
        if best_split:
            left, right = best_split
            merges.append((left, right))
            available_tokens.add(token)
        else:
            if len(token) >= 2:
                merges.append((token[0], token[1:]))
                available_tokens.add(token)
    
    return merges

def find_optimal_split(token, available_tokens):
    best_split = None
    best_score = -1
    
    for i in range(1, len(token)):
        left = token[:i]
        right = token[i:]
        
        score = 0
        if left in available_tokens:
            score += len(left)
        if right in available_tokens:
            score += len(right)
        
        if left in available_tokens and right in available_tokens:
            score += 10
        
        if score > best_score:
            best_score = score
            best_split = (left, right)
    
    if best_split and best_split[0] in available_tokens:
        return best_split
    
    return None

def filter_existing_tokens(tokenizer, existing_vocab, target_vocab_size, special_tokens):
    current_vocab = tokenizer.get_vocab()
    filtered_vocab = {}
    token_count = 0
    
    # Add special tokens first
    for token in special_tokens:
        if token in current_vocab:
            filtered_vocab[token] = token_count
            token_count += 1
    
    # Add non-existing tokens up to target vocab size
    for token, old_id in sorted(current_vocab.items(), key=lambda x: x[1]):
        if token not in existing_vocab and token not in special_tokens:
            if token_count < target_vocab_size:
                filtered_vocab[token] = token_count
                token_count += 1
            else:
                break
    
    # Create merges that only reference tokens in filtered_vocab
    merges = derive_merges_from_filtered_vocab(filtered_vocab)
    
    new_tokenizer = Tokenizer(BPE(unk_token="[UNK]", vocab=filtered_vocab, merges=merges))
    new_tokenizer.normalizer = normalizers.NFKC()
    new_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    return new_tokenizer

def derive_merges_from_filtered_vocab(vocab_dict):
    """Derive merges ensuring all referenced tokens exist in vocab_dict"""
    # Get tokens sorted by their IDs
    tokens_by_id = sorted(vocab_dict.items(), key=lambda x: x[1])
    
    # Separate base characters and compound tokens
    base_chars = set()
    compound_tokens = []
    
    for token, token_id in tokens_by_id:
        # Skip special tokens
        if token.startswith('[') and token.endswith(']'):
            continue
        elif len(token) == 1:
            base_chars.add(token)
        elif len(token) > 1:
            compound_tokens.append((token, token_id))
    
    merges = []
    available_tokens = base_chars.copy()
    
    # Process compound tokens in order of their IDs
    for token, token_id in compound_tokens:
        best_split = find_valid_split(token, available_tokens, vocab_dict)
        
        if best_split:
            left, right = best_split
            # Only add merge if both parts are in our vocabulary
            if left in vocab_dict and right in vocab_dict:
                merges.append((left, right))
                available_tokens.add(token)
        
        # Always add the current token to available tokens for future merges
        available_tokens.add(token)
    
    return merges

def find_valid_split(token, available_tokens, vocab_dict):
    """Find optimal split ensuring both parts exist in vocab_dict"""
    best_split = None
    best_score = -1
    
    for i in range(1, len(token)):
        left = token[:i]
        right = token[i:]
        
        # Both parts must be in our filtered vocabulary
        if left not in vocab_dict or right not in vocab_dict:
            continue
            
        score = 0
        if left in available_tokens:
            score += len(left)
        if right in available_tokens:
            score += len(right)
        
        # Bonus for both being available
        if left in available_tokens and right in available_tokens:
            score += 10
        
        if score > best_score:
            best_score = score
            best_split = (left, right)
    
    return best_split

# def test_tokenizer(tokenizer_path: str):
#     tokenizer = Tokenizer.from_file(tokenizer_path)
    
#     with open("emilia_ja_text_preprocessed.txt", "r", encoding="utf-8") as f:
#         sample_lines = [f.readline().strip() for _ in range(5)]
    
#     for i, text in enumerate(sample_lines, 1):
#         if text and len(text) < 100:
#             try:
#                 encoding = tokenizer.encode(text)
#                 decoded = tokenizer.decode(encoding.ids)
#                 unk_count = encoding.tokens.count('[UNK]')
#             except Exception:
#                 pass

def analyze_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    
    special_count = 0
    japanese_count = 0
    other_count = 0
    
    for token, id in vocab.items():
        if token.startswith('[') and token.endswith(']'):
            special_count += 1
        elif any(ord(char) > 127 for char in token):
            japanese_count += 1
        else:
            other_count += 1
    
    return len(vocab), special_count, japanese_count, other_count

def main():
    import os
    text_file = "vietnamese_text_corpus.txt"
    vocab_size = 500
    output_path = "tokenizer_vietnamese_new.json"
    existing_tokenizer_path = "chatterbox-project/chatterbox_weights/tokenizer.json"
    
    if not os.path.exists(text_file):
        return
    
    if not os.path.exists(existing_tokenizer_path):
        existing_tokenizer_path = None
    
    tokenizer_path = create_japanese_tokenizer(text_file, vocab_size, output_path, existing_tokenizer_path)
    analyze_tokenizer(tokenizer_path)
    # test_tokenizer(tokenizer_path)
    
    if existing_tokenizer_path:
        compare_tokenizers(existing_tokenizer_path, tokenizer_path)

def derive_merges_for_any_tokenizer(tokenizer_path: str, output_path: str = None):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    
    try:
        existing_merges = tokenizer.model.get_merges() if hasattr(tokenizer.model, 'get_merges') else []
        if existing_merges:
            return tokenizer
    except:
        pass
    
    merges = derive_merges_from_vocab(vocab)
    new_tokenizer = Tokenizer(BPE(unk_token="[UNK]", vocab=vocab, merges=merges))
    
    if hasattr(tokenizer, 'normalizer') and tokenizer.normalizer:
        new_tokenizer.normalizer = tokenizer.normalizer
    if hasattr(tokenizer, 'pre_tokenizer') and tokenizer.pre_tokenizer:
        new_tokenizer.pre_tokenizer = tokenizer.pre_tokenizer
    if hasattr(tokenizer, 'post_processor') and tokenizer.post_processor:
        new_tokenizer.post_processor = tokenizer.post_processor
        
    if output_path:
        new_tokenizer.save(output_path)
    
    return new_tokenizer

def compare_tokenizers(existing_path: str, new_path: str):
    try:
        existing_vocab = load_existing_vocab(existing_path)
        new_tokenizer = Tokenizer.from_file(new_path)
        new_vocab = set(new_tokenizer.get_vocab().keys())
        overlapping_tokens = existing_vocab.intersection(new_vocab)
        return len(existing_vocab), len(new_vocab), len(overlapping_tokens)
    except Exception:
        return 0, 0, 0

if __name__ == "__main__":
    main()