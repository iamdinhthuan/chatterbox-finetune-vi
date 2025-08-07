#!/usr/bin/env python3

import json
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokenizer_data(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_merge_format(merges: list) -> str:
    if not merges:
        return "string"
    
    first_merge = merges[0]
    if isinstance(first_merge, list):
        return "array"
    elif isinstance(first_merge, str):
        return "string"
    else:
        return "string"

def normalize_merge_to_format(merge, target_format: str):
    if target_format == "array":
        if isinstance(merge, str):
            return merge.split()
        elif isinstance(merge, list):
            return merge
    else:
        if isinstance(merge, list):
            return " ".join(merge)
        elif isinstance(merge, str):
            return merge
    return merge

def merge_tokenizers(tokenizer_a_path: str, tokenizer_b_path: str, output_path: str) -> str:
    logger.info(f"Merging {tokenizer_b_path} into {tokenizer_a_path}")
    
    data_a = load_tokenizer_data(tokenizer_a_path)
    data_b = load_tokenizer_data(tokenizer_b_path)
    
    assert 'model' in data_a and 'vocab' in data_a['model'], "Tokenizer A missing vocab"
    assert 'model' in data_b and 'vocab' in data_b['model'], "Tokenizer B missing vocab"
    
    vocab_a = data_a['model']['vocab']
    vocab_b = data_b['model']['vocab']
    merges_a = data_a['model'].get('merges', [])
    merges_b = data_b['model'].get('merges', [])
    
    logger.info(f"A: {len(vocab_a)} tokens, {len(merges_a)} merges")
    logger.info(f"B: {len(vocab_b)} tokens, {len(merges_b)} merges")
    
    merge_format_a = detect_merge_format(merges_a)
    logger.info(f"Tokenizer A merge format: {merge_format_a}")
    
    max_id_a = max(vocab_a.values()) if vocab_a else -1
    merged_vocab = vocab_a.copy()
    
    new_tokens = 0
    for token, _ in vocab_b.items():
        if token not in merged_vocab:
            merged_vocab[token] = max_id_a + 1 + new_tokens
            new_tokens += 1
    
    normalized_merges_b = [normalize_merge_to_format(merge, merge_format_a) for merge in merges_b]
    merged_merges = merges_a + normalized_merges_b
    
    logger.info(f"Merged: {len(merged_vocab)} tokens (+{new_tokens}), {len(merged_merges)} merges")
    
    data_a['model']['vocab'] = merged_vocab
    data_a['model']['merges'] = merged_merges
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_a, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved merged tokenizer to {output_path}")
    return output_path

def main():
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python merge_tokenizers.py <tokenizer_a> <tokenizer_b> <output>")
        sys.exit(1)
    
    tokenizer_a, tokenizer_b, output = sys.argv[1:4]
    merge_tokenizers(tokenizer_a, tokenizer_b, output)

if __name__ == "__main__":
    main()