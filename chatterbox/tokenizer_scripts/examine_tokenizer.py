#!/usr/bin/env python3
"""Examine the existing tokenizer structure"""

from tokenizers import Tokenizer

def examine_tokenizer(tokenizer_path):
    t = Tokenizer.from_file(tokenizer_path)
    
    print(f"Vocab size: {t.get_vocab_size()}")
    print(f"Model type: {type(t.model)}")
    
    vocab = t.get_vocab()
    print(f"\nFirst 20 vocab items:")
    for i, (token, id) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:20]):
        print(f"  {id}: '{token}'")
    
    print(f"\nLast 10 vocab items:")
    for i, (token, id) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[-10:]):
        print(f"  {id}: '{token}'")
    
    # Check for special tokens
    special_tokens = []
    for token, id in vocab.items():
        if token.startswith('[') and token.endswith(']'):
            special_tokens.append((id, token))
    
    print(f"\nSpecial tokens ({len(special_tokens)}):")
    for id, token in sorted(special_tokens):
        print(f"  {id}: '{token}'")

if __name__ == "__main__":
    examine_tokenizer("chatterbox-project/chatterbox_weights/tokenizer.json")