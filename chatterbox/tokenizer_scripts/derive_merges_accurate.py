#!/usr/bin/env python3

import json
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokenizer_data(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_merge_priority_map(vocab: Dict[str, int]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Build merge map following tokenizers library logic.
    Returns map of (token_id1, token_id2) -> (rank, new_token_id)
    """
    merge_map = {}
    tokens_by_id = {v: k for k, v in vocab.items()}
    
    logger.info(f"Building merge priority map from {len(vocab)} tokens")
    
    # Find all potential merges by analyzing compound tokens
    for token_id in sorted(tokens_by_id.keys()):
        token = tokens_by_id[token_id]
        
        # Skip special tokens and single characters
        if token.startswith('[') and token.endswith(']'):
            continue
        if len(token) <= 1:
            continue
            
        # Find the best split for this token
        best_split = find_library_split(token, vocab, tokens_by_id, token_id)
        if best_split:
            left_token, right_token = best_split
            left_id = vocab[left_token]
            right_id = vocab[right_token]
            
            # Use token_id as rank (higher ID = later merge = higher priority)
            rank = token_id
            merge_map[(left_id, right_id)] = (rank, token_id)
            
    logger.info(f"Created {len(merge_map)} merge rules")
    return merge_map

def find_library_split(target_token: str, vocab: Dict[str, int], tokens_by_id: Dict[int, str], target_id: int) -> Tuple[str, str]:
    """
    Find the split that the tokenizers library would have made.
    This follows the BPE merge construction logic from model.rs lines 168-186.
    """
    candidates = []
    
    # Try all possible splits
    for i in range(1, len(target_token)):
        left = target_token[:i]
        right = target_token[i:]
        
        # Both parts must exist in vocab with lower IDs than target
        if left in vocab and right in vocab:
            left_id = vocab[left]
            right_id = vocab[right]
            
            # Only consider if both parts have lower IDs (were created before target)
            if left_id < target_id and right_id < target_id:
                # Score: prefer earlier created tokens (lower max_id), then longer left part
                max_component_id = max(left_id, right_id)
                score = (max_component_id, -len(left))
                candidates.append((score, left, right))
    
    if candidates:
        candidates.sort()
        return candidates[0][1], candidates[0][2]
    
    return None

class BPEMerger:
    """
    Accurate BPE merger following tokenizers library Word::merge_all logic.
    """
    
    def __init__(self, merge_map: Dict[Tuple[int, int], Tuple[int, int]]):
        self.merge_map = merge_map
    
    def merge_word(self, word_tokens: List[int]) -> List[int]:
        """
        Merge word tokens following the exact algorithm from word.rs merge_all.
        """
        if len(word_tokens) <= 1:
            return word_tokens
        
        # Build symbol list with prev/next pointers (simplified)
        symbols = [{'id': token_id, 'active': True} for token_id in word_tokens]
        
        # Create priority queue of possible merges
        queue = []
        
        # Find all possible initial merges
        for i in range(len(symbols) - 1):
            if symbols[i]['active'] and symbols[i + 1]['active']:
                pair = (symbols[i]['id'], symbols[i + 1]['id'])
                if pair in self.merge_map:
                    rank, new_id = self.merge_map[pair]
                    # Use negative rank for min-heap (lower rank = higher priority)
                    heapq.heappush(queue, (-rank, i, new_id, symbols[i]['id'], symbols[i + 1]['id']))
        
        # Process merges in priority order
        while queue:
            neg_rank, pos, new_id, expected_left, expected_right = heapq.heappop(queue)
            
            # Check if this merge is still valid
            if (pos >= len(symbols) or 
                not symbols[pos]['active'] or 
                pos + 1 >= len(symbols) or 
                not symbols[pos + 1]['active'] or
                symbols[pos]['id'] != expected_left or
                symbols[pos + 1]['id'] != expected_right):
                continue
            
            # Perform the merge
            symbols[pos]['id'] = new_id
            symbols[pos + 1]['active'] = False
            
            # Add new possible merges
            # Left neighbor
            if pos > 0 and symbols[pos - 1]['active']:
                left_pair = (symbols[pos - 1]['id'], new_id)
                if left_pair in self.merge_map:
                    rank, left_new_id = self.merge_map[left_pair]
                    heapq.heappush(queue, (-rank, pos - 1, left_new_id, symbols[pos - 1]['id'], new_id))
            
            # Right neighbor
            next_active = pos + 2
            while next_active < len(symbols) and not symbols[next_active]['active']:
                next_active += 1
            
            if next_active < len(symbols):
                right_pair = (new_id, symbols[next_active]['id'])
                if right_pair in self.merge_map:
                    rank, right_new_id = self.merge_map[right_pair]
                    heapq.heappush(queue, (-rank, pos, right_new_id, new_id, symbols[next_active]['id']))
        
        # Return only active symbols
        return [s['id'] for s in symbols if s['active']]

def derive_accurate_merges(vocab: Dict[str, int]) -> List[Tuple[str, str]]:
    """
    Derive merges that will produce identical tokenization to the library.
    """
    logger.info("Deriving accurate merges using library-compatible algorithm")
    
    # Build merge priority map
    merge_map = build_merge_priority_map(vocab)
    
    # Convert to merge list sorted by rank
    merges = []
    tokens_by_id = {v: k for k, v in vocab.items()}
    
    # Sort by rank (which is the target token ID)
    sorted_merges = sorted(merge_map.items(), key=lambda x: x[1][0])
    
    for (left_id, right_id), (rank, new_id) in sorted_merges:
        left_token = tokens_by_id[left_id]
        right_token = tokens_by_id[right_id]
        merges.append((left_token, right_token))
    
    logger.info(f"Generated {len(merges)} merges")
    return merges

def test_accuracy(original_file: str, no_merge_file: str):
    """
    Test the accuracy of derived merges against original tokenizer.
    """
    logger.info(f"Testing accuracy: {original_file} vs {no_merge_file}")
    
    # Load files
    original_data = load_tokenizer_data(original_file)
    no_merge_data = load_tokenizer_data(no_merge_file)
    
    assert original_data['model']['vocab'] == no_merge_data['model']['vocab'], "Vocabularies must match"
    
    vocab = original_data['model']['vocab']
    original_merges = original_data['model']['merges']
    
    # Derive merges
    derived_merges = derive_accurate_merges(vocab)
    
    # Save derived tokenizer
    no_merge_data['model']['merges'] = derived_merges
    output_file = no_merge_file.replace('_no_merge.json', '_derived_accurate.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(no_merge_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved derived tokenizer to {output_file}")
    
    # Compare merge counts
    logger.info(f"Original merges: {len(original_merges)}")
    logger.info(f"Derived merges: {len(derived_merges)}")
    
    # Compare actual merges
    original_set = set(tuple(m) if isinstance(m, list) else tuple(m.split()) for m in original_merges)
    derived_set = set(derived_merges)
    
    common = original_set & derived_set
    only_original = original_set - derived_set
    only_derived = derived_set - original_set
    
    accuracy = len(common) / len(original_set | derived_set) if original_set or derived_set else 1.0
    
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"Common merges: {len(common)}")
    logger.info(f"Only in original: {len(only_original)}")
    logger.info(f"Only in derived: {len(only_derived)}")
    
    if only_original:
        logger.info("Examples only in original:")
        for merge in list(only_original)[:5]:
            logger.info(f"  {merge}")
    
    if only_derived:
        logger.info("Examples only in derived:")
        for merge in list(only_derived)[:5]:
            logger.info(f"  {merge}")
    
    return accuracy, output_file

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python derive_merges_accurate.py <original_tokenizer> <no_merge_tokenizer>")
        sys.exit(1)
    
    original_file, no_merge_file = sys.argv[1:3]
    test_accuracy(original_file, no_merge_file)

if __name__ == "__main__":
    main()