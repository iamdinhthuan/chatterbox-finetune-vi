#!/usr/bin/env python3

import json
from typing import List, Tuple

def load_merges_from_file(file_path: str) -> List[List[str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'model' in data and 'merges' in data['model']:
            return data['model']['merges']
        else:
            return []
            
    except Exception:
        return []

def normalize_merge(merge) -> Tuple[str, str]:
    if isinstance(merge, list) and len(merge) >= 2:
        return (merge[0], merge[1])
    elif isinstance(merge, tuple) and len(merge) >= 2:
        return (merge[0], merge[1])
    elif isinstance(merge, str):
        parts = merge.split()
        if len(parts) >= 2:
            return (parts[0], parts[1])
        else:
            raise ValueError(f"Invalid string merge format: {merge}")
    else:
        raise ValueError(f"Invalid merge format: {merge}")

def compare_merge_lists(merges1: List, merges2: List):
    set1 = set()
    set2 = set()
    
    try:
        for merge in merges1:
            set1.add(normalize_merge(merge))
    except Exception:
        return 0
    
    try:
        for merge in merges2:
            set2.add(normalize_merge(merge))
    except Exception:
        return 0
    
    common = set1 & set2
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    elif len(set1) == 0 or len(set2) == 0:
        return 0.0
    else:
        return len(common) / len(set1 | set2)

def compare_merge_order_with_stats(merges1: List, merges2: List):
    try:
        norm1 = [normalize_merge(m) for m in merges1]
        norm2 = [normalize_merge(m) for m in merges2]
    except Exception:
        return 0, 0
    
    min_len = min(len(norm1), len(norm2))
    matches = 0
    
    for i in range(min_len):
        if norm1[i] == norm2[i]:
            matches += 1
    
    return matches, min_len

def main():
    import sys
    
    original_file = "multi_1024_tokenizer.json"
    derived_file = "test_with_merges.json"
    
    if len(sys.argv) >= 3:
        original_file = sys.argv[1]
        derived_file = sys.argv[2]
    elif len(sys.argv) == 2:
        derived_file = sys.argv[1]
    
    original_merges = load_merges_from_file(original_file)
    derived_merges = load_merges_from_file(derived_file)
    
    if not original_merges and not derived_merges:
        return
    
    similarity = compare_merge_lists(original_merges, derived_merges)
    order_matches, min_length = compare_merge_order_with_stats(original_merges, derived_merges)
    
    return similarity, order_matches, min_length

if __name__ == "__main__":
    main()