"""
Script to check checkpoint structure
"""
from safetensors.torch import load_file
from pathlib import Path
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/vietnamese/checkpoint-45000/pytorch_model.safetensors"

print(f"Loading checkpoint: {checkpoint_path}")
state_dict = load_file(checkpoint_path)

print(f"\nTotal keys: {len(state_dict)}")
print("\nFirst 20 keys:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"  {i+1}. {key}")

print("\nLast 10 keys:")
for i, key in enumerate(list(state_dict.keys())[-10:]):
    print(f"  {i+1}. {key}")

# Check for different prefixes
prefixes = set()
for key in state_dict.keys():
    prefix = key.split('.')[0]
    prefixes.add(prefix)

print(f"\nFound prefixes: {prefixes}")

