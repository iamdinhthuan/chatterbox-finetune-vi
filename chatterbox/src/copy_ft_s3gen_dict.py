#!/usr/bin/env python3
"""
Merge a fine-tuned S3Gen flow checkpoint into the original full S3Gen safetensors.
"""
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(
        description="Merge fine-tuned S3Gen flow weights into a full S3Gen safetensors file"
    )
    parser.add_argument(
        "-o", "--original",
        type=Path,
        required=True,
        help="Path to the original full s3gen.safetensors"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        required=True,
        help="Path to the fine-tuned flow-only checkpoint safetensors (e.g. model.safetensors)"
    )
    parser.add_argument(
        "-O", "--output",
        type=Path,
        required=True,
        help="Path where the merged s3gen.safetensors will be written"
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load tensors
    print(f"Loading original weights from {args.original}...")
    orig_dict = load_file(str(args.original))
    print(f"Loading fine-tuned checkpoint from {args.checkpoint}...")
    ft_dict = load_file(str(args.checkpoint))

    # Determine prefix in fine-tuned checkpoint (e.g. 'flow_model')
    prefixes = set(key.split('.', 1)[0] for key in ft_dict.keys())
    if len(prefixes) == 1:
        ft_prefix = prefixes.pop()
    elif 'flow_model' in prefixes:
        ft_prefix = 'flow_model'
    else:
        ft_prefix = prefixes.pop()
        print(f"Warning: multiple prefixes detected {prefixes}. Using '{ft_prefix}'.")

    # Merge: take all checkpoint keys under ft_prefix and remap 'ft_prefix.' -> 'flow.'
    merged = dict(orig_dict)
    replaced = 0
    for key, tensor in ft_dict.items():
        if key.startswith(ft_prefix + '.'):
            new_key = 'flow.' + key[len(ft_prefix) + 1:]
            merged[new_key] = tensor
            replaced += 1

    print(f"Replaced {replaced} flow weights in the original model.")

    # Save merged safetensors
    out_path = str(args.output)
    print(f"Saving merged weights to {out_path}...")
    save_file(merged, out_path)
    print("Done.")


if __name__ == '__main__':
    main()
