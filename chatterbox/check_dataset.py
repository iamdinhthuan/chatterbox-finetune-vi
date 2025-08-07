#!/usr/bin/env python3
"""
Dataset Validation Script for Vietnamese TTS Training
This script checks the CSV dataset and audio files before training.
"""

import os
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse

def check_audio_file(audio_path):
    """Check if audio file is valid and get its duration"""
    try:
        if not os.path.exists(audio_path):
            return False, 0, "File not found"
        
        # Try to load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        if duration < 0.5:  # Too short
            return False, duration, "Too short (< 0.5s)"
        elif duration > 30:  # Too long
            return False, duration, "Too long (> 30s)"
        
        return True, duration, "OK"
    except Exception as e:
        return False, 0, f"Error: {str(e)}"

def validate_csv_dataset(csv_file, max_samples=None):
    """Validate CSV dataset"""
    print(f"\n=== Validating {csv_file} ===")
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ CSV loaded successfully: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Check columns
    required_columns = ['audio', 'transcript']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"✅ Required columns found: {required_columns}")
    
    # Check for missing data
    missing_audio = df['audio'].isna().sum()
    missing_transcript = df['transcript'].isna().sum()
    empty_transcript = (df['transcript'].str.strip() == '').sum()
    
    print(f"📊 Data quality:")
    print(f"   - Missing audio paths: {missing_audio}")
    print(f"   - Missing transcripts: {missing_transcript}")
    print(f"   - Empty transcripts: {empty_transcript}")
    
    # Sample validation
    valid_samples = 0
    invalid_samples = 0
    total_duration = 0
    durations = []
    
    # Limit samples for quick check
    samples_to_check = min(len(df), max_samples) if max_samples else len(df)
    print(f"\n🔍 Checking {samples_to_check} audio files...")
    
    for idx in tqdm(range(samples_to_check), desc="Validating audio files"):
        row = df.iloc[idx]
        audio_path = row['audio']
        transcript = row['transcript']
        
        # Skip if missing data
        if pd.isna(audio_path) or pd.isna(transcript) or str(transcript).strip() == '':
            invalid_samples += 1
            continue
        
        # Check audio file
        is_valid, duration, message = check_audio_file(audio_path)
        
        if is_valid:
            valid_samples += 1
            total_duration += duration
            durations.append(duration)
        else:
            invalid_samples += 1
            if idx < 10:  # Show first 10 errors
                print(f"   ❌ {audio_path}: {message}")
    
    # Statistics
    print(f"\n📈 Validation Results:")
    print(f"   - Valid samples: {valid_samples}")
    print(f"   - Invalid samples: {invalid_samples}")
    print(f"   - Success rate: {valid_samples/(valid_samples+invalid_samples)*100:.1f}%")
    
    if durations:
        print(f"   - Total duration: {total_duration/3600:.2f} hours")
        print(f"   - Average duration: {sum(durations)/len(durations):.2f}s")
        print(f"   - Min duration: {min(durations):.2f}s")
        print(f"   - Max duration: {max(durations):.2f}s")
    
    # Text statistics
    if valid_samples > 0:
        valid_df = df.iloc[:samples_to_check].dropna(subset=['transcript'])
        text_lengths = valid_df['transcript'].str.len()
        print(f"   - Average text length: {text_lengths.mean():.1f} characters")
        print(f"   - Min text length: {text_lengths.min()} characters")
        print(f"   - Max text length: {text_lengths.max()} characters")
    
    return valid_samples > 0

def main():
    parser = argparse.ArgumentParser(description="Validate CSV dataset for TTS training")
    parser.add_argument("--train_csv", type=str, default="train.csv",
                       help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="val.csv",
                       help="Path to validation CSV file")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to check (for quick validation)")
    parser.add_argument("--full_check", action="store_true",
                       help="Check all samples (may take a long time)")
    
    args = parser.parse_args()
    
    print("🎵 Vietnamese TTS Dataset Validator")
    print("=" * 50)
    
    max_samples = None if args.full_check else args.max_samples
    
    # Validate training set
    train_valid = validate_csv_dataset(args.train_csv, max_samples)
    
    # Validate validation set
    val_valid = True
    if os.path.exists(args.val_csv):
        val_valid = validate_csv_dataset(args.val_csv, max_samples)
    else:
        print(f"\n⚠️  Validation CSV not found: {args.val_csv}")
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 SUMMARY:")
    print(f"   - Training dataset: {'✅ Valid' if train_valid else '❌ Invalid'}")
    print(f"   - Validation dataset: {'✅ Valid' if val_valid else '❌ Invalid'}")
    
    if train_valid:
        print(f"\n🚀 Dataset is ready for training!")
        print(f"   Run: python train_vietnamese_csv.py --train_csv {args.train_csv} --val_csv {args.val_csv}")
    else:
        print(f"\n❌ Please fix dataset issues before training.")
    
    return train_valid and val_valid

if __name__ == "__main__":
    main()
