"""
Quick script to check metadata.csv format
"""
import csv
from pathlib import Path

csv_path = "metadata.csv"
audio_dir = Path(".")

print("="*80)
print("CHECKING METADATA.CSV")
print("="*80)

# Check file exists
if not Path(csv_path).exists():
    print(f"❌ File not found: {csv_path}")
    exit(1)

print(f"✅ File exists: {csv_path}")

# Read first few lines
print(f"\n📖 First 5 lines of file:")
with open(csv_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  {i+1}: {line.rstrip()}")

# Try different delimiters
print(f"\n🔍 Testing different delimiters:")

for delimiter in ['|', ',', '\t', ';']:
    print(f"\n  Trying delimiter: '{delimiter}'")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames
            print(f"    Headers: {headers}")
            
            # Try to read first row
            try:
                first_row = next(reader)
                print(f"    First row keys: {list(first_row.keys())}")
                
                # Check for audio/transcript columns
                has_audio = any(k in first_row for k in ['audio', 'audio_path', 'file', 'path'])
                has_transcript = any(k in first_row for k in ['transcript', 'text', 'sentence'])
                
                if has_audio and has_transcript:
                    print(f"    ✅ Found audio and transcript columns!")
                    
                    # Check if audio file exists
                    for key in ['audio', 'audio_path', 'file', 'path']:
                        if key in first_row:
                            audio_file = first_row[key]
                            audio_path = audio_dir / audio_file
                            print(f"    Audio column: '{key}' = '{audio_file}'")
                            print(f"    Full path: {audio_path}")
                            print(f"    Exists: {audio_path.exists()}")
                            break
                    
                    for key in ['transcript', 'text', 'sentence']:
                        if key in first_row:
                            text_value = first_row[key][:50]
                            print(f"    Text column: '{key}' = '{text_value}...'")
                            break
                            
            except StopIteration:
                print(f"    ⚠️  File has headers but no data rows")
                
    except Exception as e:
        print(f"    ❌ Error: {e}")

# Count total lines
with open(csv_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

print(f"\n📊 Total lines in file: {total_lines}")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

print("""
Expected format for preprocess_dataset.py:

metadata.csv:
```
audio|transcript
audio_001.wav|Xin chào các bạn
audio_002.wav|Hôm nay trời đẹp
```

Key points:
1. Delimiter: | (pipe)
2. Header row: 'audio' and 'transcript'
3. No spaces around delimiter
4. Audio files must exist in audio_dir

If your format is different, you may need to:
- Convert CSV to correct format
- Or modify the script to match your format
""")
