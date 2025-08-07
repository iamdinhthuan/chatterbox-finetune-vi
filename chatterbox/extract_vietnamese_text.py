#!/usr/bin/env python3
"""
Extract Vietnamese text from CSV files to create text corpus for tokenizer training.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_vietnamese_text(text: str) -> str:
    """Clean and normalize Vietnamese text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters but keep Vietnamese diacritics
    # Keep letters, numbers, spaces, and common punctuation
    text = re.sub(
        r'[^\w\s\.,!?;:\-\(\)\"\'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]',
        ' ', text)

    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text.strip())

    return text


def extract_text_from_csv(csv_file: str, output_file: str, max_samples: int = None) -> bool:
    """Extract text from CSV file and save to text file"""
    try:
        logger.info(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)

        if 'transcript' not in df.columns:
            logger.error(f"'transcript' column not found in {csv_file}")
            return False

        logger.info(f"Found {len(df)} rows in CSV")

        # Limit samples if specified
        if max_samples and len(df) > max_samples:
            df = df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples")

        # Extract and clean text
        texts = []
        valid_count = 0

        for idx, row in df.iterrows():
            text = clean_vietnamese_text(row['transcript'])
            if text and len(text.strip()) > 0:
                texts.append(text)
                valid_count += 1

        logger.info(f"Extracted {valid_count} valid text samples")

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        logger.info(f"Saved text corpus to: {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error processing {csv_file}: {e}")
        return False


def combine_csv_files(csv_files: list, output_file: str, max_samples_per_file: int = None) -> bool:
    """Combine text from multiple CSV files"""
    all_texts = []

    for csv_file in csv_files:
        if not Path(csv_file).exists():
            logger.warning(f"CSV file not found: {csv_file}")
            continue

        try:
            logger.info(f"Processing: {csv_file}")
            df = pd.read_csv(csv_file)

            if 'transcript' not in df.columns:
                logger.warning(f"'transcript' column not found in {csv_file}, skipping")
                continue

            # Limit samples if specified
            if max_samples_per_file and len(df) > max_samples_per_file:
                df = df.head(max_samples_per_file)

            # Extract text
            for idx, row in df.iterrows():
                text = clean_vietnamese_text(row['transcript'])
                if text and len(text.strip()) > 0:
                    all_texts.append(text)

            logger.info(f"Extracted {len(all_texts)} texts from {csv_file}")

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue

    if not all_texts:
        logger.error("No valid texts extracted from any CSV file")
        return False

    # Save combined text
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')

    logger.info(f"Saved {len(all_texts)} texts to: {output_file}")
    return True


def analyze_text_corpus(text_file: str):
    """Analyze the text corpus"""
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        total_chars = sum(len(line) for line in lines)
        total_words = sum(len(line.split()) for line in lines)

        # Character frequency analysis
        char_freq = {}
        for line in lines:
            for char in line:
                if char.isalpha():  # Only count letters
                    char_freq[char] = char_freq.get(char, 0) + 1

        # Most common characters
        common_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        logger.info(f"Text corpus analysis:")
        logger.info(f"  - Total lines: {total_lines:,}")
        logger.info(f"  - Total characters: {total_chars:,}")
        logger.info(f"  - Total words: {total_words:,}")
        logger.info(f"  - Average line length: {total_chars / total_lines:.1f} chars")
        logger.info(f"  - Average words per line: {total_words / total_lines:.1f}")
        logger.info(f"  - Unique characters: {len(char_freq)}")

        logger.info("Most common characters:")
        for char, freq in common_chars[:10]:
            logger.info(f"    '{char}': {freq:,}")

        return True

    except Exception as e:
        logger.error(f"Error analyzing text corpus: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract Vietnamese text for tokenizer training")
    parser.add_argument("--train_csv", default="train.csv", help="Training CSV file")
    parser.add_argument("--val_csv", default="val.csv", help="Validation CSV file")
    parser.add_argument("--output", default="vietnamese_text_corpus.txt", help="Output text file")
    parser.add_argument("--max_samples", type=int, help="Maximum samples per CSV file")
    parser.add_argument("--analyze", action="store_true", help="Analyze the generated corpus")
    parser.add_argument("--combine_only", action="store_true", help="Only combine existing files")

    args = parser.parse_args()

    if args.combine_only:
        # Combine existing CSV files
        csv_files = []
        if Path(args.train_csv).exists():
            csv_files.append(args.train_csv)
        if Path(args.val_csv).exists():
            csv_files.append(args.val_csv)

        if not csv_files:
            logger.error("No CSV files found to combine")
            return

        success = combine_csv_files(csv_files, args.output, args.max_samples)
    else:
        # Process single file or combine multiple
        csv_files = [args.train_csv]
        if Path(args.val_csv).exists():
            csv_files.append(args.val_csv)

        success = combine_csv_files(csv_files, args.output, args.max_samples)

    if success and args.analyze:
        analyze_text_corpus(args.output)

    if success:
        logger.info(f"✅ Text extraction completed!")
        logger.info(f"📁 Output file: {args.output}")
        logger.info(f"🚀 Ready for tokenizer training!")
    else:
        logger.error("❌ Text extraction failed")


if __name__ == "__main__":
    main()
