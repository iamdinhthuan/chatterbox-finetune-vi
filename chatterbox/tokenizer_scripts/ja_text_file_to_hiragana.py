import argparse
import pykakasi
from tqdm import tqdm

def convert_to_hiragana(text):
    pka_converter = pykakasi.kakasi()
    pka_converter.setMode("J","H")  # Kanji to Hiragana    
    pka_converter.setMode("K","H")  # Katakana to Hiragana
    pka_converter.setMode("H","H")  # Hiragana stays Hiragana
    conv = pka_converter.getConverter()
    text = conv.do(text)
    return text

def main():
    parser = argparse.ArgumentParser(description="Convert text to hiragana")
    parser.add_argument("--input-file", default="emilia_ja_text_data.txt", type=str, help="Path to the input text file")
    parser.add_argument("--output-file", default="emilia_ja_text_data_hiragana.txt", type=str, help="Path to the output text file")
    args = parser.parse_args()
    
    # First pass to count total lines for progress bar
    with open(args.input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    # Process file line by line with progress bar
    with open(args.input_file, "r", encoding="utf-8") as input_file, \
         open(args.output_file, "w", encoding="utf-8") as output_file:
        
        for line in tqdm(input_file, total=total_lines, desc="Converting to hiragana"):
            hiragana_line = convert_to_hiragana(line)
            output_file.write(hiragana_line)
    
    print(f"Successfully converted {args.input_file} to hiragana and saved to {args.output_file}")

if __name__ == "__main__":
    main()
        