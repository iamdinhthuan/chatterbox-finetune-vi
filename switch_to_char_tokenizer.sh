#!/bin/bash
# Switch from BPE to Char tokenizer

echo "================================"
echo "Switching to Char Tokenizer"
echo "================================"

# Backup BPE tokenizer
echo "1. Backing up BPE tokenizer..."
cp VietnameseTokenizer/tokenizer.json VietnameseTokenizer/tokenizer_bpe_backup.json
echo "   ✅ Backed up to: VietnameseTokenizer/tokenizer_bpe_backup.json"

# Copy char tokenizer
echo ""
echo "2. Installing char tokenizer..."
cp tokenizer_char.json VietnameseTokenizer/tokenizer.json
echo "   ✅ Installed char tokenizer"

# Verify
echo ""
echo "3. Verifying tokenizer..."
python3 -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('VietnameseTokenizer/tokenizer.json')
enc = tok.encode('Xin chào các bạn')
print(f'   Test text: \"Xin chào các bạn\"')
print(f'   Tokens: {len(enc.ids)} tokens')
print(f'   IDs: {enc.ids}')
if len(enc.ids) > 10:
    print('   ✅ Char tokenizer active!')
else:
    print('   ⚠️ This looks like BPE. Check if copy worked.')
"

echo ""
echo "================================"
echo "✅ DONE!"
echo "================================"
echo ""
echo "To switch back to BPE:"
echo "  cp VietnameseTokenizer/tokenizer_bpe_backup.json VietnameseTokenizer/tokenizer.json"
echo ""
echo "Now restart training:"
echo "  python train.py --csv metadata.csv --use_preprocessed --epochs 5 --batch_size 8"
