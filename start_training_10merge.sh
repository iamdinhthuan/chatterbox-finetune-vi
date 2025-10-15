#!/bin/bash
# Complete script to switch to 10-merge BPE and start training

set -e  # Exit on error

echo "============================================================"
echo "SWITCHING TO 10-MERGE BPE AND STARTING TRAINING"
echo "============================================================"
echo ""

# Step 1: Pull latest code
echo "Step 1: Pulling latest code..."
git pull origin main
echo "✅ Code updated"
echo ""

# Step 2: Verify NaN fix
echo "Step 2: Verifying NaN fix..."
if grep -q "labels_text" src/chatterbox/utils/preprocessed_dataset.py; then
    echo "✅ NaN fix present"
else
    echo "❌ ERROR: NaN fix NOT found!"
    echo "This is CRITICAL - training will have 48% NaN without it!"
    echo ""
    echo "Please run: git pull origin main"
    echo "Or check: git log --oneline -5"
    exit 1
fi
echo ""

# Step 3: Switch to 10-merge tokenizer
echo "Step 3: Switching to 10-merge tokenizer..."
bash switch_to_char_tokenizer.sh
echo ""

# Step 4: Verify switch
echo "Step 4: Verifying tokenizer switch..."
python3 -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('VietnameseTokenizer/tokenizer.json')
enc = tok.encode('Xin chào các bạn')
if len(enc.ids) > 10:
    print(f'✅ 10-merge tokenizer active ({len(enc.ids)} tokens)')
else:
    print(f'❌ Still using 465-merge tokenizer ({len(enc.ids)} tokens)')
    exit(1)
"
echo ""

# Step 5: Backup old checkpoints
echo "Step 5: Backing up old checkpoints..."
if [ -d "checkpoints/vietnamese" ]; then
    if [ ! -d "checkpoints/vietnamese_465merge_backup" ]; then
        cp -r checkpoints/vietnamese checkpoints/vietnamese_465merge_backup
        echo "✅ Old checkpoints backed up to checkpoints/vietnamese_465merge_backup"
    else
        echo "⚠️ Backup already exists, skipping"
    fi
else
    echo "ℹ️ No old checkpoints to backup"
fi
echo ""

# Step 6: Start training
echo "Step 6: Starting training with 10-merge tokenizer..."
echo ""
echo "============================================================"
echo "TRAINING CONFIGURATION"
echo "============================================================"
echo "Tokenizer: 10-merge BPE (character-level)"
echo "Dataset: preprocessed_data (2.6M samples)"
echo "Epochs: 3"
echo "Batch size: 8"
echo "Learning rate: 1e-5"
echo "Output: checkpoints/vietnamese_10merge"
echo ""
echo "EXPECTED RESULTS:"
echo "  Epoch 0.15 (~3 hours): Vietnamese starting clear ✅"
echo "  Epoch 0.30 (~6 hours): Vietnamese clear ✅"
echo "  Epoch 0.50 (~10 hours): Good quality ✅"
echo "  Epoch 1.00 (~20 hours): Good quality ✅"
echo "  Epoch 3.00 (~2.5 days): Max quality for this tokenizer ✅"
echo "============================================================"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

python3 train.py \
  --csv metadata.csv \
  --use_preprocessed \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-5 \
  --output_dir checkpoints/vietnamese_10merge \
  --save_steps 5000 \
  --eval_steps 5000 \
  --logging_steps 100

echo ""
echo "============================================================"
echo "✅ TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Model saved at: checkpoints/vietnamese_10merge"
echo ""
echo "To test inference:"
echo "  python infer.py --checkpoint checkpoints/vietnamese_10merge --text 'Xin chào'"
echo ""
echo "To switch back to 465-merge tokenizer:"
echo "  cp VietnameseTokenizer/tokenizer_bpe_backup.json VietnameseTokenizer/tokenizer.json"
echo "  python train.py --csv metadata.csv --use_preprocessed --epochs 10 --resume_from_checkpoint checkpoints/vietnamese_465merge_backup/checkpoint-XXXX"
