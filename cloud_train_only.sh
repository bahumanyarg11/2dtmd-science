#!/bin/bash
# CLOUD TRAINING SCRIPT (Lightning AI)
# Usage: ./cloud_train_only.sh

echo "=================================================="
echo "   ☁️ ANTIGRAVITY | CLOUD TRAINING ENGINE        "
echo "=================================================="

# 1. Install Scientific Dependencies
echo "[1/2] Installing Scientific Stack..."
pip install -q -r requirements.txt

# 2. Run Training (Saves 'dtmd_brain.pt')
echo "[2/2] Launching Training Job (50,000 Epochs)..."
# Using nohup so it survives disconnects
export PYTHONPATH=$PYTHONPATH:$(pwd)
nohup python3 dtmd/core_ai/train_phase2.py --epochs 50000 > training.log 2>&1 &
PID=$!

echo "=================================================="
echo "✅ Training Started (PID: $PID)"
echo "   - Monitor: 'tail -f training.log'"
echo "   - When finished, download: 'dtmd_brain.pt'"
echo "=================================================="
