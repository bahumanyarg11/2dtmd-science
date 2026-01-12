#!/bin/bash

echo "=================================================="
echo "   🚀 ANTIGRAVITY | T4 HEADLESS LAUNCHER          "
echo "=================================================="

# 0. System Setup (Headless 3D)
echo "[0/3] Setting up Graphics Drivers (XVFB)..."
sudo apt-get update -qq && sudo apt-get install -y libgl1-mesa-glx xvfb -qq > /dev/null 2>&1

# 1. Install Dependencies (Quietly)
echo "[1/3] Installing Scientific Dependencies..."
pip install -q -r requirements.txt

# 2. Run Training in Background (Long Job)
# 50,000 epochs on T4 approx 20-30 mins
echo "[2/3] Launching Scientific Training Job (Arrhenius Physics)..."
echo "      > Logs: training.log"
nohup python3 dtmd/core_ai/train_phase2.py --epochs 50000 > training.log 2>&1 &
TRAIN_PID=$!
echo "      > PID: $TRAIN_PID"

# 3. Launch UI in Background (Headless Mode)
echo "[3/3] Launching 2DTMD Console (Scientific)..."
echo "      > UI Logs: ui.log"
nohup xvfb-run -a python3 -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 > ui.log 2>&1 &
UI_PID=$!
echo "      > PID: $UI_PID"

echo "=================================================="
echo "✅ Systems Active."
echo "   - Training will continue even if you disconnect."
echo "   - Monitor training progress: 'tail -f training.log'"
echo "   - Access UI at the provided URL."
echo "=================================================="
