#!/bin/bash
# LOCAL MAC LAUNCHER
# Usage: ./mac_run_ui.sh

echo "=================================================="
echo "   🍎 ANTIGRAVITY | MAC VISUALIZATION ENGINE     "
echo "=================================================="

# 1. Check for Brain
if [ -f "dtmd_brain.pt" ]; then
    echo "✅ Trained Brain Found! Loading Scientific Model..."
else
    echo "⚠️  No 'dtmd_brain.pt' found."
    echo "   -> Running in SIMULATION MODE (Random Weights)"
    echo "   -> To fix: Download 'dtmd_brain.pt' from Lightning AI to this folder."
fi

# 2. Install Dependencies (Mac optimized)
echo "[1/2] Checking Dependencies..."
# Ensure we have plotting libs
pip install -q streamlit pyvista stpyvista plotly pandas

# 3. Launch UI
echo "[2/2] Launching Dashboard..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYVISTA_OFF_SCREEN=true
streamlit run frontend/app.py --server.enableCORS=false --server.enableXsrfProtection=false
