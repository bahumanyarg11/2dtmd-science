#!/bin/bash

# Force Headless Mode for PyVista/VTK
# This prevents "Connection Error" on cloud execution
echo "🔧 Setting up Headless 3D Environment..."
sudo apt-get update -qq && sudo apt-get install -y libgl1-mesa-glx xvfb -qq > /dev/null 2>&1

echo "🚀 Launching Antigravity Console (Headless Mode Active)..."
pkill -f streamlit
pip install -r requirements.txt -q

# Run with xvfb-run to simulate a monitor
xvfb-run -a python3 -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
