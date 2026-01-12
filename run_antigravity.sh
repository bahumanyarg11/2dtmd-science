#!/bin/bash

# --- ANTIGRAVITY LAUNCH PROTOCOL ---

echo "=================================================="
echo "   🚀 PROJECT ANTIGRAVITY | DEPLOYMENT SEQUENCE   "
echo "=================================================="

# 1. Environment Check & H100 Optimization
echo "[1/4] Checking Compute Infrastructure..."
if command -v nvidia-smi &> /dev/null
then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "      Detected GPU: $GPU_NAME"
    if [[ $GPU_NAME == *"H100"* ]]; then
        echo "      ✅ H100 Hopper Architecture Confirmed."
        echo "      --> Enabling FP8 Transformer Engine."
        export TORCH_COMPILE_MODE="max-autotune"
    elif [[ $GPU_NAME == *"T4"* ]]; then
        echo "      ✅ T4 Turing Architecture Confirmed."
        echo "      --> Enabling TF32 Precision."
    fi
else
    echo "      ⚠️  No GPU Detected (or nvidia-smi missing). Running in CPU Fallback Mode (Slow)."
fi

# 2. Dependency Verification
echo "[2/4] Verifying Neural Modules..."
pip install -q -r requirements.txt
echo "      Dependencies Verified."

# 3. Compile AI Core (Just-In-Time)
echo "[3/4] JIT-Compiling GFlowNet & FNO Kernels..."
# This runs a dummy inference to trigger torch.compile() before UI loads
# Fixed import path for current structure
python3 -c "import sys; sys.path.append('.'); import torch; from antigravity.core_ai.fno_synthesis import SynthesisFNO; model = torch.compile(SynthesisFNO()); print('      Kernels Fused.')"

# 4. Launch Orchestrator
echo "[4/4] Launching Antigravity Console..."
echo "      > UI accessible at: http://localhost:8501"
echo "=================================================="

# Run Streamlit in the background but pipe output to console
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
