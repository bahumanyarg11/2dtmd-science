import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys
import os
import argparse

# --- Scientific Imports ---
# Add root directory to path to import the data engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from antigravity_data import LiteratureSynthesisGenerator

# --- H100 Optimization Flags ---
torch.backends.cuda.matmul.allow_tf32 = True # Allow TF32 on Ampere/Hopper
torch.backends.cudnn.allow_tf32 = True

# --- FNO Architecture (Digital Twin) ---
from antigravity.core_ai.fno_synthesis import SynthesisFNO

# --- Scientific Training Loop ---
def train_scientific(epochs=50000):
    # Mac Optimization: Use MPS (Metal) if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"--> 🍎 Mac Metal Acceleration Enabled (MPS) on {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"--> Antigravity Science Engine Initializing on: {device}")
    
    # 1. Initialize The Scientific Data Generator (Arrhenius Physics)
    data_engine = LiteratureSynthesisGenerator()
    
    # 2. Initialize the Digital Twin (FNO)
    twin = SynthesisFNO(modes=12, width=32).to(device)
    
    optimizer = optim.AdamW(twin.parameters(), lr=0.002, weight_decay=1e-4) # Higher LR for fast convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"--> Scientific Training Loop Started for {epochs} epochs.")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        t0 = time.time()
        
        optimizer.zero_grad()
        
        # --- A. Data Ingestion (Physics-Based) ---
        # Inputs: [Temp, Pressure, Flow_Precursor, Flow_Carrier]
        # Targets: [Concentration] (Governed by Arrhenius Rate ~ exp(-Ea/RT))
        inputs, targets = data_engine.generate_cvd_batch(batch_size=16, device=device) # Larger batch
        
        # --- B. Forward Pass (Digital Twin) ---
        preds = twin(inputs)
        
        # --- C. Loss & Explainability ---
        # MSE Loss ensures the model learns the Kinetic Rate Law
        loss = F.mse_loss(preds, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        t1 = time.time()
        
        # --- Logging ---
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Kinetics Loss: {loss.item():.6f} | Time: {(t1-t0)*1000:.2f}ms")

    print("--> Training Complete.")
    
    # --- Save Checkpoint for Local Inference ---
    # Save to Project Root
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    params_path = os.path.join(root_path, "antigravity_brain.pt")
    
    torch.save(twin.state_dict(), params_path)
    print(f"--> 🧠 Brain Saved: {params_path}")
    print("--> Restart the dashboard to use the trained model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50000, help='Number of training epochs')
    args = parser.parse_args()
    
    try:
        train_scientific(epochs=args.epochs)
    except KeyboardInterrupt:
        print("\n--> Training Paused by Researcher.")
