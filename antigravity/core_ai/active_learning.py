import torch
import torch.nn as nn
import os
import sys

# Ensure we can find the modules from parent directory if run directly or as package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from antigravity.core_ai.fno_synthesis import SynthesisFNO

class AntigravityLearner:
    """
    The Self-Correction Mechanism.
    Allows the model to learn from new experimental data points (Active Learning).
    """
    def __init__(self, model_path="checkpoints/fno_latest.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SynthesisFNO().to(self.device)
        
        # Load existing weights if available
        try:
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("--> Learner loaded previous knowledge.")
            else:
                print("--> No checkpoint found. Learner initialized from scratch.")
        except Exception as e:
            print(f"--> Error loading checkpoint: {e}. Starting from scratch.")
            
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.MSELoss()

    def ingest_lab_result(self, temp, pressure, flow, observed_concentration_grid):
        """
        Fine-tunes the Digital Twin based on a real-world experiment.
        """
        self.model.train()
        
        # 1. Prepare Inputs
        # (Batch=1, Grid=64, Grid=64, Features=4)
        x = torch.zeros(1, 64, 64, 4).to(self.device)
        x[:, :, :, 0] = temp / 1000.0
        x[:, :, :, 1] = pressure / 100.0
        x[:, :, :, 2] = flow
        
        # 2. Prepare Target (Ground Truth from Lab)
        target = torch.tensor(observed_concentration_grid).float().to(self.device)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(-1) # Add batch and channel dims

        # 3. Optimization Step (Few-Shot Learning)
        print("--> Assimilating new experimental data...")
        loss_val = 0.0
        for i in range(5): # Rapid fine-tuning steps
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            
        print(f"--> Knowledge Updated. Error reduced to: {loss_val:.6f}")
        
        # Save new brain
        # Checkpoint dir might be relative
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        return loss_val
