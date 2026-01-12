import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Antigravity Orchestrator")
    parser.add_argument('--mode', type=str, default='deploy', choices=['train', 'deploy'], help='Operation mode')
    args = parser.parse_args()

    # Create Checkpoint Directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if args.mode == 'train':
        print("initiating Phase 2 Training Protocol...")
        # Run training script
        # Fixed path to match structure: antigravity/core_ai/train_phase2.py
        subprocess.run([sys.executable, "antigravity/core_ai/train_phase2.py"])
        
    elif args.mode == 'deploy':
        print("Deploying Antigravity Console...")
        # Make script executable
        subprocess.run(["chmod", "+x", "run_antigravity.sh"])
        # Execute Shell Script
        subprocess.run(["./run_antigravity.sh"])

if __name__ == "__main__":
    main()
