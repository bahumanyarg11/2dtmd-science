import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="2DTMD Orchestrator")
    parser.add_argument('--mode', type=str, default='deploy', choices=['train', 'deploy'], help='Operation mode')
    args = parser.parse_args()

    # Create Checkpoint Directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if args.mode == 'train':
        print("initiating Phase 2 Training Protocol...")
        # Run training script
        # Fixed path to match structure: dtmd/core_ai/train_phase2.py
        subprocess.run([sys.executable, "dtmd/core_ai/train_phase2.py"])
        
    elif args.mode == 'deploy':
        print("Deploying 2DTMD Console...")
        # Make script executable
        subprocess.run(["chmod", "+x", "run_dtmd.sh"])
        # Execute Shell Script
        subprocess.run(["./run_dtmd.sh"])

if __name__ == "__main__":
    main()
