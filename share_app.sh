#!/bin/bash
echo "=================================================="
echo "   🌍 ANTIGRAVITY | LIVE SHARE LINK GENERATOR    "
echo "=================================================="
echo "Creating a secure tunnel to your local app..."
echo "Share the URL below with your collaborator."
echo "Press Ctrl+C to stop sharing."
echo "--------------------------------------------------"

# specific port forwarding to localhost.run
ssh -R 80:localhost:8501 nokey@localhost.run
