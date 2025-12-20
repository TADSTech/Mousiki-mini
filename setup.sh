#!/bin/bash
set -e

echo "Setting up Mousiki Mini..."

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run CLI setup
echo "Verifying setup..."
python mousiki_cli.py setup

echo "Setup Complete! Run ./run_demo.sh to see it in action."
