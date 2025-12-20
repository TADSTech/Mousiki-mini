#!/bin/bash
set -e

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run demo
echo "Running Mousiki Demo..."
python mousiki_cli.py recommend --user-id 1
