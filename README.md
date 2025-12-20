# Mousiki Mini 🎵

A simplified, reproducible, and local-first music recommendation system.

## Overview
Mousiki Mini is a stripped-down version of the Mousiki Recommendation Engine, designed to demonstrate the core content-based and collaborative filtering algorithms without the complexity of a production backend.

## Features
- **Hybrid Recommendations**: Combines Content-Based Filtering (CBF) and Neural Collaborative Filtering (CF).
- **Local Execution**: Runs entirely on your machine using local model weights and data.
- **CLI Interface**: Interactively check validity and generate recommendations.
- **Reproducible**: Single-script setup.

## Quick Start

1. **Setup Environment**
   ```bash
   ./setup.sh
   # This will create a virtual environment, install dependencies, and verify models.
   ```

2. **Run Demo**
   ```bash
   ./run_demo.sh
   # Generates recommendations for User 1.
   ```

## CLI Usage

Mousiki Mini comes with a `typer`-based CLI tool `mousiki_cli.py`.

### Check Setup
Verify that all models and mappings are correctly detected.
```bash
python mousiki_cli.py setup
```

### Get Recommendations
Generate recommendations for a specific user.
```bash
python mousiki_cli.py recommend --user-id 1 --n 10
```

**Options:**
- `--user-id`: (Required) Integer ID of the user.
- `--n`: Number of recommendations (default: 10).
- `--cbf / --no-cbf`: Enable/Disable Content-Based Filtering.
- `--cf / --no-cf`: Enable/Disable Collaborative Filtering.
- `--history "123,456"`: Override user history with a comma-separated list of track IDs.

### List Users
See which users are available in the model.
```bash
python mousiki_cli.py list-users --limit 20
```

## Project Structure
- `models/`: Contains the pre-trained model weights (CBF matrix, CF checkpoints).
- `data/`: Raw and processed data (CSV).
- `scripts/`: Training and evaluation scripts (legacy/advanced usage).
- `mousiki_cli.py`: Main entry point.

## Evaluation
To reproduce the training or evaluation, refer to the scripts in `scripts/`.
For example, to evaluate the current models:
```bash
python scripts/evaluate_models.py
```

## License
MIT
