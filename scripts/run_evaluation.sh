#!/bin/bash
# Quick evaluation script for Mousiki recommender models

echo "=========================================="
echo "Mousiki Model Evaluation"
echo "=========================================="
echo ""

# Check if evaluation results directory exists
if [ ! -d "evaluation_results" ]; then
    mkdir -p evaluation_results
    echo "✓ Created evaluation_results directory"
fi

# Run evaluation
echo "Starting model evaluation..."
echo ""

python scripts/evaluate_models.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Evaluation Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to: evaluation_results/"
    echo ""
    echo "View results:"
    echo "  ls -lh evaluation_results/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Evaluation Failed"
    echo "=========================================="
    echo ""
    echo "Check logs above for errors"
    exit 1
fi
