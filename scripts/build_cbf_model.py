#!/usr/bin/env python3
"""
Content-Based Filtering Model Builder Script.

Initializes the CBF recommender by:
1. Loading or generating embeddings for all tracks
2. Computing similarity matrix
3. Saving the model for production use
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning
from utils.config import settings
from models.content.cbf import ContentBasedFilterer


def main(args: argparse.Namespace) -> None:
    """Build and save the CBF model."""
    try:
        neutral("=" * 60, "CBF_BUILDER")
        neutral("Content-Based Filtering Model Builder", "CBF_BUILDER")
        neutral("=" * 60, "CBF_BUILDER")
        
        # Build connection string
        db_connection_string = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        
        # Initialize CBF
        cbf = ContentBasedFilterer(
            device=args.device,
            use_sparse=args.sparse,
            normalize_embeddings=True
        )
        
        neutral(f"Configuration:", "CBF_BUILDER")
        neutral(f"  Device: {cbf.device}", "CBF_BUILDER")
        neutral(f"  Batch Size: {cbf.batch_size}", "CBF_BUILDER")
        neutral(f"  Use Sparse: {cbf.use_sparse}", "CBF_BUILDER")
        
        # Load or build
        start_time = time.time()
        cbf.load_or_build(db_connection_string)
        elapsed = time.time() - start_time
        
        # Get stats
        stats = cbf.get_stats()
        success(f"Model built successfully in {elapsed:.2f}s", "CBF_BUILDER")
        neutral(f"Stats: {len(stats)} metadata fields", "CBF_BUILDER")
        
        # Save model
        cbf.save_model()
        
        # Print stats
        neutral("=" * 60, "CBF_BUILDER")
        neutral("Model Statistics:", "CBF_BUILDER")
        for key, value in stats.items():
            neutral(f"  {key}: {value}", "CBF_BUILDER")
        neutral("=" * 60, "CBF_BUILDER")
        
        success("CBF model initialization complete!", "CBF_BUILDER")
        
    except Exception as e:
        error(f"Failed to build CBF model: {str(e)}", "CBF_BUILDER")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and save the Content-Based Filtering model"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for embedding generation (default: auto-detect)"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=True,
        help="Use sparse matrices for similarity (memory-efficient)"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense matrices instead of sparse"
    )
    
    args = parser.parse_args()
    
    # Handle sparse/dense flag
    if args.dense:
        args.sparse = False
    
    main(args)
