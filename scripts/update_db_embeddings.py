#!/usr/bin/env python3
"""
Fast database update using temporary table (optimized).
"""

import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import psycopg2
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning
from utils.config import settings


def find_latest_embeddings_file(embeddings_dir: Path = Path("./data/embeddings")) -> Path:
    """Find the most recent embeddings file."""
    pkl_files = sorted(embeddings_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No embeddings files found in {embeddings_dir}")
    
    latest = pkl_files[-1]
    neutral(f"Found embeddings file: {latest.name} ({latest.stat().st_size / 1024 / 1024:.1f} MB)", "LOADER")
    return latest


def load_embeddings_file(filepath: Path) -> tuple:
    """Load embeddings from pickle file."""
    try:
        neutral(f"Loading embeddings...", "LOADER")
        start = time.time()
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both 'ids' and 'track_ids' keys
        if 'track_ids' in data:
            track_ids = data['track_ids']
        elif 'ids' in data:
            track_ids = np.array(data['ids'])
        else:
            raise ValueError("No track IDs found in embeddings file")
        
        embeddings = data['embeddings']
        elapsed = time.time() - start
        
        success(f"Loaded {len(embeddings):,} embeddings in {elapsed:.2f}s (shape: {embeddings.shape})", "LOADER")
        return track_ids, embeddings
        
    except Exception as e:
        error(f"Failed to load embeddings: {str(e)}", "LOADER")
        raise


def update_database_fast(conn_string: str, track_ids: np.ndarray, embeddings: np.ndarray) -> int:
    """Fast database update using temporary table."""
    conn = None
    try:
        neutral("Connecting to database...", "LOADER")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # Add column if needed
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='tracks' AND column_name='embedding'
        """)
        
        if not cursor.fetchone():
            neutral("Adding 'embedding' column...", "LOADER")
            cursor.execute("ALTER TABLE tracks ADD COLUMN embedding FLOAT8[]")
            conn.commit()
            success("Added 'embedding' column", "LOADER")
        
        # Create temp table
        neutral("Creating temporary table...", "LOADER")
        cursor.execute("""
            DROP TABLE IF EXISTS temp_embeddings;
            CREATE TEMP TABLE temp_embeddings (
                track_id INT PRIMARY KEY,
                embedding FLOAT8[] NOT NULL
            )
        """)
        conn.commit()
        
        # Insert embeddings into temp table efficiently
        neutral(f"Inserting {len(embeddings):,} embeddings into temp table...", "LOADER")
        start = time.time()
        
        # Prepare CSV data for COPY (fastest method)
        csv_buffer = StringIO()
        for track_id, embedding in zip(track_ids, embeddings):
            embedding_str = "{" + ",".join(f"{x:.6f}" for x in embedding) + "}"
            csv_buffer.write(f"{int(track_id)}\t{embedding_str}\n")
        
        csv_buffer.seek(0)
        
        # Use COPY for fast insert
        cursor.copy_from(csv_buffer, 'temp_embeddings', columns=['track_id', 'embedding'])
        conn.commit()
        
        insert_time = time.time() - start
        success(f"Inserted into temp table in {insert_time:.2f}s", "LOADER")
        
        # Update main table from temp table
        neutral("Updating tracks table from temp table...", "LOADER")
        start = time.time()
        
        cursor.execute("""
            UPDATE tracks t
            SET embedding = te.embedding
            FROM temp_embeddings te
            WHERE t.track_id = te.track_id
        """)
        
        updated = cursor.rowcount
        conn.commit()
        
        update_time = time.time() - start
        success(f"Updated {updated:,} tracks in {update_time:.2f}s", "LOADER")
        
        cursor.close()
        return updated
        
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass
        error(f"Failed to update database: {str(e)}", "LOADER")
        raise
    finally:
        if conn:
            conn.close()


def main(args: argparse.Namespace) -> None:
    """Main workflow."""
    try:
        neutral("=" * 70, "LOADER")
        neutral("Update Database with Embeddings (Optimized)", "LOADER")
        neutral("=" * 70, "LOADER")
        
        # Find and load embeddings
        embeddings_file = Path(args.embeddings_file) if args.embeddings_file else find_latest_embeddings_file()
        neutral("\n[Step 1/2] Loading embeddings...", "LOADER")
        track_ids, embeddings = load_embeddings_file(embeddings_file)
        
        # Build connection string
        db_connection_string = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        
        # Update database
        neutral("\n[Step 2/2] Updating database...", "LOADER")
        updated_count = update_database_fast(db_connection_string, track_ids, embeddings)
        
        # Summary
        neutral("\n" + "=" * 70, "LOADER")
        success(f"✓ Updated {updated_count:,} tracks with embeddings", "LOADER")
        success(f"✓ Embedding dimension: {embeddings.shape[1]}", "LOADER")
        neutral("=" * 70, "LOADER")
        
    except Exception as e:
        error(f"Workflow failed: {str(e)}", "LOADER")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update database with embeddings (fast)")
    parser.add_argument("--embeddings-file", type=str, default=None, help="Path to embeddings file (auto-finds latest if omitted)")
    args = parser.parse_args()
    main(args)
