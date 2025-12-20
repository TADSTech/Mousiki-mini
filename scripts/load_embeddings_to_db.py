#!/usr/bin/env python3
"""
Update Database with Pre-computed Embeddings

Uses embeddings from LyricsEmbedder.save() and updates the database.
"""

import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import psycopg2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning
from utils.config import settings


def find_latest_embeddings_file(embeddings_dir: Path = Path("./data/embeddings")) -> Path:
    """Find the most recent embeddings file."""
    pkl_files = sorted(embeddings_dir.glob("all_tracks_embeddings_*.pkl"))
    if not pkl_files:
        pkl_files = sorted(embeddings_dir.glob("embeddings_*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No embeddings files found in {embeddings_dir}")
    
    latest = pkl_files[-1]
    neutral(f"Found embeddings file: {latest.name}", "LOADER")
    return latest


def load_embeddings_file(filepath: Path) -> tuple:
    """Load embeddings from pickle file."""
    try:
        neutral(f"Loading embeddings from {filepath.name}...", "LOADER")
        
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
        
        success(f"Loaded {len(embeddings):,} embeddings (shape: {embeddings.shape})", "LOADER")
        return track_ids, embeddings
        
    except Exception as e:
        error(f"Failed to load embeddings: {str(e)}", "LOADER")
        raise


def update_database_with_embeddings(
    conn_string: str,
    track_ids: np.ndarray,
    embeddings: np.ndarray,
    batch_size: int = 100
) -> int:
    """Update PostgreSQL tracks table with embeddings."""
    conn = None
    try:
        neutral(f"Connecting to database...", "LOADER")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # Check if embedding column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='tracks' AND column_name='embedding'
        """)
        
        if not cursor.fetchone():
            neutral("Adding 'embedding' column to tracks table...", "LOADER")
            cursor.execute("""
                ALTER TABLE tracks 
                ADD COLUMN embedding FLOAT8[] 
            """)
            conn.commit()
            success("Added 'embedding' column", "LOADER")
        
        neutral(f"Updating {len(track_ids):,} embeddings (batch_size={batch_size})...", "LOADER")
        start_time = time.time()
        
        updated = 0
        for batch_idx in range(0, len(track_ids), batch_size):
            batch_end = min(batch_idx + batch_size, len(track_ids))
            batch_ids = track_ids[batch_idx:batch_end]
            batch_embeddings = embeddings[batch_idx:batch_end]
            
            # Build VALUES clause for batch update
            update_rows = []
            for track_id, embedding in zip(batch_ids, batch_embeddings):
                embedding_list = embedding.tolist()
                embedding_str = "{" + ",".join(f"{x:.6f}" for x in embedding_list) + "}"
                track_id_int = int(track_id)
                update_rows.append(f"({track_id_int}, ARRAY{embedding_str}::FLOAT8[])")
            
            values_clause = ",".join(update_rows)
            query = f"""
                UPDATE tracks 
                SET embedding = data.embedding 
                FROM (VALUES {values_clause}) AS data(track_id, embedding)
                WHERE tracks.track_id = data.track_id
            """
            
            cursor.execute(query)
            rows_affected = cursor.rowcount
            updated += rows_affected
            conn.commit()
            
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(track_ids) - 1) // batch_size + 1
            neutral(f"Batch {batch_num}/{total_batches}: updated {rows_affected:,} rows", "LOADER")
        
        elapsed = time.time() - start_time
        success(f"Updated {updated:,} embeddings in {elapsed:.2f}s", "LOADER")
        
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
        neutral("Update Database with Embeddings", "LOADER")
        neutral("=" * 70, "LOADER")
        
        # Find embeddings file
        embeddings_file = Path(args.embeddings_file) if args.embeddings_file else find_latest_embeddings_file()
        
        # Load embeddings
        neutral("\n[Step 1/2] Loading embeddings...", "LOADER")
        track_ids, embeddings = load_embeddings_file(embeddings_file)
        
        # Build connection string
        db_connection_string = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        
        # Update database
        neutral("\n[Step 2/2] Updating PostgreSQL database...", "LOADER")
        updated_count = update_database_with_embeddings(
            db_connection_string,
            track_ids,
            embeddings,
            batch_size=args.batch_size
        )
        
        # Summary
        neutral("\n" + "=" * 70, "LOADER")
        success(f"✓ Successfully updated {updated_count:,} tracks with embeddings", "LOADER")
        success(f"✓ Embedding dimension: {embeddings.shape[1]}", "LOADER")
        neutral("=" * 70, "LOADER")
        
    except Exception as e:
        error(f"Workflow failed: {str(e)}", "LOADER")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update PostgreSQL database with pre-computed embeddings"
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default=None,
        help="Path to embeddings pickle file (auto-finds latest if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for database updates (default: 100)"
    )
    
    args = parser.parse_args()
    main(args)
