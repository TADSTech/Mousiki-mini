#!/usr/bin/env python3
"""
Embed All Tracks Script

Uses the LyricsEmbedder to:
1. Fetch all tracks from the database
2. Generate embeddings for track title + artist + genre
3. Save embeddings to disk
4. Update PostgreSQL tracks table with embedding vectors
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning
from utils.config import settings
from models.embedder.embedder import LyricsEmbedder


def fetch_all_tracks(conn_string: str) -> dict:
    """Fetch all tracks from database."""
    try:
        neutral("Connecting to database...", "EMBEDDER")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM tracks")
        total = cursor.fetchone()[0]
        neutral(f"Found {total:,} tracks in database", "EMBEDDER")
        
        # Fetch all tracks with relevant fields
        cursor.execute("""
            SELECT 
                track_id,
                title,
                artist,
                album,
                genre
            FROM tracks
            ORDER BY track_id
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            'total': total,
            'rows': rows
        }
        
    except Exception as e:
        error(f"Failed to fetch tracks: {str(e)}", "EMBEDDER")
        raise


def prepare_texts(rows: list) -> tuple:
    """Prepare texts for embedding from track rows."""
    track_ids = []
    texts = []
    
    for track_id, title, artist, album, genre in rows:
        track_ids.append(track_id)
        
        # Combine fields for rich text representation
        text_parts = [
            title or "",
            artist or "",
            genre or ""
        ]
        text = " ".join([p for p in text_parts if p]).strip()
        texts.append(text if text else f"Track {track_id}")
    
    return np.array(track_ids), texts


def generate_embeddings(embedder: LyricsEmbedder, texts: list, batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for texts."""
    try:
        neutral(f"Generating embeddings for {len(texts):,} tracks (batch_size={batch_size})...", "EMBEDDER")
        start_time = time.time()
        
        embeddings = embedder.embed(texts, batch_size=batch_size)
        
        elapsed = time.time() - start_time
        success(
            f"Generated {len(embeddings):,} embeddings in {elapsed:.2f}s "
            f"(avg: {elapsed/len(texts):.4f}s per track)",
            "EMBEDDER"
        )
        
        return embeddings
        
    except Exception as e:
        error(f"Failed to generate embeddings: {str(e)}", "EMBEDDER")
        raise


def update_database_with_embeddings(
    conn_string: str,
    track_ids: np.ndarray,
    embeddings: np.ndarray,
    batch_size: int = 1000
) -> int:
    """Update PostgreSQL tracks table with embeddings."""
    try:
        neutral(f"Connecting to database to update embeddings...", "EMBEDDER")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # First, check if embedding column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='tracks' AND column_name='embedding'
        """)
        
        if not cursor.fetchone():
            neutral("Adding 'embedding' column to tracks table...", "EMBEDDER")
            cursor.execute("""
                ALTER TABLE tracks 
                ADD COLUMN embedding FLOAT8[] 
            """)
            conn.commit()
            success("Added 'embedding' column", "EMBEDDER")
        
        # Batch update using individual UPDATE statements
        neutral(f"Updating {len(track_ids):,} embeddings in database (batch_size={batch_size})...", "EMBEDDER")
        
        start_time = time.time()
        
        for batch_idx in range(0, len(track_ids), batch_size):
            batch_end = min(batch_idx + batch_size, len(track_ids))
            batch_ids = track_ids[batch_idx:batch_end]
            batch_embeddings = embeddings[batch_idx:batch_end]
            
            # Build batch update query
            update_statements = []
            for track_id, embedding in zip(batch_ids, batch_embeddings):
                embedding_list = embedding.tolist()
                embedding_str = "{" + ",".join(str(x) for x in embedding_list) + "}"
                track_id_int = int(track_id)
                update_statements.append(f"({track_id_int}, ARRAY{embedding_str}::FLOAT8[])")
            
            # Use CASE statement for efficient batch update
            case_values = ",".join(update_statements)
            query = f"""
                UPDATE tracks 
                SET embedding = data.embedding 
                FROM (VALUES {case_values}) AS data(track_id, embedding)
                WHERE tracks.track_id = data.track_id
            """
            
            cursor.execute(query)
            conn.commit()
            
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(track_ids) - 1) // batch_size + 1
            neutral(f"Updated batch {batch_num}/{total_batches}", "EMBEDDER")
        
        elapsed = time.time() - start_time
        success(f"Updated {len(track_ids):,} embeddings in {elapsed:.2f}s", "EMBEDDER")
        
        cursor.close()
        conn.close()
        
        return len(track_ids)
        
    except Exception as e:
        if conn:
            conn.rollback()
        error(f"Failed to update database: {str(e)}", "EMBEDDER")
        raise


def save_embeddings_file(
    track_ids: np.ndarray,
    embeddings: np.ndarray,
    embedder: LyricsEmbedder
) -> Path:
    """Save embeddings to disk for backup."""
    try:
        embeddings_file = embedder.save(
            embeddings=embeddings,
            ids=track_ids,
            metadata={
                'source': 'all_tracks_batch',
                'timestamp': time.time()
            },
            filename="all_tracks_embeddings"
        )
        return embeddings_file
    except Exception as e:
        error(f"Failed to save embeddings file: {str(e)}", "EMBEDDER")
        raise


def detect_device(device_arg: str) -> str:
    """Detect best device to use."""
    if device_arg != "auto":
        return device_arg
    
    # Try to detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            success(f"GPU detected: {torch.cuda.get_device_name(0)}", "EMBEDDER")
            return device
    except ImportError:
        pass
    
    neutral("Using CPU (no GPU detected)", "EMBEDDER")
    return "cpu"


def main(args: argparse.Namespace) -> None:
    """Main workflow."""
    try:
        neutral("=" * 70, "EMBEDDER")
        neutral("Embedding All Tracks - Full Database Workflow", "EMBEDDER")
        neutral("=" * 70, "EMBEDDER")
        
        # Detect device
        device = detect_device(args.device)
        
        # Build connection string
        db_connection_string = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        
        # Step 1: Fetch all tracks
        neutral("\n[Step 1/5] Fetching tracks from database...", "EMBEDDER")
        fetch_result = fetch_all_tracks(db_connection_string)
        
        # Step 2: Prepare texts
        neutral("\n[Step 2/5] Preparing text for embedding...", "EMBEDDER")
        track_ids, texts = prepare_texts(fetch_result['rows'])
        neutral(f"Prepared {len(texts):,} texts", "EMBEDDER")
        
        # Step 3: Initialize embedder
        neutral("\n[Step 3/5] Initializing LyricsEmbedder...", "EMBEDDER")
        embedder = LyricsEmbedder(
            device=device,
            normalize_embeddings=True
        )
        
        # Step 4: Generate embeddings
        neutral("\n[Step 4/5] Generating embeddings...", "EMBEDDER")
        embeddings = generate_embeddings(
            embedder,
            texts,
            batch_size=args.batch_size
        )
        
        # Step 5: Update database
        neutral("\n[Step 5/5] Updating PostgreSQL database...", "EMBEDDER")
        updated_count = update_database_with_embeddings(
            db_connection_string,
            track_ids,
            embeddings,
            batch_size=args.db_batch_size
        )
        
        # Save embeddings file as backup
        neutral("\n[Bonus] Saving embeddings file as backup...", "EMBEDDER")
        embeddings_file = save_embeddings_file(track_ids, embeddings, embedder)
        
        # Summary
        neutral("\n" + "=" * 70, "EMBEDDER")
        success(f"✓ Successfully embedded {updated_count:,} tracks", "EMBEDDER")
        success(f"✓ Embeddings saved to: {embeddings_file.name}", "EMBEDDER")
        success(f"✓ Embedding dimension: {embeddings.shape[1]}", "EMBEDDER")
        success(f"✓ Database updated with vectors", "EMBEDDER")
        neutral("=" * 70, "EMBEDDER")
        
    except Exception as e:
        error(f"Workflow failed: {str(e)}", "EMBEDDER")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed all tracks from database using LyricsEmbedder"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for embedding (default: auto-detect GPU)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64 for CPU)"
    )
    parser.add_argument(
        "--db-batch-size",
        type=int,
        default=1000,
        help="Batch size for database updates (default: 1000)"
    )
    
    args = parser.parse_args()
    main(args)
