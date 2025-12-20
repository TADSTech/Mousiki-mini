#!/usr/bin/env python3
"""
Export tracks data to CSV for offline Colab training.

This is an alternative to TCP tunneling - export data locally,
upload CSV to Colab, train there, then download results.
"""

import sys
import argparse
from pathlib import Path

import psycopg2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning
from utils.config import settings


def export_tracks_to_csv(output_file: str = "./data/tracks_export.csv") -> None:
    """Export all tracks to CSV for Colab upload."""
    try:
        neutral("Connecting to database...", "EXPORTER")
        
        db_connection_string = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        
        conn = psycopg2.connect(db_connection_string)
        
        neutral("Fetching all tracks...", "EXPORTER")
        query = """
            SELECT 
                track_id,
                title,
                artist,
                album,
                genre,
                year,
                duration,
                popularity
            FROM tracks
            ORDER BY track_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        success(f"Fetched {len(df):,} tracks", "EXPORTER")
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        neutral(f"Saving to {output_file}...", "EXPORTER")
        df.to_csv(output_file, index=False)
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        success(f"Exported to {output_file} ({file_size_mb:.1f} MB)", "EXPORTER")
        
        print("\n" + "=" * 70)
        print("✓ Export complete!")
        print("=" * 70)
        print(f"File: {output_file}")
        print(f"Rows: {len(df):,}")
        print(f"Size: {file_size_mb:.1f} MB")
        print("\nNext steps:")
        print("1. Upload this file to Google Colab")
        print("2. Use the offline training section in the Colab notebook")
        print("3. Download trained embeddings")
        print("4. Import back: python scripts/update_db_embeddings.py")
        print("=" * 70)
        
    except Exception as e:
        error(f"Export failed: {str(e)}", "EXPORTER")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export tracks to CSV for offline Colab training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/tracks_export.csv",
        help="Output CSV file path (default: ./data/tracks_export.csv)"
    )
    
    args = parser.parse_args()
    export_tracks_to_csv(args.output)
