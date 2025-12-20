"""
Supabase ETL Loader

Loads processed music data into Supabase database with genre classification.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.genre_classifier.classifier import GenreClassifier
from utils.logging_config import error, neutral, success, warning, setup_logging

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    error("supabase-py not installed. Run: pip install supabase", "ETL")
    sys.exit(1)


class SupabaseLoader:
    """Load data into Supabase with genre classification."""
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        batch_size: int = 1000
    ):
        """
        Initialize Supabase loader.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service key
            batch_size: Number of records per batch
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.batch_size = batch_size
        
        # Initialize Supabase client
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            success("Connected to Supabase", "ETL")
        except Exception as e:
            error(f"Failed to connect to Supabase: {e}", "ETL")
            raise
        
        # Initialize genre classifier
        self.classifier = GenreClassifier()
        neutral("Genre classifier initialized", "ETL")
    
    def load_tracks_from_csv(
        self,
        csv_path: str,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> int:
        """
        Load tracks from CSV into Supabase.
        
        Args:
            csv_path: Path to CSV file
            limit: Maximum number of tracks to load (None = all)
            skip_existing: Skip tracks that already exist
            
        Returns:
            Number of tracks loaded
        """
        neutral(f"Loading tracks from {csv_path}", "ETL")
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path, nrows=limit)
            neutral(f"Read {len(df):,} tracks from CSV", "ETL")
        except Exception as e:
            error(f"Failed to read CSV: {e}", "ETL")
            raise
        
        # Ensure required columns exist
        required_cols = ['artist', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error(f"Missing required columns: {missing_cols}", "ETL")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Ensure genre/confidence columns exist
        if 'genre' not in df.columns:
            df['genre'] = None
        if 'genre_confidence' not in df.columns:
            df['genre_confidence'] = None

        # Classify only missing genres
        missing_mask = df['genre'].isna() | (df['genre'] == '')
        if missing_mask.any():
            neutral(f"Classifying {missing_mask.sum():,} tracks (missing genres)...", "ETL")
            records = df.loc[missing_mask].to_dict('records')
            genres_and_confidence = self.classifier.classify_batch(
                records,
                artist_col='artist',
                title_col='title',
                lyrics_col='text' if 'text' in df.columns else ('text_clean' if 'text_clean' in df.columns else '')
            )
            df.loc[missing_mask, 'genre'] = [g for g, _ in genres_and_confidence]
            df.loc[missing_mask, 'genre_confidence'] = [c for _, c in genres_and_confidence]
            success(f"Classified {missing_mask.sum():,} tracks into genres", "ETL")
        else:
            neutral("All rows already have genre; skipping classification", "ETL")
        
        # Prepare records for insertion
        records = []
        for idx, row in df.iterrows():
            record = {
                'title': str(row['title'])[:500],  # Truncate to VARCHAR(500)
                'artist': str(row['artist'])[:500],
                'album': str(row.get('album', ''))[:500] if pd.notna(row.get('album')) else None,
                'genre': row['genre'],
                'duration': int(row['duration']) if 'duration' in row and pd.notna(row['duration']) else None,
                'year': int(row['year']) if 'year' in row and pd.notna(row['year']) else None,
                'popularity': float(row['popularity']) if 'popularity' in row and pd.notna(row['popularity']) else 0.0,
            }
            records.append(record)
        
        # Insert in batches
        total_inserted = 0
        total_skipped = 0
        total_errors = 0
        
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(records) + self.batch_size - 1) // self.batch_size
            
            neutral(f"Inserting batch {batch_num}/{total_batches} ({len(batch)} tracks)...", "ETL")
            
            try:
                # Use upsert to handle duplicates
                result = self.client.table('tracks').upsert(
                    batch,
                    on_conflict='title,artist',
                    ignore_duplicates=skip_existing
                ).execute()
                
                inserted_count = len(result.data) if result.data else 0
                total_inserted += inserted_count
                
                if inserted_count < len(batch):
                    skipped = len(batch) - inserted_count
                    total_skipped += skipped
                    warning(f"Batch {batch_num}: inserted {inserted_count}, skipped {skipped} duplicates", "ETL")
                else:
                    success(f"Batch {batch_num}: inserted {inserted_count} tracks", "ETL")
                
            except Exception as e:
                error(f"Batch {batch_num} failed: {e}", "ETL")
                total_errors += len(batch)
                continue
        
        # Summary
        neutral("=" * 60, "ETL")
        success(f"Total inserted: {total_inserted:,} tracks", "ETL")
        if total_skipped > 0:
            warning(f"Total skipped: {total_skipped:,} duplicates", "ETL")
        if total_errors > 0:
            error(f"Total errors: {total_errors:,} tracks", "ETL")
        neutral("=" * 60, "ETL")
        
        return total_inserted
    
    def get_track_count(self) -> int:
        """Get total number of tracks in Supabase."""
        try:
            result = self.client.table('tracks').select('track_id', count='exact').limit(1).execute()
            return result.count if hasattr(result, 'count') else 0
        except Exception as e:
            error(f"Failed to get track count: {e}", "ETL")
            return 0
    
    def get_genre_distribution(self) -> dict:
        """Get genre distribution from Supabase."""
        try:
            result = self.client.table('tracks').select('genre').execute()
            if result.data:
                genres = [row['genre'] for row in result.data if row.get('genre')]
                from collections import Counter
                return dict(Counter(genres))
            return {}
        except Exception as e:
            error(f"Failed to get genre distribution: {e}", "ETL")
            return {}


def main():
    """Main ETL pipeline to load data into Supabase."""
    setup_logging()
    
    # Load environment variables
    SUPABASE_URL = os.getenv('VITE_SUPABASE_URL', 'https://itkniqojrujdacjcvopt.supabase.co')
    SUPABASE_KEY = os.getenv('VITE_SUPABASE_ANON_KEY', 'sb_publishable_onJ0D0L2WzLcL3k-GCMPxA_Z7R-n0Sl')
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        error("Missing Supabase credentials in environment", "ETL")
        sys.exit(1)
    
    neutral("=" * 60, "ETL")
    neutral("SUPABASE ETL PIPELINE", "ETL")
    neutral("=" * 60, "ETL")
    
    # Initialize loader
    loader = SupabaseLoader(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
        batch_size=500  # Smaller batches for Supabase
    )
    
    # Check current track count
    current_count = loader.get_track_count()
    neutral(f"Current tracks in Supabase: {current_count:,}", "ETL")
    
    # Load tracks from processed CSV (prefer pre-classified file when present)
    csv_path = os.getenv(
        'ETL_CSV_PATH',
        './data/processed/tracks_with_genre.csv'
    )
    if not Path(csv_path).exists():
        # Fallback to original preprocessed file and classify on the fly
        fallback = './data/processed/tracks_preprocessed.csv'
        warning(f"File not found: {csv_path}. Falling back to {fallback}", "ETL")
        csv_path = fallback
    
    if not Path(csv_path).exists():
        error(f"File not found: {csv_path}", "ETL")
        sys.exit(1)
    
    # Load tracks (use limit for testing, remove for full load)
    # For testing: limit=1000
    # For production: limit=None
    limit_env = os.getenv('ETL_LIMIT', 'None')
    limit = int(limit_env) if limit_env != 'None' else None
    
    if limit:
        neutral(f"Loading up to {limit:,} tracks from CSV...", "ETL")
    else:
        neutral("Loading all tracks from CSV...", "ETL")
    
    try:
        inserted = loader.load_tracks_from_csv(
            csv_path=csv_path,
            limit=limit,
            skip_existing=True
        )
        
        success(f"✓ ETL completed! Inserted {inserted:,} tracks", "ETL")
        
        # Show genre distribution
        neutral("Genre distribution:", "ETL")
        distribution = loader.get_genre_distribution()
        for genre, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            neutral(f"  {genre:15} : {count:,} tracks", "ETL")
        
    except Exception as e:
        error(f"ETL failed: {e}", "ETL")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
