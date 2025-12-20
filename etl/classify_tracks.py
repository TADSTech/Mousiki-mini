"""
Classify all tracks and write a new dataset with genres.

Reads `data/processed/tracks_preprocessed.csv` in chunks,
classifies genres using the GenreClassifier, and writes
`data/processed/tracks_with_genre.csv` with added columns:
  - genre
  - genre_confidence

Usage:
    python3 -m etl.classify_tracks

Environment variables:
    CLASSIFIER_CHUNK_SIZE  (default: 50000)
    CLASSIFIER_LIMIT       (optional: limit total rows)
"""

import os
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.genre_classifier.classifier import GenreClassifier
from utils.logging_config import neutral, success, error, warning, setup_logging

INPUT_CSV = Path("./data/processed/tracks_preprocessed.csv")
OUTPUT_CSV = Path("./data/processed/tracks_with_genre.csv")


def classify_chunk(df: pd.DataFrame, classifier: GenreClassifier) -> pd.DataFrame:
    """Classify a chunk of tracks and return the DataFrame with genre columns."""
    # Ensure required columns
    required = ['artist', 'title']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # If genre already present, fill only missing
    needs_classification = df['genre'].isna() if 'genre' in df.columns else pd.Series(True, index=df.index)

    if needs_classification.any():
        neutral(f"Classifying {needs_classification.sum():,} rows...", "CLASSIFY")
        records = df.loc[needs_classification].to_dict('records')
        genres_conf = classifier.classify_batch(
            records,
            artist_col='artist',
            title_col='title',
            lyrics_col='text' if 'text' in df.columns else ('text_clean' if 'text_clean' in df.columns else '')
        )
        df.loc[needs_classification, 'genre'] = [g for g, _ in genres_conf]
        df.loc[needs_classification, 'genre_confidence'] = [c for _, c in genres_conf]
    else:
        neutral("Genre column already populated; skipping classification", "CLASSIFY")

    return df


def main():
    setup_logging()

    if not INPUT_CSV.exists():
        error(f"Input file not found: {INPUT_CSV}", "CLASSIFY")
        sys.exit(1)

    chunk_size = int(os.getenv('CLASSIFIER_CHUNK_SIZE', '50000'))
    total_limit_env = os.getenv('CLASSIFIER_LIMIT')
    total_limit: Optional[int] = int(total_limit_env) if total_limit_env else None

    classifier = GenreClassifier()
    success("Genre classifier initialized", "CLASSIFY")

    # Remove old output if exists
    if OUTPUT_CSV.exists():
        warning(f"Output file exists, overwriting: {OUTPUT_CSV}", "CLASSIFY")
        OUTPUT_CSV.unlink()

    total_processed = 0
    chunk_idx = 0

    with pd.read_csv(INPUT_CSV, chunksize=chunk_size) as reader:
        for chunk in reader:
            if total_limit is not None and total_processed >= total_limit:
                break

            if total_limit is not None:
                # Trim chunk if exceeding limit
                remaining = total_limit - total_processed
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            chunk_idx += 1
            neutral(f"Processing chunk {chunk_idx} ({len(chunk):,} rows)...", "CLASSIFY")

            # Ensure genre/confidence columns exist
            if 'genre' not in chunk.columns:
                chunk['genre'] = None
            if 'genre_confidence' not in chunk.columns:
                chunk['genre_confidence'] = None

            chunk = classify_chunk(chunk, classifier)

            # Append to output CSV
            mode = 'a' if OUTPUT_CSV.exists() else 'w'
            header = not OUTPUT_CSV.exists()
            chunk.to_csv(OUTPUT_CSV, index=False, mode=mode, header=header)

            total_processed += len(chunk)
            success(f"Chunk {chunk_idx} done. Total processed: {total_processed:,}", "CLASSIFY")

    success(f"Completed classification. Total rows processed: {total_processed:,}", "CLASSIFY")
    success(f"Output written to: {OUTPUT_CSV}", "CLASSIFY")


if __name__ == "__main__":
    main()
