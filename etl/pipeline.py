"""
Production ETL Pipeline for Mousiki.

Orchestrates the complete ETL process: ingestion, preprocessing, embedding,
and database loading.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.ingest import DataIngester
from etl.preprocess_text import TextPreprocessor
from etl.database_loader import DatabaseLoader
from models.embedder.embedder import LyricsEmbedder, DataFrameEmbedderAdapter
from utils.logging_config import error, neutral, success, warning, setup_logging


class ETLPipeline:
    """
    Complete production ETL pipeline.
    
    Workflow:
    1. Ingest raw data
    2. Preprocess text
    3. Generate embeddings
    4. Load to database
    5. Save artifacts
    """
    
    def __init__(
        self,
        raw_data_path: str = "./data/raw",
        processed_path: str = "./data/processed"
    ):
        """
        Initialize ETL pipeline.
        
        Args:
            raw_data_path: Path to raw data directory.
            processed_path: Path to processed data directory.
        """
        self.raw_data_path = raw_data_path
        self.processed_path = processed_path
        
        # Initialize components
        self.ingester = DataIngester(raw_data_path)
        self.preprocessor = TextPreprocessor()
        self.embedder = None  # Lazy loaded
        self.db_loader = None  # Lazy loaded
        
        neutral("ETL Pipeline initialized", "PIPELINE")
    
    def run_tracks_pipeline(
        self,
        input_file: str,
        load_to_db: bool = True,
        generate_embeddings: bool = True
    ) -> pd.DataFrame:
        """
        Run complete ETL pipeline for tracks data.
        
        Args:
            input_file: Path to raw tracks file.
            load_to_db: Whether to load data to database.
            generate_embeddings: Whether to generate embeddings.
            
        Returns:
            Processed DataFrame.
        """
        neutral("="*60, "PIPELINE")
        neutral("Starting Tracks ETL Pipeline", "PIPELINE")
        neutral("="*60, "PIPELINE")
        
        # Step 1: Ingest
        neutral("Step 1/4: Ingesting data", "PIPELINE")
        df = self.ingester.ingest_tracks(input_file)
        
        # Step 2: Preprocess
        neutral("Step 2/4: Preprocessing text", "PIPELINE")
        df = self.preprocessor.preprocess_tracks(df)
        
        # Save preprocessed data
        output_file = "tracks_preprocessed.csv"
        self.ingester.save_processed(df, output_file, self.processed_path)
        
        # Step 3: Generate embeddings (optional)
        if generate_embeddings:
            neutral("Step 3/4: Generating embeddings", "PIPELINE")
            if self.embedder is None:
                self.embedder = LyricsEmbedder()
            
            # Use text_clean column if available, otherwise combined_features
            text_col = "text_clean" if "text_clean" in df.columns else "combined_features"
            
            if text_col in df.columns:
                texts = df[text_col].fillna("").tolist()
                embeddings = self.embedder.embed(texts, batch_size=64)
                
                # Save embeddings
                track_ids = df.index.tolist() if "track_id" not in df.columns else df["track_id"].tolist()
                self.embedder.save(
                    embeddings=embeddings,
                    ids=track_ids,
                    metadata={"source": "tracks", "text_column": text_col},
                    filename="track_embeddings"
                )
            else:
                warning("No text column found for embeddings", "PIPELINE")
        
        # Step 4: Load to database (optional)
        if load_to_db:
            neutral("Step 4/4: Loading to database", "PIPELINE")
            if self.db_loader is None:
                self.db_loader = DatabaseLoader()
            
            self.db_loader.load_tracks(df, batch_size=1000)
        
        success("Tracks ETL Pipeline completed successfully!", "PIPELINE")
        neutral("="*60, "PIPELINE")
        
        return df
    
    def run_interactions_pipeline(
        self,
        input_file: str,
        load_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Run ETL pipeline for interactions data.
        
        Args:
            input_file: Path to raw interactions file.
            load_to_db: Whether to load data to database.
            
        Returns:
            Processed DataFrame.
        """
        neutral("="*60, "PIPELINE")
        neutral("Starting Interactions ETL Pipeline", "PIPELINE")
        neutral("="*60, "PIPELINE")
        
        # Step 1: Ingest
        neutral("Step 1/2: Ingesting interactions data", "PIPELINE")
        df = self.ingester.ingest_interactions(input_file)
        
        # Save processed interactions
        output_file = "interactions_processed.csv"
        self.ingester.save_processed(df, output_file, self.processed_path)
        
        # Step 2: Load to database (optional)
        if load_to_db:
            neutral("Step 2/2: Loading to database", "PIPELINE")
            if self.db_loader is None:
                self.db_loader = DatabaseLoader()
            
            self.db_loader.load_interactions(df, batch_size=1000)
        
        success("Interactions ETL Pipeline completed successfully!", "PIPELINE")
        neutral("="*60, "PIPELINE")
        
        return df
    
    def initialize_database(self, schema_file: str = "./api/db/schema.sql") -> None:
        """
        Initialize database schema.
        
        Args:
            schema_file: Path to SQL schema file.
        """
        neutral("Initializing database schema", "PIPELINE")
        
        if self.db_loader is None:
            self.db_loader = DatabaseLoader()
        
        self.db_loader.initialize_schema(schema_file)
        success("Database schema initialized", "PIPELINE")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.db_loader:
            self.db_loader.close()
        neutral("Pipeline resources cleaned up", "PIPELINE")


def main():
    """
    Main entry point for ETL pipeline.
    
    Usage:
        python -m etl.pipeline
    """
    setup_logging()
    
    try:
        pipeline = ETLPipeline()
        
        # Initialize database
        neutral("Initializing database...", "MAIN")
        pipeline.initialize_database()
        
        # Run tracks pipeline
        tracks_file = "./data/raw/spotify_millsongdata.csv"
        if Path(tracks_file).exists():
            df_tracks = pipeline.run_tracks_pipeline(
                input_file=tracks_file,
                load_to_db=True,
                generate_embeddings=False  # Set to True if you want embeddings
            )
            neutral(f"Processed {len(df_tracks):,} tracks", "MAIN")
        else:
            warning(f"Tracks file not found: {tracks_file}", "MAIN")
        
        # Cleanup
        pipeline.cleanup()
        
        success("ETL Pipeline completed successfully!", "MAIN")
        
    except Exception as e:
        error(f"Pipeline failed: {str(e)}", "MAIN")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
