"""
Production-grade Data Ingestion for Mousiki.

Handles robust ingestion of music data with validation, error handling,
and structured logging.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import error, neutral, success, warning


class DataIngestionError(Exception):
    """Base exception for data ingestion errors."""
    pass


class DataIngester:
    """
    Production-ready data ingestion service.
    
    Features:
    - Multiple format support (CSV, JSON, Parquet)
    - Schema validation
    - Error handling with detailed logging
    - Memory-efficient chunked reading for large files
    """
    
    SUPPORTED_FORMATS = ["csv", "json", "parquet"]
    
    def __init__(self, raw_data_path: str = "./data/raw"):
        """
        Initialize the data ingester.
        
        Args:
            raw_data_path: Directory containing raw data files.
        """
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        neutral(f"DataIngester initialized with path: {self.raw_data_path}", "INGEST")
    
    def ingest_file(
        self, 
        filepath: Union[str, Path], 
        format: Optional[str] = None,
        chunksize: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Ingest data from a file with automatic format detection.
        
        Args:
            filepath: Path to the file.
            format: File format (csv, json, parquet). Auto-detected if None.
            chunksize: Read in chunks for large files (CSV only).
            **kwargs: Additional arguments passed to pandas reader.
            
        Returns:
            DataFrame containing the ingested data.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            error(f"File not found: {filepath}", "INGEST")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Auto-detect format
        if format is None:
            format = filepath.suffix.lstrip('.').lower()
        
        if format not in self.SUPPORTED_FORMATS:
            error(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}", "INGEST")
            raise ValueError(f"Unsupported format: {format}")
        
        neutral(f"Ingesting {format.upper()} file: {filepath}", "INGEST")
        
        try:
            if format == "csv":
                if chunksize:
                    neutral(f"Reading in chunks of {chunksize:,} rows", "INGEST")
                    chunks = pd.read_csv(filepath, chunksize=chunksize, **kwargs)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(filepath, **kwargs)
                    
            elif format == "json":
                df = pd.read_json(filepath, **kwargs)
                
            elif format == "parquet":
                df = pd.read_parquet(filepath, **kwargs)
            
            success(f"Successfully ingested {len(df):,} records", "INGEST")
            return df
            
        except Exception as e:
            error(f"Failed to ingest file: {str(e)}", "INGEST")
            raise DataIngestionError(f"Ingestion failed for {filepath}") from e
    
    def validate_schema(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate.
            required_columns: List of required column names.
            optional_columns: List of optional column names.
            
        Returns:
            True if valid.
            
        Raises:
            DataIngestionError: If validation fails.
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            error(f"Missing required columns: {missing_cols}", "INGEST")
            raise DataIngestionError(f"Missing required columns: {missing_cols}")
        
        # Log optional columns that are present
        if optional_columns:
            present_optional = [col for col in optional_columns if col in df.columns]
            if present_optional:
                neutral(f"Found optional columns: {present_optional}", "INGEST")
        
        success("Schema validation passed", "INGEST")
        return True
    
    def ingest_tracks(
        self, 
        filepath: Union[str, Path],
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Ingest track metadata with validation.
        
        Expected columns: artist, song, link, text (for Spotify dataset)
        Or: track_id, title, artist, album, genre, duration, year
        
        Args:
            filepath: Path to tracks file.
            format: File format.
            
        Returns:
            Validated DataFrame with track information.
        """
        df = self.ingest_file(filepath, format=format)
        
        # Handle different schema formats
        if 'song' in df.columns and 'title' not in df.columns:
            # Spotify dataset format
            required = ["artist", "song"]
            self.validate_schema(df, required, ["link", "text"])
            
            # Normalize column names
            df = df.rename(columns={"song": "title"})
            neutral("Normalized 'song' column to 'title'", "INGEST")
            
        else:
            # Standard format
            required = ["title", "artist"]
            optional = ["track_id", "album", "genre", "duration", "year"]
            self.validate_schema(df, required, optional)
        
        return df
    
    def ingest_interactions(
        self, 
        filepath: Union[str, Path],
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Ingest user-track interactions with validation.
        
        Expected columns: user_id, track_id, interaction_type, timestamp, duration
        
        Args:
            filepath: Path to interactions file.
            format: File format.
            
        Returns:
            Validated DataFrame with interaction data.
        """
        df = self.ingest_file(filepath, format=format)
        
        required = ["user_id", "track_id", "interaction_type"]
        optional = ["timestamp", "duration", "track_duration"]
        self.validate_schema(df, required, optional)
        
        # Parse timestamp if present
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                neutral("Parsed timestamp column", "INGEST")
            except Exception as e:
                warning(f"Failed to parse timestamp: {e}", "INGEST")
        
        return df
    
    def save_processed(
        self, 
        df: pd.DataFrame, 
        filename: str,
        processed_path: str = "./data/processed",
        format: str = "csv"
    ) -> Path:
        """
        Save processed data with versioning.
        
        Args:
            df: DataFrame to save.
            filename: Output filename (without extension).
            processed_path: Directory for processed data.
            format: Output format (csv, parquet).
            
        Returns:
            Path to saved file.
        """
        processed_dir = Path(processed_path)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Add extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        output_path = processed_dir / filename
        
        try:
            neutral(f"Saving {len(df):,} records to {output_path}", "INGEST")
            
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {format}")
            
            success(f"Saved to {output_path}", "INGEST")
            return output_path
            
        except Exception as e:
            error(f"Failed to save file: {str(e)}", "INGEST")
            raise DataIngestionError(f"Save failed for {output_path}") from e
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate data summary statistics.
        
        Args:
            df: DataFrame to summarize.
            
        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        return summary
