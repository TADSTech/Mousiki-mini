"""
Text Preprocessing Module

Handles cleaning and preprocessing of text data (track titles, artist names, lyrics).
"""

import re
import pandas as pd
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import success, warning, neutral, error


class TextPreprocessor:
    """Preprocesses text data for embedding generation."""
    
    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been'
        ])
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters except spaces and hyphens
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove common stop words from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stop words removed
        """
        words = text.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        return ' '.join(filtered_words)
    
    def create_combined_features(
        self, 
        title: str, 
        artist: str, 
        album: Optional[str] = None,
        genre: Optional[str] = None
    ) -> str:
        """
        Combine multiple text features into a single string for embedding.
        
        Args:
            title: Track title
            artist: Artist name
            album: Album name (optional)
            genre: Genre (optional)
            
        Returns:
            Combined feature string
        """
        features = []
        
        if title:
            features.append(f"title: {self.clean_text(title)}")
        
        if artist:
            features.append(f"artist: {self.clean_text(artist)}")
        
        if album and not pd.isna(album):
            features.append(f"album: {self.clean_text(album)}")
        
        if genre and not pd.isna(genre):
            features.append(f"genre: {self.clean_text(genre)}")
        
        return " | ".join(features)
    
    def preprocess_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess track dataframe.
        
        Args:
            df: DataFrame with track information
            
        Returns:
            DataFrame with preprocessed text features
        """
        try:
            neutral(f"Preprocessing {len(df)} tracks", "PREPROCESS")
            
            if df.empty:
                warning("Received empty DataFrame", "PREPROCESS")
                return df
            
            df = df.copy()
            
            # Handle different column name formats
            # Map common column variations
            if 'song' in df.columns and 'title' not in df.columns:
                df['title'] = df['song']
            if 'title' not in df.columns and 'song' not in df.columns:
                error("Dataset must have 'title' or 'song' column", "PREPROCESS")
                raise KeyError("Missing 'title' or 'song' column")
            
            if 'artist' not in df.columns:
                error("Dataset must have 'artist' column", "PREPROCESS")
                raise KeyError("Missing 'artist' column")
            
            # Clean individual fields
            df['title_clean'] = df['title'].apply(self.clean_text)
            df['artist_clean'] = df['artist'].apply(self.clean_text)
            
            # Handle optional fields
            if 'album' in df.columns:
                df['album_clean'] = df['album'].apply(self.clean_text)
            
            if 'genre' in df.columns:
                df['genre_clean'] = df['genre'].apply(self.clean_text)
            
            if 'text' in df.columns:
                neutral("Processing lyrics/text field", "PREPROCESS")
                df['text_clean'] = df['text'].apply(self.clean_text)
            
            # Create combined features for embedding
            df['combined_features'] = df.apply(
                lambda row: self.create_combined_features(
                    row.get('title', ''),
                    row.get('artist', ''),
                    row.get('album'),
                    row.get('genre')
                ),
                axis=1
            )
            
            success(f"Preprocessing complete - created {len(df.columns)} columns", "PREPROCESS")
            return df
            
        except KeyError as e:
            error(f"Missing required column: {str(e)}", "PREPROCESS")
            raise
        except Exception as e:
            error(f"Preprocessing failed: {str(e)}", "PREPROCESS")
            raise


def preprocess_text(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Main preprocessing function to load, preprocess, and save track data.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save preprocessed CSV file
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        Exception: For other processing errors
    """
    try:
        # Check if input file exists
        if not Path(input_path).exists():
            error(f"Input file not found: {input_path}", "PREPROCESS")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load data
        neutral(f"Loading data from {input_path}", "PREPROCESS")
        df = pd.read_csv(input_path)
        success(f"Loaded {len(df):,} rows", "PREPROCESS")
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Preprocess tracks
        df_preprocessed = preprocessor.preprocess_tracks(df)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed data
        neutral(f"Saving to {output_path}", "PREPROCESS")
        df_preprocessed.to_csv(output_path, index=False)
        success(f"Saved {len(df_preprocessed):,} preprocessed rows", "PREPROCESS")
        
        return df_preprocessed
        
    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError:
        error(f"Input file is empty: {input_path}", "PREPROCESS")
        raise
    except Exception as e:
        error(f"Preprocessing pipeline failed: {str(e)}", "PREPROCESS")
        raise


def main():
    sample_data = {
        'track_id': [1, 2, 3],
        'title': ['Bohemian Rhapsody', 'Stairway to Heaven', 'Hotel California'],
        'artist': ['Queen', 'Led Zeppelin', 'Eagles'],
        'album': ['A Night at the Opera', 'Led Zeppelin IV', 'Hotel California'],
        'genre': ['Rock', 'Rock', 'Rock']
    }
    
    df = pd.DataFrame(sample_data)
    
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.preprocess_tracks(df)
    
    print("\nProcessed tracks:")
    print(processed_df[['track_id', 'combined_features']])


if __name__ == "__main__":
    main()
