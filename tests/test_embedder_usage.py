"""
Test script demonstrating how to use LyricsEmbedder from main.py
"""

import pandas as pd
from models.content.embed_items import LyricsEmbedder
from utils.logging_config import setup_logging, success, neutral


def main():
    """Demonstrate LyricsEmbedder usage as it would be called from main.py"""
    
    # Setup logging
    setup_logging()
    neutral("Testing LyricsEmbedder integration", "MAIN")
    
    # Simulate loading preprocessed data
    df = pd.DataFrame({
        'artist': ['Queen', 'Eagles', 'Led Zeppelin'],
        'song': ['Bohemian Rhapsody', 'Hotel California', 'Stairway to Heaven'],
        'text_clean': [
            'rock opera multi part harmony vocal guitar piano',
            'california hotel check out never leave guitar solo',
            'stairway heaven rock progressive guitar solo'
        ]
    })
    
    neutral(f"Loaded {len(df)} tracks with cleaned text", "MAIN")
    
    # Initialize embedder (model loads once)
    embedder = LyricsEmbedder()
    
    # Generate embeddings for the cleaned text
    vecs = embedder.embed(df["text_clean"].tolist())
    
    success(f"Generated embeddings with shape: {vecs.shape}", "MAIN")
    
    # Add embeddings to dataframe
    df['embedding'] = list(vecs)
    
    print("\n" + "="*60)
    print("DataFrame with embeddings:")
    print("="*60)
    print(df[['artist', 'song']].to_string())
    print(f"\nEmbedding dimension: {len(df['embedding'].iloc[0])}")
    
    success("Integration test completed successfully!", "MAIN")


if __name__ == "__main__":
    main()
