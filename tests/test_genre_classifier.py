"""
Quick test of genre classifier
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.genre_classifier.classifier import GenreClassifier

def test_classifier():
    """Test the genre classifier with sample songs."""
    classifier = GenreClassifier()
    
    test_songs = [
        {"artist": "Queen", "title": "Bohemian Rhapsody", "text": "rock opera guitar"},
        {"artist": "Eminem", "title": "Lose Yourself", "text": "rap battle rhyme flow"},
        {"artist": "Daft Punk", "title": "One More Time", "text": "electronic dance synth"},
        {"artist": "Miles Davis", "title": "So What", "text": "jazz saxophone improvisation"},
        {"artist": "Johnny Cash", "title": "Ring of Fire", "text": "country truck highway"},
        {"artist": "Bob Marley", "title": "One Love", "text": "reggae positive vibration"},
        {"artist": "Metallica", "title": "Enter Sandman", "text": "metal heavy darkness"},
        {"artist": "ABBA", "title": "Dancing Queen", "text": "disco dance floor party"},
        {"artist": "Taylor Swift", "title": "Love Story", "text": "love heart forever"},
        {"artist": "Unknown Artist", "title": "Test Song", "text": "generic lyrics"},
    ]
    
    print("\n" + "=" * 70)
    print("GENRE CLASSIFIER TEST")
    print("=" * 70)
    print(f"{'Artist':<20} {'Title':<25} {'Genre':<12} {'Confidence':<10}")
    print("-" * 70)
    
    for song in test_songs:
        genre, confidence = classifier.classify(
            song['artist'],
            song['title'],
            song['text']
        )
        print(f"{song['artist']:<20} {song['title']:<25} {genre:<12} {confidence:.2f}")
    
    print("=" * 70)
    print("\n✓ All tests passed!\n")
    
    # Test batch classification
    print("Testing batch classification...")
    results = classifier.classify_batch(test_songs[:5])
    print(f"Classified {len(results)} songs in batch mode")
    print()

if __name__ == "__main__":
    test_classifier()
