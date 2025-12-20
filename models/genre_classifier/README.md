# Genre Classifier Module

Keyword-based genre classification for music tracks.

## Usage

```python
from models.genre_classifier.classifier import GenreClassifier

classifier = GenreClassifier()

# Classify single song
genre, confidence = classifier.classify(
    artist="Queen",
    title="Bohemian Rhapsody",
    lyrics="Is this the real life..."
)
print(f"Genre: {genre} (confidence: {confidence:.2f})")

# Classify batch
records = [
    {"artist": "Eminem", "title": "Lose Yourself", "text": "rap lyrics"},
    {"artist": "Daft Punk", "title": "Around the World", "text": "electronic"}
]
results = classifier.classify_batch(records)
```

## Supported Genres

- Rock
- Pop
- Hip-Hop
- Jazz
- Electronic
- Classical
- R&B
- Country
- Reggae
- K-Pop
- Indie
- Metal
- Soul
- Folk
- Disco

## Classification Strategy

1. **Artist Mapping**: Known artists are directly mapped to genres
2. **Keyword Matching**: Lyrics and titles are scanned for genre keywords
3. **Fallback**: Defaults to "Pop" if no match found

## Extending

Add more artists to `ARTIST_GENRES` or keywords to `GENRE_KEYWORDS` in `classifier.py`.
