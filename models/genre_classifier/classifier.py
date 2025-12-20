"""
Genre Classifier for Mousiki

Uses keyword-based and text analysis to classify songs into 15 genres.
Combines artist names, song titles, and lyrics for classification.
"""

import re
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GenreClassifier:
    """
    Keyword-based genre classifier with fallback strategies.
    
    Classification strategy:
    1. Artist-based classification (known artists)
    2. Keyword matching in lyrics/title
    3. Musical term detection
    4. Fallback to 'Pop' (most common genre)
    """
    
    GENRES = [
        'Rock', 'Pop', 'Hip-Hop', 'Jazz', 'Electronic',
        'Classical', 'R&B', 'Country', 'Reggae', 'K-Pop',
        'Indie', 'Metal', 'Soul', 'Folk', 'Disco'
    ]
    
    # Artist-genre mappings (known artists)
    ARTIST_GENRES = {
        # Rock
        'queen': 'Rock', 'beatles': 'Rock', 'led zeppelin': 'Rock', 'pink floyd': 'Rock',
        'rolling stones': 'Rock', 'nirvana': 'Rock', 'radiohead': 'Rock', 'u2': 'Rock',
        'ac dc': 'Rock', 'acdc': 'Rock', 'guns n roses': 'Rock', 'aerosmith': 'Rock',
        
        # Metal
        'metallica': 'Metal', 'iron maiden': 'Metal', 'black sabbath': 'Metal',
        'slayer': 'Metal', 'megadeth': 'Metal', 'pantera': 'Metal', 'judas priest': 'Metal',
        'motorhead': 'Metal', 'slipknot': 'Metal', 'system of a down': 'Metal',
        
        # Pop
        'michael jackson': 'Pop', 'madonna': 'Pop', 'britney spears': 'Pop',
        'justin bieber': 'Pop', 'taylor swift': 'Pop', 'ariana grande': 'Pop',
        'katy perry': 'Pop', 'lady gaga': 'Pop', 'beyonce': 'Pop', 'rihanna': 'Pop',
        'ed sheeran': 'Pop', 'adele': 'Pop', 'maroon 5': 'Pop', 'bruno mars': 'Pop',
        
        # Hip-Hop
        'eminem': 'Hip-Hop', 'jay z': 'Hip-Hop', 'drake': 'Hip-Hop', 'kanye west': 'Hip-Hop',
        'tupac': 'Hip-Hop', '2pac': 'Hip-Hop', 'biggie': 'Hip-Hop', 'nas': 'Hip-Hop',
        'kendrick lamar': 'Hip-Hop', 'snoop dogg': 'Hip-Hop', 'dr dre': 'Hip-Hop',
        '50 cent': 'Hip-Hop', 'nicki minaj': 'Hip-Hop', 'cardi b': 'Hip-Hop',
        
        # Jazz
        'miles davis': 'Jazz', 'john coltrane': 'Jazz', 'louis armstrong': 'Jazz',
        'duke ellington': 'Jazz', 'ella fitzgerald': 'Jazz', 'billie holiday': 'Jazz',
        'charlie parker': 'Jazz', 'dave brubeck': 'Jazz', 'thelonious monk': 'Jazz',
        
        # Electronic
        'daft punk': 'Electronic', 'deadmau5': 'Electronic', 'skrillex': 'Electronic',
        'calvin harris': 'Electronic', 'david guetta': 'Electronic', 'avicii': 'Electronic',
        'tiesto': 'Electronic', 'armin van buuren': 'Electronic', 'marshmello': 'Electronic',
        'the chemical brothers': 'Electronic', 'kraftwerk': 'Electronic',
        
        # Classical
        'mozart': 'Classical', 'beethoven': 'Classical', 'bach': 'Classical',
        'chopin': 'Classical', 'tchaikovsky': 'Classical', 'vivaldi': 'Classical',
        
        # R&B
        'stevie wonder': 'R&B', 'marvin gaye': 'R&B', 'alicia keys': 'R&B',
        'usher': 'R&B', 'john legend': 'R&B', 'the weeknd': 'R&B',
        
        # Country
        'johnny cash': 'Country', 'dolly parton': 'Country', 'willie nelson': 'Country',
        'garth brooks': 'Country', 'shania twain': 'Country', 'carrie underwood': 'Country',
        
        # Reggae
        'bob marley': 'Reggae', 'peter tosh': 'Reggae', 'jimmy cliff': 'Reggae',
        
        # K-Pop
        'bts': 'K-Pop', 'blackpink': 'K-Pop', 'exo': 'K-Pop', 'twice': 'K-Pop',
        
        # Indie
        'arcade fire': 'Indie', 'vampire weekend': 'Indie', 'the strokes': 'Indie',
        'arctic monkeys': 'Indie', 'tame impala': 'Indie',
        
        # Soul
        'aretha franklin': 'Soul', 'otis redding': 'Soul', 'sam cooke': 'Soul',
        
        # Folk
        'bob dylan': 'Folk', 'joan baez': 'Folk', 'simon garfunkel': 'Folk',
        
        # Disco
        'bee gees': 'Disco', 'donna summer': 'Disco', 'chic': 'Disco',
    }
    
    # Keyword patterns for genre detection
    GENRE_KEYWORDS = {
        'Rock': [
            r'\brock\b', r'\bguitar\b', r'\briff\b', r'\band\b.*\bband\b',
            r'\bconcert\b', r'\bamplifier\b', r'\belectric\b'
        ],
        'Metal': [
            r'\bmetal\b', r'\bheavy\b', r'\bscream\b', r'\bbrutal\b',
            r'\bdeath\b', r'\bdarkness\b', r'\bfire\b.*\bhell\b'
        ],
        'Pop': [
            r'\blove\b', r'\bheart\b', r'\bdance\b', r'\bparty\b',
            r'\bfeel\b', r'\bhappy\b', r'\bcelebrate\b'
        ],
        'Hip-Hop': [
            r'\brap\b', r'\bhip[\s-]?hop\b', r'\bbeat\b', r'\brhyme\b',
            r'\bflow\b', r'\bstreet\b', r'\bhustle\b', r'\bmoney\b'
        ],
        'Jazz': [
            r'\bjazz\b', r'\bswing\b', r'\bblue\s+note\b', r'\bimprovisation\b',
            r'\bsaxophone\b', r'\btrumpet\b', r'\bpiano\b.*\bsolo\b'
        ],
        'Electronic': [
            r'\belectronic\b', r'\bedm\b', r'\btechno\b', r'\bhouse\b',
            r'\bdubstep\b', r'\bsynth\b', r'\bdrop\b', r'\bbass\b'
        ],
        'Classical': [
            r'\bclassical\b', r'\bsymphony\b', r'\borchestra\b', r'\bconcerto\b',
            r'\bopera\b', r'\bstring\s+quartet\b'
        ],
        'R&B': [
            r'\br&b\b', r'\brhythm\b.*\bblues\b', r'\bsoul\b', r'\bsmooth\b',
            r'\bgroove\b', r'\bfunk\b'
        ],
        'Country': [
            r'\bcountry\b', r'\btruck\b', r'\bhighway\b', r'\btexas\b',
            r'\bcowboy\b', r'\bhome\b.*\btown\b', r'\bfarm\b'
        ],
        'Reggae': [
            r'\breggae\b', r'\brasta\b', r'\bjamaica\b', r'\bone\s+love\b',
            r'\bpositive\b.*\bvibration\b'
        ],
        'K-Pop': [
            r'\bk[\s-]?pop\b', r'\bkorean\b', r'\bseoul\b', r'\boppa\b',
            r'\bhangul\b'
        ],
        'Indie': [
            r'\bindie\b', r'\balternative\b', r'\bunderground\b', r'\bindependent\b',
            r'\bart\b.*\brock\b'
        ],
        'Soul': [
            r'\bsoul\b', r'\bspiritual\b', r'\bchurch\b', r'\bgospel\b',
            r'\bpraise\b', r'\bhallelujah\b'
        ],
        'Folk': [
            r'\bfolk\b', r'\bacoustic\b', r'\btraditional\b', r'\bballad\b',
            r'\bstorytelling\b'
        ],
        'Disco': [
            r'\bdisco\b', r'\bgroove\b', r'\bdance\s+floor\b', r'\b70s\b',
            r'\bfunky\b', r'\bnight\s+fever\b'
        ],
    }
    
    def __init__(self):
        """Initialize the genre classifier."""
        self.artist_genres_lower = {
            k.lower(): v for k, v in self.ARTIST_GENRES.items()
        }
        
        # Compile regex patterns
        self.compiled_patterns = {
            genre: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for genre, patterns in self.GENRE_KEYWORDS.items()
        }
    
    def classify(
        self,
        artist: str,
        title: str = '',
        lyrics: str = '',
        confidence_threshold: float = 0.5
    ) -> Tuple[str, float]:
        """
        Classify a song into a genre.
        
        Args:
            artist: Artist name
            title: Song title
            lyrics: Song lyrics
            confidence_threshold: Minimum confidence (0-1)
            
        Returns:
            Tuple of (genre, confidence)
        """
        if not artist:
            return 'Pop', 0.3  # Default fallback
        
        artist_lower = artist.lower().strip()
        
        # Strategy 1: Check artist mapping
        if artist_lower in self.artist_genres_lower:
            return self.artist_genres_lower[artist_lower], 1.0
        
        # Strategy 2: Keyword matching
        text = f"{artist} {title} {lyrics}".lower()
        
        genre_scores = {}
        for genre, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(pattern.findall(text))
                score += matches
            
            if score > 0:
                genre_scores[genre] = score
        
        # Find genre with highest score
        if genre_scores:
            best_genre = max(genre_scores, key=genre_scores.get)
            max_score = genre_scores[best_genre]
            
            # Normalize confidence (cap at 1.0)
            confidence = min(max_score / 5.0, 1.0)
            
            if confidence >= confidence_threshold:
                return best_genre, confidence
        
        # Fallback: Pop (most common genre)
        return 'Pop', 0.3
    
    def classify_batch(
        self,
        records: List[dict],
        artist_col: str = 'artist',
        title_col: str = 'title',
        lyrics_col: str = 'text'
    ) -> List[Tuple[str, float]]:
        """
        Classify multiple songs.
        
        Args:
            records: List of dictionaries with song data
            artist_col: Column name for artist
            title_col: Column name for title
            lyrics_col: Column name for lyrics
            
        Returns:
            List of (genre, confidence) tuples
        """
        results = []
        for record in records:
            genre, confidence = self.classify(
                artist=record.get(artist_col, ''),
                title=record.get(title_col, ''),
                lyrics=record.get(lyrics_col, '')
            )
            results.append((genre, confidence))
        
        return results


def main():
    """Test the classifier."""
    classifier = GenreClassifier()
    
    test_songs = [
        {"artist": "Queen", "title": "Bohemian Rhapsody", "text": ""},
        {"artist": "Eminem", "title": "Lose Yourself", "text": "rap battle"},
        {"artist": "Daft Punk", "title": "One More Time", "text": "electronic dance"},
        {"artist": "Miles Davis", "title": "So What", "text": "jazz improvisation"},
        {"artist": "Unknown Artist", "title": "Love Song", "text": "love heart forever"},
    ]
    
    print("Genre Classification Test:")
    print("=" * 60)
    for song in test_songs:
        genre, confidence = classifier.classify(
            song['artist'],
            song['title'],
            song['text']
        )
        print(f"{song['artist']:20} -> {genre:12} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
