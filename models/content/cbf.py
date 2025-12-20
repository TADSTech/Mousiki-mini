"""
Production-grade Content-Based Filtering (CBF) Recommender for Mousiki.

This module provides a robust CBF system that:
- Checks for existing embeddings, generates if missing
- Optimized for CPU with GPU acceleration support
- Manages embedding persistence and versioning
- Computes similarity and provides recommendations
- Includes performance monitoring and graceful error handling
"""

import hashlib
import json
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import error, neutral, success, warning


class CBFError(Exception):
    """Base exception for CBF operations."""
    pass


class EmbeddingError(CBFError):
    """Raised when embedding operations fail."""
    pass


class DataError(CBFError):
    """Raised when data loading/validation fails."""
    pass


class RecommendationError(CBFError):
    """Raised when recommendation generation fails."""
    pass


class ContentBasedFilterer:
    """
    Production-ready Content-Based Filtering recommender using embeddings.
    
    Features:
    - Lazy loading of embeddings (checks for existing)
    - CPU-optimized with automatic GPU detection
    - Sparse matrix similarity computation (memory efficient)
    - Batch processing for large datasets
    - Persistent embedding cache with versioning
    - Comprehensive error handling and logging
    """

    # Configuration
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDINGS_DIR = Path("./data/embeddings")
    MODEL_CACHE_DIR = Path("./models/content/sentence_transformer_model")
    CBF_MODEL_PATH = Path("./models/content/cbf_model.pkl")
    EMBEDDINGS_INDEX_PATH = Path("./models/content/embeddings_index.json")
    BATCH_SIZE = 64  # Optimized for CPU
    GPU_BATCH_SIZE = 256  # Larger batch for GPU
    MIN_SIMILARITY = 0.0  # Include all similarities (filter in app)

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "auto",
        use_sparse: bool = True,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the Content-Based Filtering recommender.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu', 'cuda', or 'auto' (auto-detect GPU).
            use_sparse: Use sparse matrices for similarity computation.
            normalize_embeddings: L2-normalize embeddings.
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.use_sparse = use_sparse
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = self.GPU_BATCH_SIZE if self.device == "cuda" else self.BATCH_SIZE
        
        self.embedder: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[Union[np.ndarray, csr_matrix]] = None
        self.track_ids: Optional[np.ndarray] = None
        self.track_index: Optional[Dict[int, int]] = None  # track_id -> index mapping
        
        neutral(f"CBF initialized (device={self.device}, batch_size={self.batch_size})", "CBF")

    def _detect_device(self, device: str) -> str:
        """Auto-detect GPU availability if needed."""
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    gpu_info = torch.cuda.get_device_name(0)
                    success(f"GPU detected: {gpu_info}", "CBF")
                else:
                    neutral("Using CPU (no GPU detected)", "CBF")
            except ImportError:
                device = "cpu"
                neutral("PyTorch not available, using CPU", "CBF")
        return device

    def load_embedder(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            if self.embedder is not None:
                neutral("Embedder already loaded", "CBF")
                return
            
            neutral(f"Loading embedding model '{self.model_name}' on {self.device}...", "CBF")
            start_time = time.time()
            
            self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            self.embedder = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.MODEL_CACHE_DIR),
                device=self.device
            )
            
            elapsed = time.time() - start_time
            dim = self.embedder.get_sentence_embedding_dimension()
            success(f"Embedder loaded in {elapsed:.2f}s (dim: {dim})", "CBF")
            
        except Exception as e:
            error(f"Failed to load embedder: {str(e)}", "CBF")
            raise EmbeddingError(f"Could not load model {self.model_name}") from e

    def check_embeddings_exist(self) -> Optional[Path]:
        """Check if embeddings file already exists."""
        if not self.EMBEDDINGS_INDEX_PATH.exists():
            return None
        
        try:
            with open(self.EMBEDDINGS_INDEX_PATH, 'r') as f:
                index = json.load(f)
                embeddings_file = Path(index.get("embeddings_file"))
                if embeddings_file.exists():
                    neutral(f"Found existing embeddings: {embeddings_file.name}", "CBF")
                    return embeddings_file
        except Exception as e:
            warning(f"Could not read embeddings index: {str(e)}", "CBF")
        
        return None

    def load_embeddings_from_file(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings from pickle file."""
        try:
            neutral(f"Loading embeddings from {filepath.name}...", "CBF")
            start_time = time.time()
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                embeddings = data['embeddings']
                track_ids = data['track_ids']
            
            elapsed = time.time() - start_time
            success(
                f"Loaded {len(embeddings):,} embeddings in {elapsed:.2f}s "
                f"(shape: {embeddings.shape})",
                "CBF"
            )
            
            return embeddings, track_ids
            
        except Exception as e:
            error(f"Failed to load embeddings: {str(e)}", "CBF")
            raise EmbeddingError("Could not load embeddings from file") from e

    def generate_embeddings_from_db(
        self,
        db_connection_string: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate embeddings for all tracks in the database.
        
        Args:
            db_connection_string: PostgreSQL connection string.
            
        Returns:
            Tuple of (embeddings, track_ids, track_titles).
        """
        try:
            import psycopg2
            
            neutral("Connecting to database to fetch tracks...", "CBF")
            conn = psycopg2.connect(db_connection_string)
            cursor = conn.cursor()
            
            # Fetch all tracks with their available data
            cursor.execute("""
                SELECT 
                    track_id, 
                    title, 
                    artist, 
                    album, 
                    genre,
                    year
                FROM tracks
                ORDER BY track_id
            """)
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                raise DataError("No tracks found in database")
            
            neutral(f"Fetched {len(rows):,} tracks from database", "CBF")
            
            # Prepare texts: combine title, artist, album, genre
            texts = []
            track_ids = []
            track_titles = []
            
            for row in rows:
                track_id, title, artist, album, genre, year = row
                
                track_ids.append(track_id)
                track_titles.append(title)
                
                # Create rich text representation
                text_parts = [
                    title or "",
                    artist or "",
                    album or "",
                    genre or "",
                ]
                
                # Add year if available
                if year:
                    text_parts.append(str(year))
                
                text = " ".join([p for p in text_parts if p])
                texts.append(text if text else title or f"Track {track_id}")
            
            # Generate embeddings in batches
            neutral(f"Generating embeddings for {len(texts):,} tracks (batch_size={self.batch_size})...", "CBF")
            start_time = time.time()
            
            if self.embedder is None:
                self.load_embedder()
            
            embeddings = self.embedder.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                device=self.device
            )
            
            elapsed = time.time() - start_time
            success(
                f"Generated embeddings in {elapsed:.2f}s "
                f"(shape: {embeddings.shape}, avg: {elapsed/len(texts):.3f}s/track)",
                "CBF"
            )
            
            return embeddings, np.array(track_ids), track_titles
            
        except Exception as e:
            error(f"Failed to generate embeddings: {str(e)}", "CBF")
            traceback.print_exc()
            raise EmbeddingError("Could not generate embeddings from database") from e

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        track_ids: np.ndarray,
        track_titles: Optional[List[str]] = None
    ) -> Path:
        """Save embeddings to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()[:12]
            
            self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            embeddings_file = self.EMBEDDINGS_DIR / f"cbf_embeddings_{timestamp}_{content_hash}.pkl"
            
            neutral(f"Saving embeddings to {embeddings_file.name}...", "CBF")
            
            # Save as temporary file first (atomic write)
            temp_file = self.EMBEDDINGS_DIR / f".tmp_{embeddings_file.name}"
            
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'track_ids': track_ids,
                    'track_titles': track_titles,
                    'timestamp': timestamp,
                    'model': self.model_name,
                    'device': self.device,
                    'hash': content_hash
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_file.rename(embeddings_file)
            
            # Update index
            self.EMBEDDINGS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.EMBEDDINGS_INDEX_PATH, 'w') as f:
                json.dump({
                    'embeddings_file': str(embeddings_file),
                    'timestamp': timestamp,
                    'hash': content_hash,
                    'count': len(embeddings),
                    'dimension': embeddings.shape[1] if len(embeddings.shape) > 1 else 1
                }, f, indent=2)
            
            success(f"Saved {len(embeddings):,} embeddings to {embeddings_file.name}", "CBF")
            return embeddings_file
            
        except Exception as e:
            error(f"Failed to save embeddings: {str(e)}", "CBF")
            raise EmbeddingError("Could not save embeddings") from e

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> Union[np.ndarray, csr_matrix]:
        """Compute pairwise similarity matrix."""
        try:
            neutral(f"Computing similarity matrix ({len(embeddings):,} tracks)...", "CBF")
            start_time = time.time()
            
            # Compute cosine similarity
            similarities = cosine_similarity(embeddings)
            
            # Convert to sparse if enabled (memory efficient for large datasets)
            if self.use_sparse:
                similarities = csr_matrix(similarities)
                neutral(f"Converted to sparse matrix (memory usage ~{similarities.data.nbytes / 1024 / 1024:.1f}MB)", "CBF")
            
            elapsed = time.time() - start_time
            success(f"Similarity matrix computed in {elapsed:.2f}s", "CBF")
            
            return similarities
            
        except Exception as e:
            error(f"Failed to compute similarity matrix: {str(e)}", "CBF")
            raise RecommendationError("Could not compute similarities") from e

    def build_index(self, track_ids: np.ndarray) -> Dict[int, int]:
        """Build track_id -> index mapping."""
        return {int(tid): idx for idx, tid in enumerate(track_ids)}

    def recommend(
        self,
        track_id: int,
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get content-based recommendations for a track.
        
        Args:
            track_id: ID of the query track.
            n_recommendations: Number of recommendations to return.
            min_similarity: Minimum similarity threshold.
            
        Returns:
            List of (track_id, similarity_score) tuples.
        """
        try:
            if self.track_index is None or self.similarity_matrix is None:
                raise RecommendationError("CBF model not initialized. Call load_or_build() first.")
            
            if track_id not in self.track_index:
                raise DataError(f"Track {track_id} not found in index")
            
            track_idx = self.track_index[track_id]
            
            # Get similarity row
            if self.use_sparse:
                similarities = self.similarity_matrix.getrow(track_idx).toarray().flatten()
            else:
                similarities = self.similarity_matrix[track_idx]
            
            # Sort by similarity (excluding the track itself)
            sorted_indices = np.argsort(-similarities)
            
            recommendations = []
            for idx in sorted_indices:
                if idx == track_idx:  # Skip the track itself
                    continue
                
                sim_score = float(similarities[idx])
                
                if sim_score < min_similarity:
                    break
                
                recommendations.append({
                    'track_id': int(self.track_ids[idx]),
                    'similarity_score': sim_score
                })
                
                if len(recommendations) >= n_recommendations:
                    break
            
            return recommendations
            
        except Exception as e:
            error(f"Recommendation generation failed: {str(e)}", "CBF")
            raise RecommendationError("Could not generate recommendations") from e

    def load_or_build(self, db_connection_string: Optional[str] = None) -> None:
        """
        Load existing embeddings or build new ones from database.
        
        Args:
            db_connection_string: PostgreSQL connection string (needed if building new).
        """
        try:
            # Check for existing embeddings
            embeddings_file = self.check_embeddings_exist()
            
            if embeddings_file:
                self.embeddings, self.track_ids = self.load_embeddings_from_file(embeddings_file)
            else:
                if db_connection_string is None:
                    # Try to build connection string from config
                    try:
                        from utils.config import settings
                        db_connection_string = (
                            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
                            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
                        )
                    except ImportError:
                        raise DataError("db_connection_string required when embeddings don't exist")
                
                self.load_embedder()
                embeddings, track_ids, track_titles = self.generate_embeddings_from_db(db_connection_string)
                self.embeddings = embeddings
                self.track_ids = track_ids
                self.save_embeddings(embeddings, track_ids, track_titles)
            
            # Build indices and compute similarities
            self.track_index = self.build_index(self.track_ids)
            self.similarity_matrix = self.compute_similarity_matrix(self.embeddings)
            
            success(f"CBF model ready ({len(self.track_ids):,} tracks)", "CBF")
            
        except Exception as e:
            error(f"Failed to load or build CBF model: {str(e)}", "CBF")
            traceback.print_exc()
            raise

    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """Save the CBF model to disk."""
        try:
            filepath = filepath or self.CBF_MODEL_PATH
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            neutral(f"Saving CBF model to {filepath.name}...", "CBF")
            
            # Save as temporary file first
            temp_file = filepath.parent / f".tmp_{filepath.name}"
            
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'similarity_matrix': self.similarity_matrix,
                    'track_ids': self.track_ids,
                    'track_index': self.track_index,
                    'model_name': self.model_name,
                    'device': self.device,
                    'normalize_embeddings': self.normalize_embeddings,
                    'timestamp': datetime.now().isoformat()
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            temp_file.rename(filepath)
            success(f"Model saved to {filepath.name}", "CBF")
            return filepath
            
        except Exception as e:
            error(f"Failed to save CBF model: {str(e)}", "CBF")
            raise CBFError("Could not save model") from e

    def load_model(self, filepath: Optional[Path] = None) -> None:
        """Load a previously saved CBF model."""
        try:
            filepath = filepath or self.CBF_MODEL_PATH
            
            if not filepath.exists():
                raise DataError(f"Model file not found: {filepath}")
            
            neutral(f"Loading CBF model from {filepath.name}...", "CBF")
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.similarity_matrix = data['similarity_matrix']
                self.track_ids = data['track_ids']
                self.track_index = data['track_index']
                self.model_name = data['model_name']
            
            success(f"Model loaded with {len(self.track_ids):,} tracks", "CBF")
            
        except Exception as e:
            error(f"Failed to load CBF model: {str(e)}", "CBF")
            raise CBFError("Could not load model") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the CBF model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'n_tracks': len(self.track_ids) if self.track_ids is not None else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None and len(self.embeddings.shape) > 1 else 0,
            'similarity_matrix_type': 'sparse' if self.use_sparse else 'dense',
            'has_model': self.embedder is not None,
            'is_built': self.similarity_matrix is not None
        }
