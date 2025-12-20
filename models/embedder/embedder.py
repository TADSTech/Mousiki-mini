"""
Production-grade Embedding Service for Mousiki.

This module provides a robust, callable service for generating, managing, and persisting
text embeddings using Sentence Transformers. It is designed for CPU-based inference
in production environments.
"""

import hashlib
import json
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import error, neutral, success, warning


class EmbedderError(Exception):
    """Base exception for embedding operations."""
    pass


class ModelLoadError(EmbedderError):
    """Raised when the model fails to load."""
    pass


class EmbeddingGenerationError(EmbedderError):
    """Raised when embedding generation fails."""
    pass


class PersistenceError(EmbedderError):
    """Raised when saving or loading embeddings fails."""
    pass


class ValidationError(EmbedderError):
    """Raised when input validation fails."""
    pass


class LyricsEmbedder:
    """
    Production-ready service for generating text embeddings.
    
    Features:
    - Singleton-like model loading
    - Atomic persistence with versioning
    - Input validation and safety checks
    - CPU-optimized execution
    - Callable interface
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_CACHE_DIR = "./models/content/sentence_transformer_model"
    EMBEDDINGS_DIR = "../../data/embeddings"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: str = "cpu",
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: HuggingFace model identifier.
            cache_dir: Local directory for model caching.
            device: Device to run the model on ('cpu' or 'cuda').
            normalize_embeddings: Whether to L2-normalize outputs.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model: Optional[SentenceTransformer] = None
        
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model with error handling."""
        try:
            neutral(f"Loading model '{self.model_name}' on {self.device}...", "EMBEDDER")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device
            )
            elapsed = time.time() - start_time
            
            dim = self.model.get_sentence_embedding_dimension()
            success(f"Model loaded in {elapsed:.2f}s (dim: {dim})", "EMBEDDER")
            
        except Exception as e:
            error(f"Failed to load model: {str(e)}", "EMBEDDER")
            raise ModelLoadError(f"Could not load model {self.model_name}") from e

    def __call__(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Callable interface for embedding generation.

        Args:
            texts: Single string or list of strings to embed.
            batch_size: Batch size for processing.

        Returns:
            Numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.embed(texts, batch_size=batch_size)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size for processing.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            warning("Received empty input list. Returning empty array.", "EMBEDDER")
            return np.array([])

        # Validation
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValidationError("Input must be a list of strings.")

        try:
            neutral(f"Generating embeddings for {len(texts):,} items (batch_size={batch_size})", "EMBEDDER")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                device=self.device
            )
            
            return embeddings

        except Exception as e:
            error(f"Embedding generation failed: {str(e)}", "EMBEDDER")
            traceback.print_exc()
            raise EmbeddingGenerationError("Failed to generate embeddings") from e

    def save(
        self,
        embeddings: np.ndarray,
        ids: List[Union[str, int]],
        metadata: Optional[Dict[str, Any]] = None,
        filename: str = "embeddings"
    ) -> Path:
        """
        Persist embeddings to disk with versioning and atomic writes.

        Args:
            embeddings: Numpy array of embeddings.
            ids: List of identifiers corresponding to embeddings.
            metadata: Optional dictionary of metadata.
            filename: Base filename.

        Returns:
            Path to the saved file.
        """
        if len(embeddings) != len(ids):
            raise ValidationError(f"Length mismatch: {len(embeddings)} embeddings vs {len(ids)} IDs")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.EMBEDDINGS_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Compute hash for validation
        content_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()[:16]
        
        final_filename = f"{filename}_{timestamp}_{content_hash}.pkl"
        final_path = save_dir / final_filename
        temp_path = save_dir / f".tmp_{final_filename}"

        data = {
            "embeddings": embeddings,
            "ids": ids,
            "metadata": metadata or {},
            "model_name": self.model_name,
            "timestamp": timestamp,
            "hash": content_hash,
            "version": "1.0"
        }

        try:
            neutral(f"Saving {len(embeddings):,} embeddings to {final_path}...", "EMBEDDER")
            
            # Atomic write
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            
            shutil.move(str(temp_path), str(final_path))
            success(f"Successfully saved embeddings to {final_path}", "EMBEDDER")
            return final_path

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            error(f"Failed to save embeddings: {str(e)}", "EMBEDDER")
            raise PersistenceError("Could not save embeddings") from e

    def load(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load embeddings from disk and validate hash.

        Args:
            filepath: Path to the embeddings file.

        Returns:
            Dictionary containing embeddings, ids, and metadata.
        """
        path = Path(filepath)
        if not path.exists():
            raise PersistenceError(f"File not found: {path}")

        try:
            neutral(f"Loading embeddings from {path}...", "EMBEDDER")
            
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Validate structure
            required_keys = ["embeddings", "ids", "hash"]
            if not all(k in data for k in required_keys):
                raise PersistenceError("Invalid embedding file format")

            # Validate hash
            current_hash = hashlib.sha256(data["embeddings"].tobytes()).hexdigest()[:16]
            if current_hash != data["hash"]:
                warning(f"Hash mismatch! Stored: {data['hash']}, Computed: {current_hash}", "EMBEDDER")
            
            success(f"Loaded {len(data['embeddings']):,} embeddings", "EMBEDDER")
            return data

        except Exception as e:
            error(f"Failed to load embeddings: {str(e)}", "EMBEDDER")
            raise PersistenceError("Could not load embeddings") from e

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LyricsEmbedder":
        """Factory method for dependency injection."""
        return cls(
            model_name=config.get("model_name", cls.DEFAULT_MODEL),
            cache_dir=config.get("cache_dir", cls.DEFAULT_CACHE_DIR),
            device=config.get("device", "cpu"),
            normalize_embeddings=config.get("normalize_embeddings", True)
        )


class DataFrameEmbedderAdapter:
    """
    Adapter to use LyricsEmbedder with Pandas DataFrames.
    """
    
    def __init__(self, embedder: LyricsEmbedder):
        self.embedder = embedder

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str = "track_id"
    ) -> pd.DataFrame:
        """
        Generate embeddings for a DataFrame and return a new DataFrame with results.
        """
        if df.empty:
            warning("Empty DataFrame provided.", "ADAPTER")
            return pd.DataFrame()

        if text_column not in df.columns:
            raise ValidationError(f"Column '{text_column}' not found in DataFrame.")

        texts = df[text_column].astype(str).tolist()
        ids = df[id_column].tolist() if id_column in df.columns else list(range(len(df)))

        embeddings = self.embedder.embed(texts)
        
        # Return a new DataFrame with IDs and embeddings
        return pd.DataFrame({
            id_column: ids,
            "embedding": list(embeddings)
        })


def get_embedder(config: Optional[Dict[str, Any]] = None) -> LyricsEmbedder:
    """Convenience factory function."""
    return LyricsEmbedder.from_config(config or {})
