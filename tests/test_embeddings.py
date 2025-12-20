"""
Tests for the production embedding service.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Adjust path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.embedder.embedder import (
    LyricsEmbedder, 
    DataFrameEmbedderAdapter, 
    ValidationError, 
    PersistenceError
)


class TestLyricsEmbedder(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize embedder once for all tests."""
        cls.test_dir = tempfile.mkdtemp()
        cls.embedder = LyricsEmbedder(
            cache_dir=cls.test_dir,
            device="cpu"
        )
        # Override embeddings dir for testing
        cls.embedder.EMBEDDINGS_DIR = cls.test_dir

    @classmethod
    def tearDownClass(cls):
        """Cleanup temporary directory."""
        shutil.rmtree(cls.test_dir)

    def test_embedding_shape(self):
        """Test that embeddings have correct shape and dimension."""
        texts = ["Hello world", "Testing embeddings"]
        embeddings = self.embedder(texts)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 384)  # MiniLM dimension

    def test_empty_input(self):
        """Test handling of empty inputs."""
        result = self.embedder([])
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_invalid_input(self):
        """Test validation of input types."""
        with self.assertRaises(ValidationError):
            self.embedder([1, 2, 3])  # type: ignore

    def test_persistence(self):
        """Test saving and loading embeddings."""
        texts = ["Persist me"]
        embeddings = self.embedder(texts)
        ids = ["id_1"]
        
        # Save
        path = self.embedder.save(embeddings, ids, filename="test_persist")
        self.assertTrue(path.exists())
        
        # Load
        data = self.embedder.load(path)
        self.assertTrue(np.array_equal(data["embeddings"], embeddings))
        self.assertEqual(data["ids"], ids)
        self.assertEqual(data["hash"], data["hash"])  # Just checking key exists

    def test_dataframe_adapter(self):
        """Test the DataFrame adapter."""
        df = pd.DataFrame({
            "id": [1, 2],
            "text": ["Text A", "Text B"]
        })
        
        adapter = DataFrameEmbedderAdapter(self.embedder)
        result_df = adapter.embed_dataframe(df, text_column="text", id_column="id")
        
        self.assertEqual(len(result_df), 2)
        self.assertIn("embedding", result_df.columns)
        self.assertIn("id", result_df.columns)
        self.assertEqual(len(result_df.iloc[0]["embedding"]), 384)

    def test_deterministic_output(self):
        """Test that the same input produces the same output."""
        text = "Deterministic test"
        emb1 = self.embedder([text])
        emb2 = self.embedder([text])
        self.assertTrue(np.allclose(emb1, emb2))


if __name__ == "__main__":
    unittest.main()
