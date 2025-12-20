"""
Utility Tests

Tests for utility functions.
"""

import pytest
from utils.helpers import (
    generate_id,
    safe_divide,
    truncate_text,
    flatten_dict,
    batch_iterator,
    format_duration,
    calculate_percentage
)


class TestHelpers:
    """Tests for helper functions."""
    
    def test_generate_id(self):
        """Test ID generation."""
        id1 = generate_id("test")
        id2 = generate_id("test")
        id3 = generate_id("different")
        
        assert id1 == id2  # Same input = same ID
        assert id1 != id3  # Different input = different ID
        assert len(id1) == 32  # MD5 hash length
    
    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0
    
    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long text that needs to be truncated"
        truncated = truncate_text(text, max_length=20)
        
        assert len(truncated) <= 20
        assert truncated.endswith("...")
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            }
        }
        
        flattened = flatten_dict(nested)
        
        assert 'a' in flattened
        assert 'b.c' in flattened
        assert 'b.d.e' in flattened
        assert flattened['b.d.e'] == 3
    
    def test_batch_iterator(self):
        """Test batch iteration."""
        items = list(range(10))
        batches = list(batch_iterator(items, batch_size=3))
        
        assert len(batches) == 4  # 3 full batches + 1 partial
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(65) == "1:05"
        assert format_duration(225) == "3:45"
        assert format_duration(0) == "0:00"
    
    def test_calculate_percentage(self):
        """Test percentage calculation."""
        assert calculate_percentage(25, 100) == 25.0
        assert calculate_percentage(50, 200) == 25.0
        assert calculate_percentage(10, 0) == 0.0


class TestConfig:
    """Tests for configuration."""
    
    def test_settings_import(self):
        """Test settings can be imported."""
        from utils.config import settings
        
        assert settings is not None
        assert hasattr(settings, 'API_HOST')
        assert hasattr(settings, 'API_PORT')
    
    def test_default_values(self):
        """Test default configuration values."""
        from utils.config import settings
        
        assert isinstance(settings.API_PORT, int)
        assert isinstance(settings.NUM_RECOMMENDATIONS, int)
        assert 0 <= settings.EPSILON <= 1
