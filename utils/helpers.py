"""
Helper Utilities

Common utility functions.
"""

import hashlib
import time
from typing import Any, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def generate_id(text: str) -> str:
    """
    Generate a unique ID from text.
    
    Args:
        text: Input text
        
    Returns:
        MD5 hash as hex string
    """
    return hashlib.md5(text.encode()).hexdigest()


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Usage:
        @timing_decorator
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(
            f"{func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        
        return result
    
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def batch_iterator(items: list, batch_size: int):
    """
    Iterate over items in batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "3:45")
    """
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        whole: Whole value
        
    Returns:
        Percentage (0-100)
    """
    return safe_divide(part * 100, whole, 0.0)
