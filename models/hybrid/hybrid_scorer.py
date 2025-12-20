"""
Hybrid Scoring Module

Combines content-based and collaborative filtering scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridScorer:
    """Combines multiple recommendation signals into a hybrid score."""
    
    def __init__(
        self,
        content_weight: float = 0.4,
        collaborative_weight: float = 0.4,
        popularity_weight: float = 0.2
    ):
        """
        Initialize hybrid scorer.
        
        Args:
            content_weight: Weight for content-based scores
            collaborative_weight: Weight for collaborative filtering scores
            popularity_weight: Weight for popularity scores
        """
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.popularity_weight = popularity_weight
        
        # Normalize weights
        total = content_weight + collaborative_weight + popularity_weight
        self.content_weight /= total
        self.collaborative_weight /= total
        self.popularity_weight /= total
    
    def combine_scores(
        self,
        content_scores: Dict[int, float],
        collaborative_scores: Dict[int, float],
        popularity_scores: Optional[Dict[int, float]] = None,
        item_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Combine multiple scoring signals.
        
        Args:
            content_scores: Content-based similarity scores
            collaborative_scores: Collaborative filtering scores
            popularity_scores: Popularity scores (optional)
            item_ids: List of item IDs to score (if None, use all from scores)
            
        Returns:
            Dictionary of combined scores
        """
        if item_ids is None:
            item_ids = set(content_scores.keys()) | set(collaborative_scores.keys())
            if popularity_scores:
                item_ids |= set(popularity_scores.keys())
            item_ids = list(item_ids)
        
        combined = {}
        
        for item_id in item_ids:
            score = 0.0
            
            # Add weighted content score
            if item_id in content_scores:
                score += self.content_weight * content_scores[item_id]
            
            # Add weighted collaborative score
            if item_id in collaborative_scores:
                score += self.collaborative_weight * collaborative_scores[item_id]
            
            # Add weighted popularity score
            if popularity_scores and item_id in popularity_scores:
                score += self.popularity_weight * popularity_scores[item_id]
            
            combined[item_id] = score
        
        return combined
    
    def rank_items(
        self,
        content_scores: Dict[int, float],
        collaborative_scores: Dict[int, float],
        popularity_scores: Optional[Dict[int, float]] = None,
        top_k: int = 20,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank items using hybrid scores.
        
        Args:
            content_scores: Content-based similarity scores
            collaborative_scores: Collaborative filtering scores
            popularity_scores: Popularity scores (optional)
            top_k: Number of items to return
            exclude_items: Items to exclude from ranking
            
        Returns:
            List of (item_id, score) tuples, sorted by score
        """
        combined_scores = self.combine_scores(
            content_scores,
            collaborative_scores,
            popularity_scores
        )
        
        # Filter out excluded items
        if exclude_items:
            combined_scores = {
                k: v for k, v in combined_scores.items()
                if k not in exclude_items
            }
        
        # Sort by score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def update_weights(
        self,
        content_weight: float,
        collaborative_weight: float,
        popularity_weight: float
    ):
        """
        Update scoring weights.
        
        Args:
            content_weight: New content weight
            collaborative_weight: New collaborative weight
            popularity_weight: New popularity weight
        """
        total = content_weight + collaborative_weight + popularity_weight
        self.content_weight = content_weight / total
        self.collaborative_weight = collaborative_weight / total
        self.popularity_weight = popularity_weight / total
        
        logger.info(f"Updated weights: content={self.content_weight:.2f}, "
                   f"cf={self.collaborative_weight:.2f}, "
                   f"pop={self.popularity_weight:.2f}")


class AdaptiveHybridScorer(HybridScorer):
    """Hybrid scorer with adaptive weights based on context."""
    
    def __init__(
        self,
        default_content_weight: float = 0.4,
        default_collaborative_weight: float = 0.4,
        default_popularity_weight: float = 0.2
    ):
        super().__init__(
            default_content_weight,
            default_collaborative_weight,
            default_popularity_weight
        )
        self.default_content_weight = self.content_weight
        self.default_collaborative_weight = self.collaborative_weight
        self.default_popularity_weight = self.popularity_weight
    
    def get_context_weights(
        self,
        user_interaction_count: int,
        is_cold_start_item: bool = False
    ) -> Tuple[float, float, float]:
        """
        Determine weights based on context.
        
        Args:
            user_interaction_count: Number of user interactions
            is_cold_start_item: Whether item is new with few interactions
            
        Returns:
            Tuple of (content_weight, cf_weight, popularity_weight)
        """
        # For new users, rely more on content and popularity
        if user_interaction_count < 10:
            return 0.5, 0.2, 0.3
        
        # For cold-start items, rely more on content
        if is_cold_start_item:
            return 0.6, 0.2, 0.2
        
        # Default weights for established users and items
        return (
            self.default_content_weight,
            self.default_collaborative_weight,
            self.default_popularity_weight
        )
    
    def rank_items_adaptive(
        self,
        content_scores: Dict[int, float],
        collaborative_scores: Dict[int, float],
        popularity_scores: Optional[Dict[int, float]] = None,
        user_interaction_count: int = 0,
        top_k: int = 20,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank items with adaptive weights.
        
        Args:
            content_scores: Content-based similarity scores
            collaborative_scores: Collaborative filtering scores
            popularity_scores: Popularity scores (optional)
            user_interaction_count: Number of user interactions
            top_k: Number of items to return
            exclude_items: Items to exclude
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get adaptive weights
        content_w, cf_w, pop_w = self.get_context_weights(user_interaction_count)
        
        # Temporarily update weights
        original_weights = (
            self.content_weight,
            self.collaborative_weight,
            self.popularity_weight
        )
        self.update_weights(content_w, cf_w, pop_w)
        
        # Rank items
        ranked = self.rank_items(
            content_scores,
            collaborative_scores,
            popularity_scores,
            top_k,
            exclude_items
        )
        
        # Restore original weights
        self.content_weight, self.collaborative_weight, self.popularity_weight = original_weights
        
        return ranked


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Dictionary of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return {}
    
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return {k: 0.5 for k in scores.keys()}
    
    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }


def main():
    """Example usage of hybrid scorer."""
    # Sample scores from different models
    content_scores = {
        101: 0.8,
        102: 0.6,
        103: 0.7,
        104: 0.5,
        105: 0.9
    }
    
    cf_scores = {
        101: 0.7,
        102: 0.9,
        103: 0.5,
        104: 0.8,
        106: 0.6
    }
    
    popularity_scores = {
        101: 0.9,
        102: 0.7,
        103: 0.8,
        104: 0.6,
        105: 0.5,
        106: 0.4
    }
    
    # Standard hybrid scorer
    scorer = HybridScorer(content_weight=0.4, collaborative_weight=0.4, popularity_weight=0.2)
    
    ranked = scorer.rank_items(content_scores, cf_scores, popularity_scores, top_k=5)
    
    print("Hybrid Ranking:")
    for item_id, score in ranked:
        print(f"  Item {item_id}: {score:.3f}")
    
    # Adaptive hybrid scorer
    adaptive_scorer = AdaptiveHybridScorer()
    
    # New user (few interactions)
    new_user_ranked = adaptive_scorer.rank_items_adaptive(
        content_scores,
        cf_scores,
        popularity_scores,
        user_interaction_count=5,
        top_k=5
    )
    
    print("\nNew User Ranking (adaptive):")
    for item_id, score in new_user_ranked:
        print(f"  Item {item_id}: {score:.3f}")
    
    # Established user
    established_user_ranked = adaptive_scorer.rank_items_adaptive(
        content_scores,
        cf_scores,
        popularity_scores,
        user_interaction_count=100,
        top_k=5
    )
    
    print("\nEstablished User Ranking (adaptive):")
    for item_id, score in established_user_ranked:
        print(f"  Item {item_id}: {score:.3f}")


if __name__ == "__main__":
    main()
