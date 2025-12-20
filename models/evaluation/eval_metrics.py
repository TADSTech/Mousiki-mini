"""
Evaluation Metrics Module

Provides metrics for evaluating recommendation quality.
"""

import numpy as np
import pandas as pd
from typing import List, Set, Dict, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Metrics for evaluating recommendation systems."""
    
    @staticmethod
    def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Precision@K: Proportion of recommended items that are relevant.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0 or len(recommended) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        hits = len([item for item in recommended_k if item in relevant])
        
        return hits / min(k, len(recommended_k))
    
    @staticmethod
    def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Recall@K: Proportion of relevant items that are recommended.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        hits = len([item for item in recommended_k if item in relevant])
        
        return hits / len(relevant)
    
    @staticmethod
    def f1_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        F1@K: Harmonic mean of precision and recall at K.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            F1@K score
        """
        precision = RecommendationMetrics.precision_at_k(recommended, relevant, k)
        recall = RecommendationMetrics.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def average_precision(recommended: List[int], relevant: Set[int]) -> float:
        """
        Average Precision: Average of precision values at each relevant item position.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            
        Returns:
            Average precision score
        """
        if len(relevant) == 0:
            return 0.0
        
        precisions = []
        num_hits = 0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                num_hits += 1
                precisions.append(num_hits / (i + 1))
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(relevant)
    
    @staticmethod
    def mean_average_precision(
        recommendations: Dict[int, List[int]],
        relevance: Dict[int, Set[int]]
    ) -> float:
        """
        Mean Average Precision (MAP): Mean of average precision across all users.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommended item lists
            relevance: Dictionary mapping user IDs to relevant item sets
            
        Returns:
            MAP score
        """
        aps = []
        
        for user_id, recommended in recommendations.items():
            if user_id in relevance:
                ap = RecommendationMetrics.average_precision(recommended, relevance[user_id])
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def ndcg_at_k(
        recommended: List[int],
        relevant: Set[int],
        k: int,
        relevance_scores: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Number of recommendations to consider
            relevance_scores: Optional dictionary of relevance scores (default: binary)
            
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        
        # DCG calculation
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant:
                if relevance_scores and item in relevance_scores:
                    rel = relevance_scores[item]
                else:
                    rel = 1.0
                
                dcg += rel / np.log2(i + 2)  # i+2 because positions start at 1
        
        # IDCG calculation (ideal DCG)
        if relevance_scores:
            ideal_relevance = sorted(
                [relevance_scores.get(item, 0) for item in relevant],
                reverse=True
            )[:k]
        else:
            ideal_relevance = [1.0] * min(len(relevant), k)
        
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
        """
        Hit Rate@K: Whether at least one relevant item is in top-K.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        recommended_k = recommended[:k]
        return 1.0 if any(item in relevant for item in recommended_k) else 0.0
    
    @staticmethod
    def coverage(recommendations: Dict[int, List[int]], total_items: int) -> float:
        """
        Catalog Coverage: Proportion of items that are recommended to at least one user.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommended item lists
            total_items: Total number of items in catalog
            
        Returns:
            Coverage ratio
        """
        recommended_items = set()
        for items in recommendations.values():
            recommended_items.update(items)
        
        return len(recommended_items) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def diversity(recommendations: List[int], item_features: pd.DataFrame) -> float:
        """
        Diversity: Average pairwise distance between recommended items.
        
        Args:
            recommendations: List of recommended item IDs
            item_features: DataFrame with item features/embeddings
            
        Returns:
            Diversity score
        """
        if len(recommendations) < 2:
            return 0.0
        
        # Get embeddings for recommended items
        rec_items = item_features[item_features['track_id'].isin(recommendations)]
        
        if len(rec_items) < 2:
            return 0.0
        
        # Assuming embeddings are stored in 'embedding' column
        embeddings = np.stack(rec_items['embedding'].values)
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings)
        
        # Average distance (excluding diagonal)
        n = len(distances)
        diversity_score = distances.sum() / (n * (n - 1)) if n > 1 else 0.0
        
        return diversity_score
    
    @staticmethod
    def novelty(
        recommendations: List[int],
        popularity_scores: Dict[int, float]
    ) -> float:
        """
        Novelty: Average unexpectedness of recommendations.
        
        Args:
            recommendations: List of recommended item IDs
            popularity_scores: Dictionary of item popularity scores (0-1)
            
        Returns:
            Novelty score (higher = more novel)
        """
        if len(recommendations) == 0:
            return 0.0
        
        # Novelty is inverse of popularity
        novelties = [1.0 - popularity_scores.get(item, 0.5) for item in recommendations]
        
        return np.mean(novelties)


class EvaluationSuite:
    """Comprehensive evaluation suite for recommendation systems."""
    
    def __init__(self):
        self.metrics = RecommendationMetrics()
        self.results = defaultdict(list)
    
    def evaluate_user(
        self,
        user_id: int,
        recommended: List[int],
        relevant: Set[int],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            user_id: User identifier
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        for k in k_values:
            results[f'precision@{k}'] = self.metrics.precision_at_k(recommended, relevant, k)
            results[f'recall@{k}'] = self.metrics.recall_at_k(recommended, relevant, k)
            results[f'f1@{k}'] = self.metrics.f1_at_k(recommended, relevant, k)
            results[f'ndcg@{k}'] = self.metrics.ndcg_at_k(recommended, relevant, k)
            results[f'hit_rate@{k}'] = self.metrics.hit_rate_at_k(recommended, relevant, k)
        
        results['average_precision'] = self.metrics.average_precision(recommended, relevant)
        
        return results
    
    def evaluate_all(
        self,
        recommendations: Dict[int, List[int]],
        relevance: Dict[int, Set[int]],
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Evaluate recommendations for all users.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommended item lists
            relevance: Dictionary mapping user IDs to relevant item sets
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        all_results = []
        
        for user_id in recommendations:
            if user_id in relevance:
                user_results = self.evaluate_user(
                    user_id,
                    recommendations[user_id],
                    relevance[user_id],
                    k_values
                )
                user_results['user_id'] = user_id
                all_results.append(user_results)
        
        results_df = pd.DataFrame(all_results)
        
        # Compute average metrics
        logger.info("\nAverage Metrics:")
        for col in results_df.columns:
            if col != 'user_id':
                avg = results_df[col].mean()
                logger.info(f"  {col}: {avg:.4f}")
        
        return results_df


def main():
    """Example usage of evaluation metrics."""
    # Sample recommendations and ground truth
    recommendations = {
        1: [101, 102, 103, 104, 105],
        2: [201, 202, 203, 204, 205],
        3: [301, 302, 303, 304, 305]
    }
    
    relevance = {
        1: {101, 103, 106},
        2: {202, 204, 206, 207},
        3: {305, 306}
    }
    
    # Evaluate
    evaluator = EvaluationSuite()
    results_df = evaluator.evaluate_all(recommendations, relevance, k_values=[3, 5])
    
    print("\nEvaluation Results:")
    print(results_df)
    
    # Test individual metrics
    metrics = RecommendationMetrics()
    
    user_recs = [101, 102, 103, 104, 105]
    user_relevant = {101, 103, 106}
    
    print(f"\nPrecision@5: {metrics.precision_at_k(user_recs, user_relevant, 5):.3f}")
    print(f"Recall@5: {metrics.recall_at_k(user_recs, user_relevant, 5):.3f}")
    print(f"NDCG@5: {metrics.ndcg_at_k(user_recs, user_relevant, 5):.3f}")


if __name__ == "__main__":
    main()
