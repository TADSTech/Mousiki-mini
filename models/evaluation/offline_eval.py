"""
Offline Evaluation Module

Performs offline evaluation of recommendation models using historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import logging

from eval_metrics import RecommendationMetrics, EvaluationSuite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineEvaluator:
    """Offline evaluation using train/test splits."""
    
    def __init__(self, test_size: float = 0.2, min_interactions: int = 5):
        """
        Initialize offline evaluator.
        
        Args:
            test_size: Proportion of data for test set
            min_interactions: Minimum interactions per user to include in evaluation
        """
        self.test_size = test_size
        self.min_interactions = min_interactions
        self.evaluation_suite = EvaluationSuite()
    
    def temporal_split(
        self,
        interactions_df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (earlier for train, later for test).
        
        Args:
            interactions_df: DataFrame with user interactions
            timestamp_col: Name of timestamp column
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Sort by timestamp
        df_sorted = interactions_df.sort_values(timestamp_col)
        
        # Split based on time
        split_idx = int(len(df_sorted) * (1 - self.test_size))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        logger.info(f"Temporal split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    
    def user_split(
        self,
        interactions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split each user's interactions (earlier for train, latest for test).
        
        Args:
            interactions_df: DataFrame with user interactions
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_list = []
        test_list = []
        
        for user_id, user_df in interactions_df.groupby('user_id'):
            user_sorted = user_df.sort_values('timestamp')
            
            # Skip users with too few interactions
            if len(user_sorted) < self.min_interactions:
                continue
            
            # Split this user's data
            n_test = max(1, int(len(user_sorted) * self.test_size))
            train_list.append(user_sorted.iloc[:-n_test])
            test_list.append(user_sorted.iloc[-n_test:])
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        logger.info(f"User-based split: {len(train_df)} train, {len(test_df)} test")
        
        return train_df, test_df
    
    def prepare_test_relevance(
        self,
        test_df: pd.DataFrame,
        relevance_threshold: float = 3.0
    ) -> Dict[int, Set[int]]:
        """
        Prepare ground truth relevance from test set.
        
        Args:
            test_df: Test interactions DataFrame
            relevance_threshold: Score threshold for relevance
            
        Returns:
            Dictionary mapping user IDs to sets of relevant item IDs
        """
        relevance = {}
        
        for user_id, user_df in test_df.groupby('user_id'):
            # Items with high scores are relevant
            relevant_items = set(
                user_df[user_df['normalized_score'] >= relevance_threshold]['track_id']
            )
            
            if relevant_items:
                relevance[user_id] = relevant_items
        
        logger.info(f"Prepared relevance for {len(relevance)} users")
        
        return relevance
    
    def evaluate_model(
        self,
        recommendations: Dict[int, List[int]],
        test_df: pd.DataFrame,
        relevance_threshold: float = 3.0,
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Evaluate a recommendation model.
        
        Args:
            recommendations: Dictionary mapping user IDs to recommended item lists
            test_df: Test interactions DataFrame
            relevance_threshold: Score threshold for relevance
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        # Prepare ground truth
        relevance = self.prepare_test_relevance(test_df, relevance_threshold)
        
        # Evaluate
        results_df = self.evaluation_suite.evaluate_all(
            recommendations,
            relevance,
            k_values
        )
        
        return results_df
    
    def compare_models(
        self,
        model_recommendations: Dict[str, Dict[int, List[int]]],
        test_df: pd.DataFrame,
        relevance_threshold: float = 3.0,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_recommendations: Dictionary mapping model names to recommendations
            test_df: Test interactions DataFrame
            relevance_threshold: Score threshold for relevance
            k: K value for metrics
            
        Returns:
            DataFrame comparing models
        """
        comparison = []
        
        relevance = self.prepare_test_relevance(test_df, relevance_threshold)
        
        for model_name, recommendations in model_recommendations.items():
            results_df = self.evaluation_suite.evaluate_all(
                recommendations,
                relevance,
                k_values=[k]
            )
            
            # Compute average metrics
            avg_metrics = {
                'model': model_name,
                f'precision@{k}': results_df[f'precision@{k}'].mean(),
                f'recall@{k}': results_df[f'recall@{k}'].mean(),
                f'f1@{k}': results_df[f'f1@{k}'].mean(),
                f'ndcg@{k}': results_df[f'ndcg@{k}'].mean(),
                f'hit_rate@{k}': results_df[f'hit_rate@{k}'].mean()
            }
            
            comparison.append(avg_metrics)
        
        comparison_df = pd.DataFrame(comparison)
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def leave_one_out_evaluation(
        self,
        interactions_df: pd.DataFrame,
        recommender_fn,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Leave-one-out evaluation: hold out one item per user for testing.
        
        Args:
            interactions_df: Full interactions DataFrame
            recommender_fn: Function that takes train_df and user_id, returns recommendations
            k: Number of recommendations
            
        Returns:
            Dictionary of average metrics
        """
        metrics = RecommendationMetrics()
        
        hits = []
        ndcgs = []
        
        for user_id, user_df in interactions_df.groupby('user_id'):
            if len(user_df) < 2:
                continue
            
            # Leave out the last (most recent) interaction
            user_sorted = user_df.sort_values('timestamp')
            train_user = user_sorted.iloc[:-1]
            test_item = user_sorted.iloc[-1]['track_id']
            
            # Get recommendations
            train_df = interactions_df[interactions_df['user_id'] != user_id]
            train_df = pd.concat([train_df, train_user])
            
            recommendations = recommender_fn(train_df, user_id)
            
            # Evaluate
            hit = 1.0 if test_item in recommendations[:k] else 0.0
            ndcg = metrics.ndcg_at_k(recommendations, {test_item}, k)
            
            hits.append(hit)
            ndcgs.append(ndcg)
        
        results = {
            f'hit_rate@{k}': np.mean(hits),
            f'ndcg@{k}': np.mean(ndcgs)
        }
        
        logger.info("\nLeave-One-Out Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results


class ABTestSimulator:
    """Simulates A/B testing for recommendation algorithms."""
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        self.metrics = RecommendationMetrics()
    
    def simulate_test(
        self,
        model_a_recs: Dict[int, List[int]],
        model_b_recs: Dict[int, List[int]],
        test_interactions: pd.DataFrame,
        traffic_split: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Simulate A/B test between two models.
        
        Args:
            model_a_recs: Recommendations from model A
            model_b_recs: Recommendations from model B
            test_interactions: Test interactions DataFrame
            traffic_split: Proportion of traffic to model A
            
        Returns:
            Dictionary with results for each model
        """
        all_users = list(set(model_a_recs.keys()) & set(model_b_recs.keys()))
        
        # Randomly assign users to groups
        n_a = int(len(all_users) * traffic_split)
        self.rng.shuffle(all_users)
        
        group_a_users = all_users[:n_a]
        group_b_users = all_users[n_a:]
        
        # Simulate interactions
        results = {
            'model_a': self._simulate_group(group_a_users, model_a_recs, test_interactions),
            'model_b': self._simulate_group(group_b_users, model_b_recs, test_interactions)
        }
        
        logger.info("\nA/B Test Results:")
        logger.info(f"Model A (n={len(group_a_users)}): {results['model_a']}")
        logger.info(f"Model B (n={len(group_b_users)}): {results['model_b']}")
        
        return results
    
    def _simulate_group(
        self,
        users: List[int],
        recommendations: Dict[int, List[int]],
        test_interactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Simulate metrics for a user group."""
        clicks = 0
        engagements = 0
        total_recs = 0
        
        for user_id in users:
            if user_id not in recommendations:
                continue
            
            user_recs = recommendations[user_id]
            user_interactions = test_interactions[test_interactions['user_id'] == user_id]
            
            # Simulate clicks (did user interact with any recommended item?)
            clicked = any(
                item in user_interactions['track_id'].values
                for item in user_recs[:5]  # Consider top 5
            )
            
            if clicked:
                clicks += 1
            
            # Simulate engagement (positive interactions)
            engaged = any(
                item in user_interactions[
                    user_interactions['normalized_score'] >= 4.0
                ]['track_id'].values
                for item in user_recs[:5]
            )
            
            if engaged:
                engagements += 1
            
            total_recs += 1
        
        return {
            'ctr': clicks / total_recs if total_recs > 0 else 0.0,
            'engagement_rate': engagements / total_recs if total_recs > 0 else 0.0
        }


def main():
    """Example usage of offline evaluation."""
    # Sample interaction data
    sample_data = pd.DataFrame({
        'user_id': np.repeat(range(1, 11), 10),
        'track_id': np.random.randint(100, 200, 100),
        'normalized_score': np.random.uniform(0, 5, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    # Initialize evaluator
    evaluator = OfflineEvaluator(test_size=0.2)
    
    # Split data
    train_df, test_df = evaluator.user_split(sample_data)
    
    # Mock recommendations (in practice, generated by your model)
    recommendations = {}
    for user_id in test_df['user_id'].unique():
        recommendations[user_id] = list(np.random.choice(range(100, 200), 10, replace=False))
    
    # Evaluate
    results = evaluator.evaluate_model(recommendations, test_df, k_values=[5, 10])
    
    print("\nEvaluation Results:")
    print(results.describe())


if __name__ == "__main__":
    main()
