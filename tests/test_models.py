"""
Model Tests

Tests for recommendation models and algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from models.rankers.baseline_ranker import BaselineRanker, RandomRanker
from models.rankers.epsilon_greedy import EpsilonGreedy
from models.rankers.rewards import RewardCalculator
from models.hybrid.hybrid_scorer import HybridScorer
from models.evaluation.eval_metrics import RecommendationMetrics


class TestBaselineRanker:
    """Tests for baseline ranker."""
    
    def test_popularity_ranking(self):
        """Test popularity-based ranking."""
        sample_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'track_id': [101, 102, 101, 103, 102],
            'normalized_score': [4.5, 3.2, 5.0, 2.8, 4.0]
        })
        
        ranker = BaselineRanker()
        ranker.fit_popularity(sample_data)
        
        candidates = [101, 102, 103]
        ranked = ranker.rank_by_popularity(candidates, top_k=3)
        
        assert len(ranked) == 3
        assert all(item in candidates for item in ranked)
    
    def test_random_ranker(self):
        """Test random ranker."""
        ranker = RandomRanker(seed=42)
        candidates = list(range(1, 11))
        
        ranked = ranker.rank(candidates, top_k=5)
        
        assert len(ranked) == 5
        assert all(item in candidates for item in ranked)


class TestEpsilonGreedy:
    """Tests for epsilon-greedy exploration."""
    
    def test_selection(self):
        """Test epsilon-greedy selection."""
        eg = EpsilonGreedy(epsilon=0.2, seed=42)
        
        candidates = list(range(1, 21))
        scores = {i: float(i) for i in candidates}
        
        selected = eg.select(candidates, scores, top_k=10)
        
        assert len(selected) == 10
        assert all(item in candidates for item in selected)
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        eg = EpsilonGreedy(epsilon=0.5, decay_rate=0.9)
        initial_epsilon = eg.get_current_epsilon()
        
        eg.step()
        
        assert eg.get_current_epsilon() < initial_epsilon


class TestRewardCalculator:
    """Tests for reward calculator."""
    
    def test_immediate_reward(self):
        """Test immediate reward calculation."""
        calc = RewardCalculator()
        
        # Test like reward
        reward = calc.compute_immediate_reward('like')
        assert reward > 0
        
        # Test skip reward
        reward = calc.compute_immediate_reward('skip')
        assert reward < 0
    
    def test_play_reward_with_completion(self):
        """Test play reward with completion ratio."""
        calc = RewardCalculator()
        
        # Full play
        reward_full = calc.compute_immediate_reward('play', duration=240, track_duration=240)
        
        # Partial play
        reward_partial = calc.compute_immediate_reward('play', duration=120, track_duration=240)
        
        assert reward_full > reward_partial


class TestHybridScorer:
    """Tests for hybrid scorer."""
    
    def test_combine_scores(self):
        """Test combining multiple scores."""
        scorer = HybridScorer(
            content_weight=0.5,
            collaborative_weight=0.3,
            popularity_weight=0.2
        )
        
        content_scores = {1: 0.8, 2: 0.6, 3: 0.9}
        cf_scores = {1: 0.7, 2: 0.9, 3: 0.5}
        pop_scores = {1: 0.9, 2: 0.7, 3: 0.8}
        
        combined = scorer.combine_scores(content_scores, cf_scores, pop_scores)
        
        assert len(combined) == 3
        assert all(0 <= score <= 1 for score in combined.values())
    
    def test_rank_items(self):
        """Test ranking items."""
        scorer = HybridScorer()
        
        content_scores = {1: 0.8, 2: 0.6, 3: 0.9, 4: 0.5}
        cf_scores = {1: 0.7, 2: 0.9, 3: 0.5, 4: 0.8}
        
        ranked = scorer.rank_items(content_scores, cf_scores, top_k=3)
        
        assert len(ranked) == 3
        assert all(isinstance(item, tuple) for item in ranked)


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@K metric."""
        metrics = RecommendationMetrics()
        
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6}
        
        precision = metrics.precision_at_k(recommended, relevant, k=5)
        
        assert 0 <= precision <= 1
        assert precision == 0.4  # 2 out of 5
    
    def test_recall_at_k(self):
        """Test recall@K metric."""
        metrics = RecommendationMetrics()
        
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6}
        
        recall = metrics.recall_at_k(recommended, relevant, k=5)
        
        assert 0 <= recall <= 1
        assert recall == pytest.approx(2/3)  # 2 out of 3
    
    def test_ndcg_at_k(self):
        """Test NDCG@K metric."""
        metrics = RecommendationMetrics()
        
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3}
        
        ndcg = metrics.ndcg_at_k(recommended, relevant, k=5)
        
        assert 0 <= ndcg <= 1
