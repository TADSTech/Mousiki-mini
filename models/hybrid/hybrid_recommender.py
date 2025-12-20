"""
Production Hybrid Recommender System

Combines Content-Based Filtering (CBF) with Collaborative Filtering (CF)
for state-of-the-art music recommendations.

Components:
- Content-based: Track embeddings + semantic similarity
- Collaborative: User-item embeddings from neural networks
- Hybrid: Adaptive weighted combination with fallback strategies
"""

import numpy as np
import torch
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Result of a recommendation request."""
    user_id: int
    recommendations: List[Tuple[int, float]]  # [(track_id, score), ...]
    cbf_scores: Dict[int, float]
    cf_scores: Dict[int, float]
    hybrid_scores: Dict[int, float]
    method: str  # 'hybrid', 'cbf_fallback', 'cf_fallback', 'popularity'
    timestamp: str


class ContentBasedComponent:
    """Content-based filtering using track embeddings and similarity."""
    
    def __init__(self, similarity_matrix_path: str, top_k: int = 100):
        """
        Initialize CBF component.
        
        Args:
            similarity_matrix_path: Path to saved similarity matrix
            top_k: Top K similar tracks to consider
        """
        self.top_k = top_k
        
        # Load similarity matrix
        try:
            data = sparse.load_npz(similarity_matrix_path)
            self.similarity_matrix = data.astype(np.float32)
            logger.info(f"✓ Loaded CBF similarity matrix: {self.similarity_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to load CBF matrix: {e}")
            self.similarity_matrix = None
    
    def get_similar_tracks(
        self,
        track_id: int,
        n_recommendations: int = 10,
        threshold: float = 0.1
    ) -> Dict[int, float]:
        """
        Get similar tracks using content-based similarity.
        
        Args:
            track_id: Target track ID
            n_recommendations: Number of recommendations to return
            threshold: Minimum similarity threshold
            
        Returns:
            Dict of {track_id: similarity_score}
        """
        if self.similarity_matrix is None:
            return {}
            
        try:
            # Check if track_id is valid (within matrix range)
            if track_id >= self.similarity_matrix.shape[0]:
                return {}

            # Get similarity row for this track
            similarities = self.similarity_matrix[track_id].toarray().flatten()
            
            # Filter by threshold and get top-k
            top_indices = np.argsort(-similarities)[:self.top_k]
            top_indices = top_indices[similarities[top_indices] >= threshold]
            
            # Build result dictionary
            recommendations = {
                int(idx): float(similarities[idx])
                for idx in top_indices[:n_recommendations]
                if similarities[idx] > 0 and idx != track_id
            }
            
            return recommendations
        except Exception as e:
            logger.warning(f"CBF error for track {track_id}: {e}")
            return {}
    
    def score_from_history(
        self,
        user_history: List[int],
        n_recommendations: int = 10,
        decay: float = 0.95
    ) -> Dict[int, float]:
        """
        Score tracks based on user's listening history using CBF.
        
        Args:
            user_history: List of track IDs user has interacted with
            n_recommendations: Number of recommendations to return
            decay: Decay factor for older history items (newer = higher weight)
            
        Returns:
            Dict of {track_id: score}
        """
        track_scores = {}
        
        # Process history with recency weighting
        for i, track_id in enumerate(user_history):
            weight = decay ** (len(user_history) - i - 1)  # Recent items have higher weight
            
            # Get similar tracks
            similar = self.get_similar_tracks(track_id, n_recommendations * 3, threshold=0.05)
            
            for similar_track_id, similarity in similar.items():
                score = similarity * weight
                track_scores[similar_track_id] = track_scores.get(similar_track_id, 0) + score
        
        # Normalize and sort
        if track_scores:
            max_score = max(track_scores.values())
            track_scores = {k: v / max_score for k, v in track_scores.items()}
        
        # Remove already-listened tracks
        user_history_set = set(user_history)
        track_scores = {k: v for k, v in track_scores.items() if k not in user_history_set}
        
        # Sort and return top-n
        return dict(sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])


class CollaborativeFilteringComponent:
    """Collaborative filtering using neural embeddings."""
    
    def __init__(
        self,
        model_path: str,
        mappings_path: str,
        device: str = "cuda"
    ):
        """
        Initialize CF component.
        
        Args:
            model_path: Path to trained CF model
            mappings_path: Path to ID mappings pickle
            device: Device to use ('cuda' or 'cpu')
        """
        from models.collaborative.train_cf import NeuralCollaborativeFiltering
        
        self.device = device if torch.cuda.is_available() else "cpu"
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = NeuralCollaborativeFiltering(
                num_users=checkpoint['num_users'],
                num_items=checkpoint['num_items'],
                embedding_dim=checkpoint['embedding_dim'],
                hidden_layers=checkpoint['hidden_layers']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.user_id_map = mappings['user_id_map']
                self.item_id_map = mappings['item_id_map']
                self.reverse_user_map = mappings['reverse_user_map']
                self.reverse_item_map = mappings['reverse_item_map']
            
            logger.info(f"✓ Loaded CF model: {checkpoint['num_users']} users, {checkpoint['num_items']} items")
        except Exception as e:
            logger.error(f"Failed to load CF model: {e}")
            self.model = None
            self.user_id_map = {}
            self.item_id_map = {}

    def predict_user_item_score(self, user_id: int, track_id: int) -> float:
        """
        Predict score for a user-track pair.
        
        Args:
            user_id: Original user ID
            track_id: Original track ID
            
        Returns:
            Predicted score (0-5 range)
        """
        if not self.model:
            return 0.0

        try:
            # Map IDs
            user_idx = self.user_id_map.get(user_id)
            item_idx = self.item_id_map.get(track_id)
            
            if user_idx is None or item_idx is None:
                return 0.0
            
            # Get prediction
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                score = self.model(user_tensor, item_tensor).cpu().item()
            
            # Clamp to 0-5 range
            return float(max(0, min(5, score)))
        except Exception as e:
            logger.debug(f"CF prediction error: {e}")
            return 0.0
    
    def score_candidates(
        self,
        user_id: int,
        candidate_tracks: List[int],
        batch_size: int = 256
    ) -> Dict[int, float]:
        """
        Score multiple candidate tracks for a user.
        
        Args:
            user_id: User ID
            candidate_tracks: List of track IDs to score
            batch_size: Batch size for inference
            
        Returns:
            Dict of {track_id: score}
        """
        if not self.model or user_id not in self.user_id_map:
            return {}
        
        scores = {}
        
        # Process in batches
        for i in range(0, len(candidate_tracks), batch_size):
            batch_tracks = candidate_tracks[i:i+batch_size]
            
            with torch.no_grad():
                user_tensor = torch.LongTensor(
                    [self.user_id_map[user_id]] * len(batch_tracks)
                ).to(self.device)
                
                item_tensors = []
                valid_tracks = []
                
                for track_id in batch_tracks:
                    if track_id in self.item_id_map:
                        item_tensors.append(self.item_id_map[track_id])
                        valid_tracks.append(track_id)
                
                if item_tensors:
                    item_tensor = torch.LongTensor(item_tensors).to(self.device)
                    batch_scores = self.model(user_tensor[:len(valid_tracks)], item_tensor)
                    batch_scores = batch_scores.cpu().numpy()
                    
                    for track_id, score in zip(valid_tracks, batch_scores):
                        scores[track_id] = float(max(0, min(5, score)))
        
        return scores


class HybridRecommender:
    """Production hybrid recommender combining CBF + CF."""
    
    def __init__(
        self,
        cbf_similarity_path: str,
        cf_model_path: str,
        cf_mappings_path: str,
        cbf_weight: float = 0.4,
        cf_weight: float = 0.4,
        popularity_weight: float = 0.2,
        device: str = "cuda"
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            cbf_similarity_path: Path to CBF similarity matrix
            cf_model_path: Path to CF model
            cf_mappings_path: Path to CF ID mappings
            cbf_weight: Weight for CBF scores
            cf_weight: Weight for CF scores
            popularity_weight: Weight for popularity scores
            device: Device to use
        """
        # Load components
        logger.info("Initializing Hybrid Recommender...")
        self.cbf = ContentBasedComponent(cbf_similarity_path)
        self.cf = CollaborativeFilteringComponent(cf_model_path, cf_mappings_path, device)
        
        # Weights (normalized)
        total_weight = cbf_weight + cf_weight + popularity_weight
        if total_weight > 0:
            self.cbf_weight = cbf_weight / total_weight
            self.cf_weight = cf_weight / total_weight
            self.popularity_weight = popularity_weight / total_weight
        else:
            self.cbf_weight = 0.33
            self.cf_weight = 0.33
            self.popularity_weight = 0.33
            
        logger.info(f"✓ Weights: CBF={self.cbf_weight:.2f}, CF={self.cf_weight:.2f}, POP={self.popularity_weight:.2f}")
    
    def _get_user_history(self, user_id: int, limit: int = 50) -> List[int]:
        """Get user's listening history (Placeholder for local mode)."""
        # In a real app this would query the DB. 
        # For local demo, we rely on passed-in history or return empty.
        return []
    
    def _get_popular_tracks(self, n: int = 100, days: int = 30) -> List[Tuple[int, int]]:
        """Get trending tracks (Placeholder for local mode)."""
        # If CF model is loaded, use its items as "popular" pool to ensure we recommend items known to the model
        if hasattr(self, 'cf') and self.cf.model and self.cf.item_id_map:
             # Return the first N items from the map for the demo
             all_items = list(self.cf.item_id_map.keys())
             # Return as (id, score) tuples
             return [(item_id, 100) for item_id in all_items[:n]]
        
        # Fallback to 0-N if no CF model
        return [(i, 1000 - i) for i in range(n)]
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 20,
        use_cbf: bool = True,
        use_cf: bool = True,
        exclude_history: bool = True,
        user_history_override: List[int] = None
    ) -> RecommendationResult:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            use_cbf: Whether to use content-based filtering
            use_cf: Whether to use collaborative filtering
            exclude_history: Whether to exclude already-listened tracks
            user_history_override: Optional list of track IDs the user has listened to
            
        Returns:
            RecommendationResult with scores and recommendations
        """
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        cbf_scores = {}
        cf_scores = {}
        hybrid_scores = {}
        method = "hybrid"
        
        try:
            # Get user history
            if user_history_override is not None:
                user_history = user_history_override
            else:
                user_history = self._get_user_history(user_id) if exclude_history else []
            
            history_set = set(user_history)
            
            if not user_history and use_cbf:
                if not use_cf: # If also not using CF, basic fallback
                    method = "popularity"
                    # Fallback to popularity
                    popular_tracks = self._get_popular_tracks(n_recommendations * 2)
                    recommendations = [(track_id, float(count) / 1000) for track_id, count in popular_tracks[:n_recommendations]]
                    
                    return RecommendationResult(
                        user_id=user_id,
                        recommendations=recommendations,
                        cbf_scores={},
                        cf_scores={},
                        hybrid_scores={},
                        method=method,
                        timestamp=timestamp
                    )
            
            # Get candidate pool
            candidate_set = set()
            if use_cbf and user_history:
                for track_id in user_history[:10]:  # Use top 10 recent
                    similar = self.cbf.get_similar_tracks(track_id, n_recommendations * 2)
                    candidate_set.update(similar.keys())
                    cbf_scores.update(similar)
            
            if use_cf:
                # Use "popular" set as candidates to score if we don't have other candidates
                # This is a simplification for local demo vs full catalog scan
                popular_tracks = self._get_popular_tracks(n_recommendations * 5)
                candidate_set.update([track_id for track_id, _ in popular_tracks])
            
            # Remove history
            candidate_set = candidate_set - history_set
            candidate_list = list(candidate_set)
            
            if not candidate_list:
                # Fallback
                popular_tracks = self._get_popular_tracks(n_recommendations)
                recommendations = [(track_id, float(count) / 1000) for track_id, count in popular_tracks[:n_recommendations]]
                
                return RecommendationResult(
                    user_id=user_id,
                    recommendations=recommendations,
                    cbf_scores={},
                    cf_scores={},
                    hybrid_scores={},
                    method="popularity_fallback",
                    timestamp=timestamp
                )
            
            # Score with CF
            if use_cf:
                cf_scores = self.cf.score_candidates(user_id, candidate_list)
            
            # Hybrid combination
            for track_id in candidate_list:
                cbf_score = cbf_scores.get(track_id, 0.0)
                cf_score = cf_scores.get(track_id, 0.0) / 5.0 if track_id in cf_scores else 0.0
                
                # Only use CF if available, otherwise fall back to CBF
                if cf_score > 0 and cbf_score > 0:
                    hybrid_score = (self.cbf_weight * cbf_score + self.cf_weight * cf_score) / (self.cbf_weight + self.cf_weight)
                elif cf_score > 0:
                     hybrid_score = cf_score
                else:
                    hybrid_score = cbf_score
                
                if hybrid_score > 0:
                    hybrid_scores[track_id] = hybrid_score
            
            # Determine method used
            if cbf_scores and cf_scores:
                method = "hybrid"
            elif cf_scores:
                method = "cf_fallback"
            else:
                method = "cbf_fallback"
            
            # Sort and return top-n
            recommendations = sorted(
                hybrid_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            return RecommendationResult(
                user_id=user_id,
                recommendations=recommendations,
                cbf_scores=cbf_scores,
                cf_scores=cf_scores,
                hybrid_scores=hybrid_scores,
                method=method,
                timestamp=timestamp
            )
        
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            return RecommendationResult(
                user_id=user_id,
                recommendations=[],
                cbf_scores={},
                cf_scores={},
                hybrid_scores={},
                method="error_fallback",
                timestamp=timestamp
            )
    
    def batch_recommend(
        self,
        user_ids: List[int],
        n_recommendations: int = 20
    ) -> Dict[int, RecommendationResult]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dict of {user_id: RecommendationResult}
        """
        results = {}
        for user_id in user_ids:
            logger.info(f"Recommending for user {user_id}...")
            results[user_id] = self.recommend(user_id, n_recommendations)
        
        return results


def load_hybrid_recommender(
    config_path: str = "./config/hybrid_config.json",
    device: str = "cuda"
) -> HybridRecommender:
    """
    Load hybrid recommender from config.
    
    Args:
        config_path: Path to configuration file
        device: Device to use
        
    Returns:
        Initialized HybridRecommender
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return HybridRecommender(
        cbf_similarity_path=config['cbf_similarity_path'],
        cf_model_path=config['cf_model_path'],
        cf_mappings_path=config['cf_mappings_path'],
        cbf_weight=config.get('cbf_weight', 0.4),
        cf_weight=config.get('cf_weight', 0.4),
        popularity_weight=config.get('popularity_weight', 0.2),
        device=device
    )


if __name__ == "__main__":
    # Example usage
    recommender = HybridRecommender(
        cbf_similarity_path="./models/content/model/track_similarity_sparse.npz",
        cf_model_path="./models/collaborative/model_weights/ncf_model_20251206_123201.pt",
        cf_mappings_path="./models/collaborative/model_weights/id_mappings_20251206_123201.pkl"
    )
    
    # Get recommendations for user 1
    result = recommender.recommend(user_id=1, n_recommendations=10)
    
    print(f"\nRecommendations for User {result.user_id} (Method: {result.method})")
    print("="*60)
    for i, (track_id, score) in enumerate(result.recommendations, 1):
        print(f"{i}. Track {track_id}: {score:.4f}")
