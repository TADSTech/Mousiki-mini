"""
Production Model Evaluation Script

Comprehensive offline evaluation of CBF, CF, and Hybrid recommender systems.
Generates detailed metrics, comparisons, and visualizations.
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from typing import Dict, List, Set, Tuple
import json
import pickle

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models.evaluation.eval_metrics import RecommendationMetrics, EvaluationSuite
from models.hybrid.hybrid_recommender import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate recommendation models using historical data."""
    
    def __init__(self, db_connection_string: str):
        """Initialize evaluator."""
        self.db_string = db_connection_string
        self.metrics = RecommendationMetrics()
        self.results = {}
    
    def load_test_data(
        self,
        test_size: float = 0.2,
        min_interactions: int = 5
    ) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """
        Load and split user interactions into train/test sets.
        
        Uses temporal split: earlier interactions for train, later for test.
        
        Args:
            test_size: Proportion of each user's interactions for test
            min_interactions: Minimum interactions per user
            
        Returns:
            Tuple of (train_data, test_data) as {user_id: set(track_ids)}
        """
        logger.info("Loading test data from database...")
        
        conn = psycopg2.connect(self.db_string)
        
        # Load user interactions with play events
        query = """
        SELECT user_id, track_id, timestamp
        FROM interactions
        WHERE interaction_type = 'play'
        ORDER BY user_id, timestamp
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df):,} interactions")
        
        # Filter users with minimum interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        logger.info(f"Filtered to {len(df):,} interactions from {len(valid_users)} users")
        
        # Temporal split per user
        train_data = {}
        test_data = {}
        
        for user_id in valid_users:
            user_interactions = df[df['user_id'] == user_id].sort_values('timestamp')
            n_test = max(1, int(len(user_interactions) * test_size))
            
            train_tracks = set(user_interactions.iloc[:-n_test]['track_id'].values)
            test_tracks = set(user_interactions.iloc[-n_test:]['track_id'].values)
            
            train_data[user_id] = train_tracks
            test_data[user_id] = test_tracks
        
        logger.info(f"✓ Split complete:")
        logger.info(f"  Train: {sum(len(v) for v in train_data.values()):,} interactions")
        logger.info(f"  Test: {sum(len(v) for v in test_data.values()):,} interactions")
        
        return train_data, test_data
    
    def evaluate_model(
        self,
        model_name: str,
        recommender,
        test_data: Dict[int, Set[int]],
        k_values: List[int] = [5, 10, 20],
        n_recommendations: int = 20
    ) -> pd.DataFrame:
        """
        Evaluate a recommendation model.
        
        Args:
            model_name: Name of the model being evaluated
            recommender: Recommender instance with recommend() method
            test_data: Ground truth test data {user_id: set(track_ids)}
            k_values: K values to evaluate
            n_recommendations: Number of recommendations to generate
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        results = []
        
        for user_id, relevant_tracks in test_data.items():
            try:
                # Get recommendations
                result = recommender.recommend(
                    user_id=user_id,
                    n_recommendations=n_recommendations,
                    exclude_history=True
                )
                
                recommended_tracks = [track_id for track_id, _ in result.recommendations]
                
                # Compute metrics
                user_metrics = {
                    'user_id': user_id,
                    'model': model_name,
                    'method': result.method,
                    'n_relevant': len(relevant_tracks),
                    'n_recommended': len(recommended_tracks)
                }
                
                # Metrics at different K values
                for k in k_values:
                    user_metrics[f'precision@{k}'] = self.metrics.precision_at_k(
                        recommended_tracks, relevant_tracks, k
                    )
                    user_metrics[f'recall@{k}'] = self.metrics.recall_at_k(
                        recommended_tracks, relevant_tracks, k
                    )
                    user_metrics[f'ndcg@{k}'] = self.metrics.ndcg_at_k(
                        recommended_tracks, relevant_tracks, k
                    )
                    user_metrics[f'hit_rate@{k}'] = self.metrics.hit_rate_at_k(
                        recommended_tracks, relevant_tracks, k
                    )
                
                # Average precision
                user_metrics['avg_precision'] = self.metrics.average_precision(
                    recommended_tracks, relevant_tracks
                )
                
                results.append(user_metrics)
            
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Compute aggregate metrics
        logger.info("\nAggregate Metrics:")
        
        metric_cols = [col for col in results_df.columns 
                      if col not in ['user_id', 'model', 'method', 'n_relevant', 'n_recommended']]
        
        for col in metric_cols:
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            logger.info(f"  {col:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Method breakdown
        if 'method' in results_df.columns:
            logger.info("\nMethod Distribution:")
            method_counts = results_df['method'].value_counts()
            for method, count in method_counts.items():
                pct = 100 * count / len(results_df)
                logger.info(f"  {method:15s}: {count:4d} ({pct:5.1f}%)")
        
        return results_df
    
    def compare_models(
        self,
        results_dict: Dict[str, pd.DataFrame],
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.
        
        Args:
            results_dict: Dictionary of {model_name: results_df}
            k_values: K values to compare
            
        Returns:
            Comparison DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info("Model Comparison")
        logger.info(f"{'='*60}")
        
        comparison = []
        
        for model_name, results_df in results_dict.items():
            model_stats = {'model': model_name}
            
            # Aggregate metrics
            for k in k_values:
                model_stats[f'precision@{k}'] = results_df[f'precision@{k}'].mean()
                model_stats[f'recall@{k}'] = results_df[f'recall@{k}'].mean()
                model_stats[f'ndcg@{k}'] = results_df[f'ndcg@{k}'].mean()
                model_stats[f'hit_rate@{k}'] = results_df[f'hit_rate@{k}'].mean()
            
            model_stats['avg_precision'] = results_df['avg_precision'].mean()
            model_stats['n_users'] = len(results_df)
            
            comparison.append(model_stats)
        
        comparison_df = pd.DataFrame(comparison)
        
        logger.info("\n" + comparison_df.to_string(index=False))
        
        # Find best model for each metric
        logger.info("\nBest Models:")
        metric_cols = [col for col in comparison_df.columns if col not in ['model', 'n_users']]
        
        for col in metric_cols:
            best_idx = comparison_df[col].idxmax()
            best_model = comparison_df.loc[best_idx, 'model']
            best_score = comparison_df.loc[best_idx, col]
            logger.info(f"  {col:20s}: {best_model:15s} ({best_score:.4f})")
        
        return comparison_df
    
    def save_results(
        self,
        results_dict: Dict[str, pd.DataFrame],
        comparison_df: pd.DataFrame,
        output_dir: str = "./evaluation_results"
    ):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual model results
        for model_name, results_df in results_dict.items():
            filename = output_path / f"{model_name}_results_{timestamp}.csv"
            results_df.to_csv(filename, index=False)
            logger.info(f"✓ Saved {model_name} results: {filename}")
        
        # Save comparison
        comparison_file = output_path / f"model_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"✓ Saved comparison: {comparison_file}")
        
        # Save summary JSON
        summary = {
            'timestamp': timestamp,
            'models_evaluated': list(results_dict.keys()),
            'metrics': {}
        }
        
        for _, row in comparison_df.iterrows():
            model = row['model']
            summary['metrics'][model] = row.to_dict()
        
        summary_file = output_path / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary: {summary_file}")


def main():
    """Run comprehensive model evaluation."""
    
    logger.info("="*60)
    logger.info("Mousiki Model Evaluation Suite")
    logger.info("="*60)
    
    # Configuration
    db_string = "postgresql://mousiki_user:mousiki_password@localhost:5432/mousiki"
    
    # Find latest model files
    cbf_files = list(Path("models/content/model").glob("similarity_*.npz"))
    cf_files = list(Path("models/collaborative/model_weights").glob("ncf_model_*.pt"))
    
    if not cbf_files or not cf_files:
        logger.error("❌ Model files not found. Train models first.")
        return
    
    cbf_path = sorted(cbf_files)[-1]
    cf_model = sorted(cf_files)[-1]
    timestamp = cf_model.stem.split("_", 2)[-1]
    cf_mappings = cf_model.parent / f"id_mappings_{timestamp}.pkl"
    
    logger.info(f"Using models:")
    logger.info(f"  CBF: {cbf_path.name}")
    logger.info(f"  CF:  {cf_model.name}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(db_string)
    
    # Load test data
    train_data, test_data = evaluator.load_test_data(
        test_size=0.2,
        min_interactions=5
    )
    
    if not test_data:
        logger.error("❌ No test data available")
        return
    
    # Initialize recommender
    logger.info("\nInitializing Hybrid Recommender...")
    recommender = HybridRecommender(
        cbf_similarity_path=str(cbf_path.absolute()),
        cf_model_path=str(cf_model.absolute()),
        cf_mappings_path=str(cf_mappings.absolute()),
        db_connection_string=db_string,
        cbf_weight=0.4,
        cf_weight=0.4,
        popularity_weight=0.2
    )
    
    # Evaluate hybrid model
    k_values = [5, 10, 20]
    hybrid_results = evaluator.evaluate_model(
        model_name="Hybrid (CBF+CF)",
        recommender=recommender,
        test_data=test_data,
        k_values=k_values,
        n_recommendations=20
    )
    
    # Evaluate CBF-only
    logger.info("\nTesting CBF-only mode...")
    
    class CBFOnlyWrapper:
        """Wrapper to test CBF-only recommendations."""
        def __init__(self, recommender):
            self.recommender = recommender
        
        def recommend(self, user_id, n_recommendations, exclude_history=True):
            return self.recommender.recommend(
                user_id, n_recommendations, 
                use_cbf=True, use_cf=False, exclude_history=exclude_history
            )
    
    cbf_results = evaluator.evaluate_model(
        model_name="CBF Only",
        recommender=CBFOnlyWrapper(recommender),
        test_data=test_data,
        k_values=k_values,
        n_recommendations=20
    )
    
    # Evaluate CF-only
    logger.info("\nTesting CF-only mode...")
    
    class CFOnlyWrapper:
        """Wrapper to test CF-only recommendations."""
        def __init__(self, recommender):
            self.recommender = recommender
        
        def recommend(self, user_id, n_recommendations, exclude_history=True):
            return self.recommender.recommend(
                user_id, n_recommendations,
                use_cbf=False, use_cf=True, exclude_history=exclude_history
            )
    
    cf_results = evaluator.evaluate_model(
        model_name="CF Only",
        recommender=CFOnlyWrapper(recommender),
        test_data=test_data,
        k_values=k_values,
        n_recommendations=20
    )
    
    # Compare models
    results_dict = {
        'Hybrid': hybrid_results,
        'CBF_Only': cbf_results,
        'CF_Only': cf_results
    }
    
    comparison_df = evaluator.compare_models(results_dict, k_values=k_values)
    
    # Save results
    evaluator.save_results(results_dict, comparison_df)
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info("Results saved to: ./evaluation_results/")


if __name__ == "__main__":
    main()
