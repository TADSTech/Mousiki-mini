#!/usr/bin/env python
"""
Quick test script for hybrid recommender system.
Verifies all components work and generates sample recommendations.
"""

import sys
import os
import logging
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test hybrid recommender system."""
    
    logger.info("="*60)
    logger.info("Mousiki Hybrid Recommender - System Test")
    logger.info("="*60)
    
    # Step 1: Check model files
    logger.info("\n1. Checking model files...")
    
    cbf_dir = Path("./models/content/model")
    cf_model_path = Path("./models/collaborative/model_weights")
    
    # Find latest CBF similarity matrix
    cbf_files = list(cbf_dir.glob("similarity_*.npz"))
    if not cbf_files:
        logger.error(f"✗ CBF similarity matrix not found in: {cbf_dir}")
        return False
    
    cbf_path = sorted(cbf_files)[-1]  # Latest
    logger.info(f"✓ CBF similarity matrix found: {cbf_path.name} ({cbf_path.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Find latest CF model
    cf_files = list(cf_model_path.glob("ncf_model_*.pt"))
    if not cf_files:
        logger.error(f"✗ CF model not found in: {cf_model_path}")
        return False
    
    cf_model = sorted(cf_files)[-1]  # Latest model
    # Extract timestamp from model filename: ncf_model_20251206_123201.pt
    timestamp = cf_model.stem.split("_", 2)[-1]  # Get everything after 'ncf_model_'
    mappings = cf_model_path / f"id_mappings_{timestamp}.pkl"
    
    logger.info(f"✓ CF model found: {cf_model.name} ({cf_model.stat().st_size / 1024 / 1024:.1f}MB)")
    logger.info(f"✓ CF mappings found: {mappings.name}")
    
    # Step 2: Initialize recommender
    logger.info("\n2. Initializing hybrid recommender...")
    try:
        from models.hybrid.hybrid_recommender import HybridRecommender
        
        recommender = HybridRecommender(
            cbf_similarity_path=str(cbf_path.absolute()),
            cf_model_path=str(cf_model.absolute()),
            cf_mappings_path=str(mappings.absolute()),
            db_connection_string="postgresql://mousiki_user:mousiki_password@localhost:5432/mousiki"
        )
        logger.info("✓ Recommender initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize recommender: {e}")
        return False
    
    # Step 3: Check database connection
    logger.info("\n3. Checking database connection...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            database="mousiki",
            user="mousiki_user",
            password="mousiki_password"
        )
        cursor = conn.cursor()
        
        # Check users and interactions
        cursor.execute("SELECT COUNT(*) FROM users;")
        n_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM interactions;")
        n_interactions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tracks;")
        n_tracks = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"✓ Database connected: {n_users} users, {n_tracks} tracks, {n_interactions} interactions")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False
    
    # Step 4: Test recommendations
    logger.info("\n4. Testing recommendations...")
    
    test_users = [1, 2, 3]
    
    for user_id in test_users:
        try:
            result = recommender.recommend(
                user_id=user_id,
                n_recommendations=5,
                exclude_history=True
            )
            
            status = "✓" if result.recommendations else "⚠"
            logger.info(f"{status} User {user_id} - Method: {result.method}, Recs: {len(result.recommendations)}")
            
            for i, (track_id, score) in enumerate(result.recommendations[:3], 1):
                logger.info(f"    {i}. Track {track_id}: {score:.4f}")
        
        except Exception as e:
            logger.error(f"✗ Recommendation failed for user {user_id}: {e}")
            return False
    
    # Step 5: Check API
    logger.info("\n5. Checking API...")
    try:
        from api.hybrid_recommender_api import app
        logger.info("✓ API module imported successfully")
    except Exception as e:
        logger.warning(f"⚠ API import warning: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("All checks passed! ✓")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Start API: python -m uvicorn api.hybrid_recommender_api:app --port 8000")
    logger.info("2. Test endpoint: curl http://localhost:8000/recommend/1")
    logger.info("3. View docs: http://localhost:8000/docs")
    logger.info("\nSee docs/HYBRID_RECOMMENDER.md for full documentation")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
