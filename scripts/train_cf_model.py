"""
Production-Ready Collaborative Filtering Training Script

Trains Neural Collaborative Filtering (NCF) model on user-track interactions.
Supports database input, checkpointing, and model export.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import psycopg2
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Tuple
import pickle
import json

from models.collaborative.train_cf import (
    NeuralCollaborativeFiltering,
    CFTrainer,
    InteractionDataset
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionCFTrainer:
    """Production-ready CF trainer with full pipeline."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_layers: list = [256, 128, 64],
        learning_rate: float = 0.001,
        device: str = "cuda",
        model_dir: str = "./models/collaborative/model_weights"
    ):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.trainer = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        logger.info(f"Using device: {self.device}")
    
    def load_from_database(
        self,
        db_connection_string: str,
        min_interactions: int = 5,
        sample_frac: float = 1.0
    ) -> pd.DataFrame:
        """
        Load interaction data from PostgreSQL database.
        
        Args:
            db_connection_string: PostgreSQL connection string
            min_interactions: Minimum interactions per user to include
            sample_frac: Fraction of data to sample (1.0 = all data)
            
        Returns:
            DataFrame with user_id, track_id, normalized_score
        """
        logger.info("Loading interactions from database...")
        
        conn = psycopg2.connect(db_connection_string)
        
        query = """
            SELECT 
                user_id,
                track_id,
                interaction_type,
                COUNT(*) as interaction_count,
                AVG(COALESCE(duration, 0)) as avg_duration
            FROM interactions
            WHERE user_id IS NOT NULL AND track_id IS NOT NULL
            GROUP BY user_id, track_id, interaction_type
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df):,} raw interaction records")
        
        # Pivot to get counts per interaction type
        interaction_pivot = df.pivot_table(
            index=['user_id', 'track_id'],
            columns='interaction_type',
            values='interaction_count',
            fill_value=0
        ).reset_index()
        
        # Calculate normalized score (weighted combination)
        df = interaction_pivot
        df['normalized_score'] = (
            df.get('play', 0) * 1.0 +
            df.get('like', 0) * 3.0 +
            df.get('playlist_add', 0) * 2.0 +
            df.get('share', 0) * 2.5 -
            df.get('skip', 0) * 0.5
        ).clip(0, 100)  # Clip to 0-100 range
        
        # Normalize to 0-5 scale
        if df['normalized_score'].max() > 0:
            df['normalized_score'] = 5 * df['normalized_score'] / df['normalized_score'].max()
        
        # Filter users with minimum interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        logger.info(f"After filtering (min {min_interactions} interactions): {len(df):,} interactions")
        logger.info(f"Unique users: {df['user_id'].nunique():,}")
        logger.info(f"Unique tracks: {df['track_id'].nunique():,}")
        
        # Sample if requested
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            logger.info(f"Sampled {sample_frac*100:.1f}%: {len(df):,} interactions")
        
        return df[['user_id', 'track_id', 'normalized_score']]
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load interaction data from CSV file.
        
        Args:
            csv_path: Path to CSV file with user_id, track_id, normalized_score
            
        Returns:
            DataFrame with interactions
        """
        logger.info(f"Loading interactions from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        required_cols = ['user_id', 'track_id', 'normalized_score']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        logger.info(f"Loaded {len(df):,} interactions")
        logger.info(f"Unique users: {df['user_id'].nunique():,}")
        logger.info(f"Unique tracks: {df['track_id'].nunique():,}")
        
        return df[required_cols]
    
    def create_mappings(self, df: pd.DataFrame):
        """Create ID mappings for users and items."""
        logger.info("Creating ID mappings...")
        
        unique_users = df['user_id'].unique()
        unique_items = df['track_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        logger.info(f"Mapped {len(self.user_id_map):,} users and {len(self.item_id_map):,} items")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        batch_size: int = 512
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train/validation data loaders.
        
        Args:
            df: DataFrame with user_id, track_id, normalized_score
            test_size: Validation set proportion
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Preparing data loaders...")
        
        # Create mappings
        self.create_mappings(df)
        
        # Map IDs
        df['user_idx'] = df['user_id'].map(self.user_id_map)
        df['item_idx'] = df['track_id'].map(self.item_id_map)
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
        
        logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
        
        # Create datasets
        train_dataset = InteractionDataset(
            train_df['user_idx'].values,
            train_df['item_idx'].values,
            train_df['normalized_score'].values
        )
        
        val_dataset = InteractionDataset(
            val_df['user_idx'].values,
            val_df['item_idx'].values,
            val_df['normalized_score'].values
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == "cuda" else False
        )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Train the CF model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
        """
        # Initialize model
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        
        logger.info(f"Initializing NCF model (users={num_users}, items={num_items}, dim={self.embedding_dim})")
        
        self.model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        )
        
        self.trainer = CFTrainer(
            self.model,
            learning_rate=self.learning_rate,
            device=self.device
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss = self.trainer.train_epoch(train_loader)
            val_loss = self.trainer.evaluate(val_loader)
            
            log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                log_msg += " ✓ Best!"
                
                # Save checkpoint
                if checkpoint_dir:
                    self._save_checkpoint(checkpoint_dir, epoch, val_loss)
            else:
                patience_counter += 1
                log_msg += f" | Patience: {patience_counter}/{early_stopping_patience}"
            
            logger.info(log_msg)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training complete! Best Val Loss: {best_val_loss:.4f}")
    
    def _save_checkpoint(self, checkpoint_dir: str, epoch: int, val_loss: float):
        """Save training checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_path / f"checkpoint_epoch{epoch+1}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
            'embedding_dim': self.model.embedding_dim,
            'hidden_layers': self.hidden_layers
        }, checkpoint_file)
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save trained model and metadata.
        
        Args:
            output_dir: Directory to save model (uses default if None)
        """
        if output_dir:
            save_dir = Path(output_dir)
        else:
            save_dir = self.model_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model weights
        model_file = save_dir / f"ncf_model_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
            'embedding_dim': self.model.embedding_dim,
            'hidden_layers': self.hidden_layers
        }, model_file)
        
        logger.info(f"✓ Model saved: {model_file}")
        
        # Save ID mappings
        mappings_file = save_dir / f"id_mappings_{timestamp}.pkl"
        with open(mappings_file, 'wb') as f:
            pickle.dump({
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_item_map': self.reverse_item_map
            }, f)
        
        logger.info(f"✓ ID mappings saved: {mappings_file}")
        
        # Save metadata
        metadata_file = save_dir / f"metadata_{timestamp}.json"
        metadata = {
            'timestamp': timestamp,
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
            'embedding_dim': self.embedding_dim,
            'hidden_layers': self.hidden_layers,
            'device': self.device,
            'model_file': str(model_file.name),
            'mappings_file': str(mappings_file.name)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved: {metadata_file}")
        
        return model_file, mappings_file, metadata_file


def main():
    parser = argparse.ArgumentParser(description="Train Collaborative Filtering Model")
    
    # Data source
    parser.add_argument('--data-source', type=str, choices=['database', 'csv'], required=True,
                       help='Data source: database or csv')
    parser.add_argument('--db-connection', type=str,
                       help='PostgreSQL connection string (for database source)')
    parser.add_argument('--csv-path', type=str,
                       help='Path to CSV file (for csv source)')
    
    # Data options
    parser.add_argument('--min-interactions', type=int, default=5,
                       help='Minimum interactions per user')
    parser.add_argument('--sample-frac', type=float, default=1.0,
                       help='Fraction of data to sample (0.0-1.0)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Validation set proportion')
    
    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                       default='./models/collaborative/model_weights',
                       help='Output directory for model')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Directory to save training checkpoints')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.data_source == 'database' and not args.db_connection:
        parser.error("--db-connection required when --data-source is database")
    if args.data_source == 'csv' and not args.csv_path:
        parser.error("--csv-path required when --data-source is csv")
    
    # Initialize trainer
    trainer = ProductionCFTrainer(
        embedding_dim=args.embedding_dim,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        device=args.device,
        model_dir=args.output_dir
    )
    
    # Load data
    if args.data_source == 'database':
        df = trainer.load_from_database(
            args.db_connection,
            min_interactions=args.min_interactions,
            sample_frac=args.sample_frac
        )
    else:
        df = trainer.load_from_csv(args.csv_path)
    
    # Prepare data loaders
    train_loader, val_loader = trainer.prepare_data(
        df,
        test_size=args.test_size,
        batch_size=args.batch_size
    )
    
    # Train model
    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save model
    model_file, mappings_file, metadata_file = trainer.save_model(args.output_dir)
    
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info("="*50)
    logger.info(f"Model: {model_file}")
    logger.info(f"Mappings: {mappings_file}")
    logger.info(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
