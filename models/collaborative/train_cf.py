"""
Collaborative Filtering Training Module

Trains neural collaborative filtering models on user-item interactions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionDataset(Dataset):
    """Dataset for user-item interactions."""
    
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering model."""
    
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int = 128,
        hidden_layers: list = [256, 128, 64]
    ):
        """
        Initialize NCF model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of user/item embeddings
            hidden_layers: List of hidden layer dimensions
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass.
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            Predicted ratings
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        return output.squeeze()
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding for a specific user."""
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            embedding = self.user_embedding(user_tensor)
            return embedding.numpy()[0]
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get embedding for a specific item."""
        with torch.no_grad():
            item_tensor = torch.LongTensor([item_id])
            embedding = self.item_embedding(item_tensor)
            return embedding.numpy()[0]


class CFTrainer:
    """Trainer for collaborative filtering models."""
    
    def __init__(
        self,
        model: NeuralCollaborativeFiltering,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for user_ids, item_ids, ratings in dataloader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        save_path: str = "./models/collaborative/model_weights/ncf_model.pt"
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            save_path: Path to save best model
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
            
            if val_loader:
                val_loss = self.evaluate(val_loader)
                log_msg += f" - Val Loss: {val_loss:.4f}"
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_path)
                    log_msg += " - Saved!"
            
            logger.info(log_msg)
    
    def save_model(self, path: str):
        """Save model weights."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.model.num_users,
            'num_items': self.model.num_items,
            'embedding_dim': self.model.embedding_dim
        }, output_path)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def prepare_data(
    interactions_df: pd.DataFrame,
    test_size: float = 0.2,
    batch_size: int = 256
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders from interactions dataframe.
    
    Args:
        interactions_df: DataFrame with user_id, track_id, normalized_score
        test_size: Proportion for validation set
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_df, val_df = train_test_split(interactions_df, test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = InteractionDataset(
        train_df['user_id'].values,
        train_df['track_id'].values,
        train_df['normalized_score'].values
    )
    
    val_dataset = InteractionDataset(
        val_df['user_id'].values,
        val_df['track_id'].values,
        val_df['normalized_score'].values
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def main():
    """Example training script."""
    # Example: Load your interaction data
    # interactions_df = pd.read_csv("./data/processed/interactions_aggregated.csv")
    
    # For demo purposes, create sample data
    sample_data = pd.DataFrame({
        'user_id': np.random.randint(0, 100, 1000),
        'track_id': np.random.randint(0, 500, 1000),
        'normalized_score': np.random.uniform(0, 5, 1000)
    })
    
    # Prepare data
    train_loader, val_loader = prepare_data(sample_data)
    
    # Initialize model
    num_users = sample_data['user_id'].nunique()
    num_items = sample_data['track_id'].nunique()
    
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=128
    )
    
    # Train model
    trainer = CFTrainer(model)
    trainer.fit(train_loader, val_loader, epochs=10)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
