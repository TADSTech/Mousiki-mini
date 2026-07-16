import numpy as np
import pandas as pd
from typing import List, Optional
import random


class BaselineRanker:
    def fit_popularity(self, interactions: pd.DataFrame):
        pop = interactions.groupby('track_id')['normalized_score'].mean().to_dict()
        self.popularity_scores_ = pop

    def rank_by_popularity(self, candidates: List[int], top_k: int = 10) -> List[int]:
        scored = [(t, self.popularity_scores_.get(t, 0.0)) for t in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_k]]


class RandomRanker:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def rank(self, candidates: List[int], top_k: int = 10) -> List[int]:
        if self.seed is not None:
            rng = random.Random(self.seed)
            return rng.sample(candidates, min(top_k, len(candidates)))
        return random.sample(candidates, min(top_k, len(candidates)))
