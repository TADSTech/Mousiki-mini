import numpy as np
from typing import Dict, List, Optional


class EpsilonGreedy:
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.0, seed: Optional[int] = None):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(seed)

    def select(
        self, candidates: List[int], scores: Dict[int, float], top_k: int = 10
    ) -> List[int]:
        if self.rng.random() < self.epsilon:
            return list(self.rng.choice(candidates, size=min(top_k, len(candidates)), replace=False))
        sorted_items = sorted(candidates, key=lambda x: scores.get(x, 0.0), reverse=True)
        return sorted_items[:top_k]

    def step(self):
        self.epsilon *= (1.0 - self.decay_rate)

    def get_current_epsilon(self) -> float:
        return self.epsilon
