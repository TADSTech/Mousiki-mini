class RewardCalculator:
    def compute_immediate_reward(
        self, interaction_type: str, duration: float = 0.0, track_duration: float = 0.0
    ) -> float:
        if interaction_type == 'like':
            return 1.0
        elif interaction_type == 'skip':
            return -0.5
        elif interaction_type == 'play':
            if track_duration > 0:
                ratio = duration / track_duration
                return min(ratio, 1.0)
            return 0.5
        return 0.0
