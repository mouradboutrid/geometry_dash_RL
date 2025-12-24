class ShipExpert:
    """
    Expert Logic for Ship Mode (Slice 4, 9).
    Focus: Stability, Centering, Flow.
    """
    
    @staticmethod
    def get_reward(state, action, prev_percent):
        reward = 0.0

        # Survival Reward
        reward += 0.05  # small reward for staying alive

        # Progress Reward
        delta = state.percent - prev_percent
        if delta > 0:
            reward += delta * 10.0  # scaled heavily to encourage forward progress

        # Centering / Safety Corridor Reward
        target_y = 235.0
        dist_from_center = abs(state.player_y - target_y)
        if dist_from_center < 15:
            reward += 10 * (1 - dist_from_center / 50.0)
        else:
            reward -= 10 * (dist_from_center / 200.0)
        # Instability Penalty
        if abs(state.player_vel_y) > 3.20:
            reward -= 2000  # penalize jerky vertical motion

        # Precision Bonus near Obstacles
        if hasattr(state, "dy_nearest_hazard"):
            if abs(state.dy_nearest_hazard) <= 30:
                reward += 3

        # Passed Obstacles / Blocks Bonus
        if hasattr(state, "passed_obstacles"):
            # Assuming 'passed_obstacles' increments every time a block/obstacle is passed
            reward += state.passed_obstacles * 20

        return reward

    @staticmethod
    def should_reset_weights(prev_mode):
        """
        Logic for switching INTO Ship mode.
        Reset weights if switching from Cube (Mode 0).
        """
        return prev_mode == 0
