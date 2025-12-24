class ShipExpert:
    """
    aligned reward shaping for Ship (Stereo Madness).
    Focus: survival, forward progress, vertical stability around Y=235.
    Minimal bias, DDQN-safe.
    """

    SHIP_CENTER_Y = 235.0  # <- TRUE vertical center for ship corridor

    @staticmethod
    def get_reward(
        state,
        action,
        prev_percent,
        prev_dist_nearest_hazard=None,
        reward_context=None
    ):
        if reward_context is None:
            reward_context = {}

        # Tuned parameters (Ship-specific)
        progress_scale = float(reward_context.get("progress_scale", 20.0))
        step_penalty = float(reward_context.get("step_penalty", 0.0001))

        thrust_penalty = float(reward_context.get("thrust_penalty", 0.0003))
        spam_thrust_penalty = float(reward_context.get("spam_thrust_penalty", 0.0005))

        vertical_stability_bonus = float(
            reward_context.get("vertical_stability_bonus", 0.01)
        )

        clearance_bonus = float(reward_context.get("clearance_bonus", 0.005))
        hazard_proximity_threshold = float(
            reward_context.get("hazard_proximity_threshold", 30.0)
        )

        death_penalty = float(reward_context.get("death_penalty", 5.0))

        reward = 0.0

        # Forward progress (MAIN SIGNAL)
        progress_delta = state.percent - prev_percent
        if progress_delta > 0:
            reward += progress_delta * progress_scale

        # Small per-step penalty
        reward -= step_penalty

        # Thrust efficiency penalty
        prev_action = reward_context.get("prev_action", None)
        if action != 0:
            reward -= thrust_penalty
            if prev_action == 1:
                reward -= spam_thrust_penalty

        # Vertical stability around Y=235 (TINY shaping)
        if hasattr(state, "y"):
            dy_center = state.y - ShipExpert.SHIP_CENTER_Y
            # Smooth reward ∈ [0, vertical_stability_bonus]
            reward += vertical_stability_bonus * max(
                0.0, 1.0 - abs(dy_center) / 80.0
            )

        # Optional clearance shaping (very small)
        if prev_dist_nearest_hazard is not None and hasattr(
            state, "dist_nearest_hazard"
        ):
            if (
                state.dist_nearest_hazard > prev_dist_nearest_hazard
                and prev_dist_nearest_hazard < hazard_proximity_threshold
            ):
                reward += clearance_bonus

        # Death penalty
        if hasattr(state, "dead") and state.dead:
            reward -= death_penalty

        # Safety clamp
        reward = max(min(reward, 10.0), -10.0)

        return reward

    @staticmethod
    def should_reset_weights(prev_mode):
        # Reset when switching from Cube -> Ship
        return prev_mode == 0
