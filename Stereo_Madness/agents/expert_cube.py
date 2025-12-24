class CubeExpert:
    """
    SOTA-aligned reward shaping for Cube (Stereo Madness).
    Focus: survival + forward progress, minimal bias.
    """

    @staticmethod
    def get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None):
        if reward_context is None:
            reward_context = {}

        # ---- Tuned parameters for convergence & exploration ----
        # Stronger progress reward to push forward and avoid local minima
        progress_scale = float(reward_context.get("progress_scale", 20.0))
        # Lower per-step penalty to encourage exploration
        step_penalty = float(reward_context.get("step_penalty", 0.0001))
        # Reduced jump penalty to allow necessary jumps
        jump_penalty = float(reward_context.get("jump_penalty", 0.0005))
        # Lower spam penalty to not over-discourage defensive jumps
        spam_jump_penalty = float(reward_context.get("spam_jump_penalty", 0.001))
        # Increased clearance bonus to reward hazard avoidance
        clearance_bonus = float(reward_context.get("clearance_bonus", 0.01))
        hazard_proximity_threshold = float(reward_context.get("hazard_proximity_threshold", 30.0))
        # Landing on obstacles higher than player
        landing_bonus = float(reward_context.get("landing_bonus", 20))
        # Hazard cleaning reward (scales with count: 10, 20, 30... max 50)
        hazards_cleaned_count = int(reward_context.get("hazards_cleaned_count", 0))

        reward = 0.0

        # 1️⃣ Forward progress (MAIN SIGNAL)
        progress_delta = state.percent - prev_percent
        if progress_delta > 0:
            reward += progress_delta * progress_scale

        # 2️⃣ Small per-step penalty (anti-idle)
        reward -= step_penalty

        # 3️⃣ Jump efficiency penalty
        prev_action = reward_context.get("prev_action", None)
        if action != 0:
            reward -= jump_penalty
            if prev_action == 1:
                reward -= spam_jump_penalty

        # 4️⃣ Optional clearance shaping (VERY SMALL)
        if prev_dist_nearest_hazard is not None and hasattr(state, "dist_nearest_hazard"):
            if (
                state.dist_nearest_hazard > prev_dist_nearest_hazard and
                prev_dist_nearest_hazard < hazard_proximity_threshold
            ):
                reward += clearance_bonus

        # 5️⃣ Landing on obstacles/blocks higher than player (strategic jumps)
        if hasattr(state, "dy_block") and hasattr(state, "dy_player"):
            # dy_block > dy_player means block is higher (lower Y value)
            if state.dy_block > state.dy_player:
                reward += landing_bonus

        # 6️⃣ Hazard cleaning reward (if within ±20 dy and agent passes/survives)
        if prev_dist_nearest_hazard is not None and hasattr(state, "dist_nearest_hazard"):
            prev_d = prev_dist_nearest_hazard
            curr_d = state.dist_nearest_hazard
            # Check if hazard was passed (prev > 0, curr <= 0) and within vertical range
            crossed = (prev_d > 0 and curr_d <= 0)
            if crossed and hasattr(state, "dy_hazard"):
                # Within ±20 dy means the hazard was vertically aligned
                if abs(state.dy_hazard) <= 20:
                    # Reward scales: 10, 20, 30... capped at 50
                    cleaning_reward = min(10 + (hazards_cleaned_count * 10), 50)
                    reward += cleaning_reward
                    # Increment counter for next hazard
                    reward_context["hazards_cleaned_count"] = hazards_cleaned_count + 1

        return reward

    @staticmethod
    def should_reset_weights(prev_mode):
        return prev_mode == 1
