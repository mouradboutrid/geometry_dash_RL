Reward Shaping
==============

Detailed explanation of reward functions for each physics mode.

Overview
--------

Dense reward shaping is critical for learning in sparse-reward environments. We design separate reward functions for Cube and Ship modes, each exploiting mode-specific dynamics.

**Core Insight**: Reward signal should guide exploration toward successful policies without over-constraining behavior.

Cube Mode Expert
----------------

The CubeExpert class provides mode-tailored rewards for Slices 1, 2, 3, 5, 6, 8.

.. code-block:: python

   class CubeExpert:
       """
       SOTA-aligned reward shaping for Cube jumping mechanics.
       Focus: Forward progress + survival + precision.
       """
       
       @staticmethod
       def get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None):
           if reward_context is None:
               reward_context = {}
           
           # Tuned coefficients (found via experimentation)
           progress_scale = float(reward_context.get("progress_scale", 20.0))
           step_penalty = float(reward_context.get("step_penalty", 0.0001))
           jump_penalty = float(reward_context.get("jump_penalty", 0.0005))
           spam_jump_penalty = float(reward_context.get("spam_jump_penalty", 0.001))
           clearance_bonus = float(reward_context.get("clearance_bonus", 0.01))
           hazard_proximity_threshold = float(reward_context.get("hazard_proximity_threshold", 30.0))
           landing_bonus = float(reward_context.get("landing_bonus", 20))
           
           reward = 0.0
           
           # 1. Forward Progress (Main Signal)
           progress_delta = state.percent - prev_percent
           if progress_delta > 0:
               reward += progress_delta * progress_scale
           
           # 2. Per-Step Penalty (Anti-Idle)
           reward -= step_penalty
           
           # 3. Jump Efficiency Penalty
           prev_action = reward_context.get("prev_action", None)
           if action != 0:  # Action 1 = holding/jumping
               reward -= jump_penalty
               if prev_action == 1:  # Consecutive jump
                   reward -= spam_jump_penalty
           
           # 4. Hazard Clearance Bonus (Optional)
           if prev_dist_nearest_hazard is not None and hasattr(state, "dist_nearest_hazard"):
               if (state.dist_nearest_hazard > prev_dist_nearest_hazard and 
                   prev_dist_nearest_hazard < hazard_proximity_threshold):
                   reward += clearance_bonus
           
           # 5. Landing on Blocks (Strategic Jumps)
           if hasattr(state, "dy_block") and hasattr(state, "dy_player"):
               if state.dy_block > state.dy_player:  # Block higher than player
                   reward += landing_bonus
           
           return reward

**Component Breakdown**

.. list-table:: Cube Reward Terms
   :header-rows: 1

   * - Component
     - Formula
     - Coefficient
     - Motivation
   * - Progress
     - Δ% × scale
     - 20.0
     - Primary objective
   * - Step Penalty
     - -penalty
     - 0.0001
     - Discourage idling
   * - Jump Penalty
     - -penalty
     - 0.0005
     - Encourage efficient jumps
   * - Spam Penalty
     - -penalty
     - 0.001
     - Prevent button mashing
   * - Clearance Bonus
     - +bonus
     - 0.01
     - Reward hazard avoidance
   * - Landing Bonus
     - +bonus
     - 20
     - Reward strategic positioning

**Why These Coefficients?**

.. code-block:: text

   Progress Scale = 20.0:
   - Progress from 0% → 10% yields +200 reward
   - Dominates other signals
   - Ensures agent prioritizes completion
   
   Jump Penalty = 0.0005:
   - Small enough to allow exploration
   - Large enough to discourage spam
   - 1000 jumps per episode × 0.0005 = -0.5 total penalty
   
   Step Penalty = 0.0001:
   - Anti-idle mechanism
   - 300 steps per episode × 0.0001 = -0.03 penalty
   - Does not suppress action taking
   
   Clearance Bonus = 0.01:
   - Minimal (1/2000 of progress bonus)
   - Guides exploration without dominating
   - Encourages safe trajectories

Ship Mode Expert
----------------

For Slices 4, 7 (Ship physics):

.. code-block:: python

   class ShipExpert:
       """
       Reward shaping for Ship (Stereo Madness).
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

**Why Different from Cube?**

Ship mode requires:

1. **Stability**: Minimize |vel_y| (vertical velocity)
2. **Centering**: Stay at y=235 (corridor center)
3. **Smooth Transitions**: Avoid oscillations
4. **Different Control Model**: Tap to flip, not hold to jump

Result: Reward function explicitly targets these behaviors.

Comparison Example
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   SAME TRAJECTORY (agent navigates correctly):
   
   Cube Reward:
   ├─ Progress (0% → 5%): +100
   ├─ 10 steps: -0.001
   ├─ 2 jumps: -0.001
   └─ Total: ~+99.98
   
   Ship Reward:
   ├─ Progress (30% → 35%): +50
   ├─ Survival: +0.05
   ├─ Centering (near corridor): +8
   ├─ Low velocity (stable): 0 (no penalty)
   └─ Total: ~+58.05
   
   Both positive, but different magnitudes reflect difficulty differences.

Reward Shaping Challenges
--------------------------

**Challenge 1: Scale Imbalance**

Small coefficients (0.0001) vs large (20.0) can cause:
- Numerical instability (underflow/overflow)
- Difficult optimization (mixed scales)

**Solution**: Normalize rewards at episode end.

.. code-block:: python

   episode_rewards = [r1, r2, r3, ...]  # Raw rewards
   mean_reward = mean(episode_rewards)
   std_reward = std(episode_rewards)
   
   normalized_rewards = [
       (r - mean_reward) / (std_reward + 1e-8)
       for r in episode_rewards
   ]

**Challenge 2: Sparse High-Magnitude Rewards**

Progress bonus (e.g., +100 for 5% completion) can dominate learning.

**Solution**: Use relative progress, not absolute.

.. code-block:: python

   # Current (used):
   progress_delta = state.percent - prev_percent
   reward += progress_delta * 20.0
   
   # NOT: reward += state.percent * 2.0 (absolute progress, biased)

**Challenge 3: Hyperparameter Sensitivity**

Small tuning changes cause large performance swings:

.. code-block:: text

   progress_scale = 15.0: Converges in 50 episodes
   progress_scale = 20.0: Converges in 45 episodes (optimal)
   progress_scale = 25.0: Converges in 50 episodes
   progress_scale = 10.0: Fails (explores inefficiently)
   progress_scale = 50.0: Fails (overestimates progress value)

**Solution**: Curriculum learning reduces sensitivity. Reward shapes are less critical with good curriculum.

Potential Improvements
----------------------

**1. Learned Reward Functions (Inverse RL)**

Instead of manual design, learn from human demonstrations:

.. code-block:: python

   # Collect human playthroughs
   trajectories = collect_from_human(level, num=10)
   
   # Inverse RL: Infer reward function from trajectories
   reward_fn = train_inverse_rl(trajectories)
   
   # Use inferred reward for RL training
   agent = train_dqn(level, reward_fn=reward_fn)

**Benefits**:
- Objective (data-driven)
- Generalizable (same human can demonstrate multiple levels)
- Removes tuning burden

**Timeline**: 3-6 months development

**2. Curriculum over Rewards**

Use reward shaping conservatively; let curriculum provide structure:

.. code-block:: python

   # Minimal rewards
   reward = {
       'progress': progress_delta * 5.0,  # Reduced from 20.0
       'death': -1.0 if terminated else 0.0,  # Explicit death penalty
   }
   # Curriculum handles difficulty scaling
   # Simpler reward function more general across levels

**Benefits**:
- Simpler, more interpretable
- Potentially better transfer learning
- Less hyperparameter tuning

**Current Approach**: Balanced (moderate reward shaping + curriculum).

**3. Auxiliary Tasks**

Train auxiliary heads to predict auxiliary signals:

.. code-block:: python

   # Network predicts:
   # ├─ Q-values (main task)
   # ├─ Next frame prediction (auxiliary)
   # ├─ Mode detection (auxiliary)
   # └─ Depth estimation (auxiliary)
   
   # Auxiliary losses guide feature learning
   # Improves generalization and interpretability

**Benefits**:
- Richer feature representations
- Self-supervised learning
- Better transfer across levels

**Timeline**: 4-6 months (significant architectural change)

Practical Tuning Guide
----------------------

If agent gets stuck on a slice:

1. **Check if progress reward too low**:

   .. code-block:: python

      # Increase progress_scale
      progress_scale = 30.0  # was 20.0
      # Retrain; agent should make faster progress

2. **Check if jump penalty too high**:

   .. code-block:: python

      # Reduce jump penalty
      jump_penalty = 0.0001  # was 0.0005
      # Agent encouraged to explore more actions

3. **Check if step penalty too high**:

   .. code-block:: python

      # Reduce step penalty
      step_penalty = 0.0  # Disable
      # Check if agent still explores (should, via epsilon-greedy)

4. **Add clearance bonus if high failure rate on hazards**:

   .. code-block:: python

      # Increase safety incentive
      clearance_bonus = 0.05  # was 0.01

**Remember**: Reward shaping is an art. Start with curriculum (best leverage) before tuning rewards.

See :doc:`../performance` for empirical reward contribution analysis.
