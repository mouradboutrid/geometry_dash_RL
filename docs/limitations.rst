Known Limitations
=================

This section documents current limitations and their implications.

Inference Latency
-----------------

**Issue**

Real-time neural network inference adds ~4-5 ms per frame. Game runs at 60 FPS (16.67 ms/frame).

.. code-block:: text

   Frame Budget (16.67 ms)
   ├─ Game physics: 10 ms
   ├─ Rendering: 2 ms
   ├─ Network inference: 3-5 ms ← BOTTLENECK
   └─ Total: 15-17 ms (sometimes exceeds budget)

**Impact**

- Occasional frame drops during training
- Agent occasionally misses time-critical decisions
- Performance ceiling: ~70% success (missed frames cause unavoidable failures)

**Mitigation Strategies**

1. **Frame Stacking**: Repeat actions 4 frames → only decide every 67 ms
   - Trade-off: Agent must plan 200+ ms ahead
   - Current: Effective; sufficient temporal window

2. **Model Pruning** (future): Remove <5% importance parameters
   - Potential speedup: 2-3x
   - Would reduce inference to <1 ms

3. **Quantization** (future): Convert float32 → int8
   - Potential speedup: 4-8x
   - Hardware-dependent; requires retraining

4. **Async Inference** (future): Predict in separate thread
   - Non-blocking I/O
   - Requires careful synchronization

**Workaround (Current)**

Frame stacking (action repeat every 4 frames) effectively amortizes latency over 67 ms. Acceptable for current use case.

Limited Generalization
----------------------

**Issue**

Agent trained exclusively on Stereo Madness. Generalization to other levels untested.

.. code-block:: text

   Transfer Learning Challenges
   ├─ Different obstacle layouts
   ├─ Different rhythm patterns
   ├─ Different physics tuning (gravity, speed)
   └─ Different slice decomposition needed (manual per level)

**Current Scope**

- ✓ Handles Stereo Madness variations (game updates)
- ✗ Cannot play other Geometry Dash levels
- ✗ Reward shaping is Stereo Madness-specific

**Impact**

Agent is specialized, not generalized. Development of other levels requires:
1. Manual level analysis and slice definition
2. Reward shaping tuning per mode
3. Retraining from scratch or fine-tuning

Estimated effort: 8-16 hours per new level (including manual design). (without training ofcourse)

**Mitigation**

1. **Domain Adaptation** (future): Train reward predictor that generalizes across levels
   - Use pre-trained features from Stereo Madness
   - Fine-tune on new level (10% data)
   - Potential 50% reduction in training time

2. **Automatic Slice Discovery** (future): Meta-learning to find curriculum structure
   - Analyze level geometry automatically
   - Propose slices based on learning progress
   - Enable transfer across level-like segments

3. **Imitation Learning** (future): Pre-train on human playthroughs
   - Human demonstrates level
   - Agent learns from demo via behavioral cloning
   - Reduces exploration from 18 hours to 4-6 hours

Mode-Switch Performance Dips
-----------------------------

**Issue**

Physics transitions (Cube → Ship or Ship → Cube) cause temporary performance degradation.

.. code-block:: text

   Success Rate Around Mode Transitions
   
   Slice 3 (End Cube): 72% ┐
                          │ DROP: -15%
   Slice 4 (Ship):     57% ┴─ (episode 1-5 on new mode)
                          │ RECOVERY: +18% as agent adapts
   Slice 4 (End Ship): 75% ┘

**Root Cause**

Physics transition requires:
- Forgetting jump-timing intuitions (useless for ship)
- Learning velocity-balancing (specific to ship)
- Exploring unfamiliar action consequences
- Rebuilding reward prediction model

**Impact**

~5-10 extra episodes per mode switch (2 switches = 10-20 episode penalty).

**Mitigation**

1. **Mode-Aware Architecture** (future): Separate heads for different modes
   - Shared trunk: Generic features
   - Cube head: Jump-timing specialization
   - Ship head: Stability specialization
   - Benefit: No weight reset; gradual adaptation

2. **Multi-Task Learning** (future): Auxiliary task to predict mode
   - Network learns: "Am I in Cube or Ship mode?"
   - Implicit mode representation in hidden layers
   - Faster transfer when mode changes

3. **Domain Randomization** (future): Train with random gravity/physics
   - Builds robust representations
   - Better transfer to unseen physics

**Current Workaround**

Fresh network + full exploration on mode change. Acceptable; ~38 episodes per ship slice is manageable.

Deterministic Environment
-------------------------

**Issue**

Game engine is deterministic. Same input sequence always produces same output.

.. code-block:: text

   Determinism Implications
   ├─ Agent can memorize exact sequences
   ├─ No robustness to input noise
   ├─ No resilience to lag spikes
   ├─ Policy overfits to precise timing
   └─ Real-world performance unknown

**Impact**

- In-game success ≠ Real-world success under perturbations
- Agent brittle to lag, input delays, physics patches
- Cannot test robustness directly

**Evidence**

If I add ±50 ms input lag:
- Agent success rate: 50% → X% <<50% (failure)
- Shows heavy overfitting to precise timing

**Mitigation**

1. **Domain Randomization** (future): Train with noisy inputs
   - Add ±20 ms random action latency
   - Add ±5% random physics perturbations
   - Improves robustness to real-world variations
   - Estimated cost: +30% training time

2. **Ensemble Methods** (future): Train multiple agents with different seeds
   - Average predictions
   - Reduced sensitivity to initialization noise
   - Natural defense against overfitting

3. **Uncertainty Quantification** (future): Predict confidence intervals
   - When agent unsure, take conservative action
   - Better handles edge cases

**Current Workaround**

None. Assume deterministic environment. Useful for development; not deployed in real-world.

Reward Shaping Brittleness
---------------------------

**Issue**

Reward functions manually designed per mode. Poor hyperparameter choices lead to failure.

.. code-block:: python

   # Example: What if you set jump_penalty too high?
   jump_penalty = 0.1  # Too high!
   
   # Agent learns: "Never jump!" (safer but useless)
   # Result: Dies at first obstacle
   
   # What if progress_scale too low?
   progress_scale = 1.0  # Too low!
   
   # Agent learns: "Explore randomly, make slow progress"
   # Result: Stagnates on Triple Spike (45 episodes → 200 episodes)

**Current Hyperparameters** (well-tuned via trial-and-error)

.. code-block:: text

   Cube Expert:
   ├─ progress_scale: 20.0    ✓ (balanced)
   ├─ jump_penalty: 0.0005    ✓ (allows exploration)
   ├─ spam_penalty: 0.001     ✓ (discourages rapid jumping)
   └─ clearance_bonus: 0.01   ✓ (encourages safety)
   
   Ship Expert:
   ├─ progress_scale: 10.0    ✓ (slightly lower; harder to progress)
   ├─ stability_penalty: 2000  ✓ (discourages velocity spikes)
   ├─ centerline_bonus: 10.0  ✓ (encourages centering)
   └─ survival_reward: 0.05   ✓ (encourages endurance)

**Impact**

If hyperparameters suboptimal:
- Best case: 50% more training episodes needed
- Worst case: Agent fails to learn (infinite training)

**Mitigation**

1. **Inverse Reinforcement Learning** (future): Learn reward from human demos
   - Human demonstrates level
   - Inverse RL extracts reward function
   - Objective, learnable (vs. manual design)
   - Estimated cost: Requires 5-10 human playthroughs

2. **Automated Reward Tuning** (future): Meta-learning for hyperparameters
   - Train multiple agents with different reward coefficients
   - Meta-learner predicts best coefficients
   - Faster convergence for new levels

3. **Curriculum over Rewards** (current workaround)
   - Simpler rewards (just progress + survival)
   - Curriculum provides structure
   - Works; requires good slice design

Limited Exploration
-------------------

**Issue**

Epsilon-greedy exploration is basic. Complex environments might need sophisticated exploration.

.. code-block:: text

   Epsilon-Greedy Limitations
   ├─ Uniform random actions (no structure)
   ├─ No exploitation of curiosity
   ├─ No intrinsic motivation
   ├─ Exploration diminishes over time (epsilon → 0.01)
   └─ May miss important state regions

**Current Effectiveness**

Epsilon-greedy sufficient for Stereo Madness because:
- Only 2 actions (limited action space)
- Reward signal strong enough (curriculum provides dense rewards)
- Slice difficulty bounded (not exponentially hard)

**Would Break On**

- Huge action spaces (e.g., continuous control with 20+ DOF)
- Sparse rewards without curriculum
- Exploration traps (dead ends with no rewards)

**Impact**

N/A for current task. Curriculum learning compensates. If removing curriculum, would hit exploration ceiling.

**Mitigation**

1. **Curiosity-Driven Exploration** (future): Intrinsic motivation
   - Agent curious about novel states
   - Self-motivated exploration
   - Better discovery of complex patterns

2. **Count-Based Exploration** (future): Bonus for visiting rare states
   - Episodic memory tracks visited states
   - Bonus reward for exploration
   - Directed exploration

3. **Thompson Sampling** (future): Uncertainty-aware exploration
   - Maintains posterior over Q-values
   - Explores uncertain regions
   - Theoretically grounded

**For Geometry Dash, curriculum + epsilon-greedy sufficient.**

Checkpoint Management
---------------------

**Issue**

Manual checkpoint saving (every 50 episodes). Risk of data loss if training crashes mid-slice.

.. code-block:: text

   Risk Timeline
   ├─ Episode 300 (of 345): Save slice_03_current.pth
   ├─ Episode 31-45: No checkpoints created
   ├─ Episode 44: CPU driver crash
   ├─ Result: Lose episodes restart from episode 300
   └─ Wasted time (For stereoMadness it not mush but for longer levels it became a problem)

**Current Safeguards**

- Auto-save every 50 episodes
- Manual save on Ctrl+C
- Progress saved to JSON every slice advance

**Insufficient Because**

- 500-episode intervals can lose ~40 min of work
- CPU crashes without warning
- No rollback mechanism

**Mitigation**

1. **Continuous Checkpointing** (future):
   - Save after every episode (JSON metric)
   - Keep rolling buffer (last 10 checkpoints)
   - Fast recovery to within 1 episode

2. **Cloud Backup** (future):
   - Sync checkpoints to cloud storage
   - Redundancy against hardware failure
   - Enable distributed training

3. **Distributed Training** (future):
   - Multiple workers on different machines
   - Automatic failover if one crashes
   - Faster training + fault tolerance

**Current Workaround**

Manual checkpoint discipline. Save frequently; monitor training loop.

System Compatibility
--------------------

**Issue**

C++ Geode mod requires:
- Windows 10/11 (no Linux, macOS)
- Specific Geometry Dash version (v2.2074)
- Admin privileges (memory mapping)

**Impact**

Not portable. Limits use to Windows systems with access to specific GD version.

**Mitigation**

1. **Web-based Version** (future): Run GD in browser (WebGL)
   - No native mod required
   - Cross-platform
   - Significant engineering effort

2. **Docker Container** (future): Pre-configured Windows VM
   - One-click setup
   - Reproducible environment
   - Supports remote training

3. **Multi-Version Support** (future): Adapt mod to multiple GD versions
   - Version detection + conditional code paths
   - Maintains compatibility as GD updates

**Current Scope: Windows 10/11 + GD 2.2074 only.**
