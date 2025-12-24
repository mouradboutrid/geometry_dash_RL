# Geometry Dash Reinforcement Learning Agent

## Summary

This project implements a deep reinforcement learning agent that learns to play Geometry Dash (v2.2074) through curriculum-based training with expertise specialization. The system integrates native C++ game hooking via the Geode modding framework with a sophisticated Python-based training pipeline, enabling the agent to master progressively complex level sections through dueling double Q-learning with mode-aware policy transfer.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technical Implementation](#technical-implementation)
4. [Algorithm Selection & Justification](#algorithm-selection--justification)
5. [Curriculum Learning Strategy](#curriculum-learning-strategy)
6. [Getting Started](#getting-started)
7. [Results & Performance](#results--performance)
8. [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

### Objectives

The core objectives of this project are:

1. **Master Stereo Madness**: Train an agent to autonomously complete the "Stereo Madness" level in Geometry Dash with high reliability.
2. **Handle Mode Transitions**: Successfully navigate mode changes (Cube → Ship → Cube) within a single playthrough.
3. **Efficient Learning**: Minimize wall-clock training time while maximizing sample efficiency through curriculum design.
4. **Scalability**: Establish a framework generalizable to other Geometry Dash levels.

### Game Environment Context

Geometry Dash presents unique challenges for reinforcement learning:

- **Precise Timing**: Frame-perfect input timing required for obstacle avoidance.
- **Sparse Rewards**: Success only achievable after mastering long sequences of actions (100% completion).
- **Non-Stationary Physics**: Physics behavior changes based on player mode (Cube ≠ Ship).
- **High-Dimensional State**: Real-time rendering of dynamic obstacles requires robust feature extraction.
- **Binary Action Space**: Only two discrete actions (Jump/Hold vs. Release).

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Training Loop                   │
│                  (main.py / GDAgentOrchestrator)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────┼──────────┐
                │          │          │
         ┌──────▼──┐  ┌────▼───┐  ┌───▼────────┐
         │  DDQN   │  │Replay  │  │ Curriculum │
         │ Agent   │  │Buffer  │  │ Manager    │
         └──────┬──┘  └───┬────┘  └─┬──────────┘
                │         │         │
         ┌──────▼─────────▼─────────▼──────┐
         │  Shared Memory Interface        │
         │  (MemoryBridge / mmap)          │
         └──────┬──────────────────────────┘
                │
         ┌──────▼────────────────────────┐
         │  C++ Geode Mod (utridu)       │
         │  - State Collection           │
         │  - Action Injection           │
         │  - Physics Synchronization    │
         └──────┬──────────────────────┘
                │
         ┌──────▼──────────────────────┐
         │  Geometry Dash Engine       │
         │  (Cocos2D-based Game Loop)  │
         └─────────────────────────────┘
```

### Module Hierarchy

#### 1. **Python Core** (`Stereo_Madness/`)

**`main.py`** - GDAgentOrchestrator
- Orchestrates training loop and episode management
- Manages expert model cache for curriculum navigation
- Implements relay race logic for level checkpoint handoff
- Coordinates with curriculum manager for slice advancement

**`agents/ddqn.py`** - Dueling DQN Architecture
- Implements dual-stream value/advantage architecture
- Manages online and target networks with periodic synchronization
- Implements Double DQN loss computation with decoupled action selection
- Tracks epsilon-greedy exploration decay

**`agents/replay_buffer.py`** - Experience Memory
- Fixed-size circular buffer storing (state, action, reward, next_state, done) tuples
- Supports uniform random sampling for batch training
- Capacity: 50,000 transitions

**`curriculum/manager.py`** - Curriculum Progression
- Tracks 8 progressive slices of "Stereo Madness"
- Implements promotion criteria (≥70% success rate over 50 episodes, minimum 700 episodes(for faster training else it can be expanded to 90, 95, 100%))
- Maintains rolling success metrics and checkpoint recovery

**`core/environment.py`** - Gymnasium Wrapper
- Wraps Geometry Dash via MemoryBridge
- Implements frame stacking (configurable, default 2x)
- Implements frame skipping (configurable, default 4x)
- Normalizes raw state vectors to [-∞, ∞] float ranges
- Calculates dense reward signals via expert modules

**`core/memory_bridge.py`** - Shared Memory I/O
- Implements cross-language synchronization via Windows named memory mapping
- Provides spinlock-based mutual exclusion between C++ and Python
- Marshals 154-dimensional state vectors and 2D action vectors

**`core/state_utils.py`** - Feature Engineering
- Normalizes player physics (velocity, position, ground contact, mode)
- Processes 30 proximate objects (obstacles, hazards, portals)
- Scales numeric features to unit ranges for neural network stability

**`agents/expert_cube.py`, `expert_ship.py`** - Domain-Specific Reward Shaping
- Defines mode-aware reward functions optimized for physics characteristics
- Cube Expert: Progress-driven with jump efficiency penalties
- Ship Expert: Stability-centric with velocity damping and centerline bonuses

#### 2. **C++ Geode Integration** (`geode/mods/utridu/src/main.cpp`)

**Key Responsibilities:**

- **State Collection**: Extracts player physics, nearby object data, and level progress
- **Real-Time Synchronization**: Writes to shared memory at game loop frequency (~60 Hz)
- **Action Injection**: Translates discrete actions into button press/release events
- **Death/Completion Detection**: Implements stuck-frame detection and terminal state identification
- **Visual Debugging**: Renders HUD with shared memory state and object proximity metrics

**Synchronization Protocol:**

```cpp
// Mutual exclusion via volatile flags
while (pSharedMem->py_writing == 1 && safety < 5000) { safety++; }
pSharedMem->cpp_writing = 1;
// ... write state ...
pSharedMem->cpp_writing = 0;
```

---

## Technical Implementation

### State Representation

**Dimension**: 154 floats

```python
# Player Features (4)
[vel_y/30.0, player_y/900.0, is_on_ground, player_mode]

# Object Features (30 objects × 5 = 150)
For each of 30 nearest objects:
[dx/1000.0, dy/300.0, width/50.0, height/50.0, type/10.0]
```

**Normalization Strategy**:
- Velocity: Assumes max ~30 units/frame
- Position: Assumes playable area ~900 pixels tall
- Object distance: Assumes visibility range ~1000 pixels ahead
- NaN/Inf Safety: Replaced with 0.0/±1.0 clamping

### Action Space

**Discrete(2)**: 
- Action 0: Release jump button (gravity applies)
- Action 1: Hold jump button (active ascending control)

**Implementation**: Stateful transition logic in C++ prevents repeated press/release events for the same logical action.

### Reward Function

**Cube Mode Expert** (Slices 1-3, 5-8):
```python
progress_reward = (percent_delta) × 20.0
step_penalty = -0.0001
jump_penalty = -0.0005
spam_jump_penalty = -0.001 (applied if prev_action was also jump)
clearance_bonus = +0.01 (for hazard avoidance)
landing_bonus = +20 (for strategic block jumps)
hazard_cleaning_reward = min(10 + (count × 10), 50)
```

**Ship Mode Expert** (Slices 4, 9):
```python
progress_reward = (percent_delta) × 10.0
survival_reward = +0.05
centerline_bonus = +10 × (1 - |y - target_y| / 50.0) if within corridor
instability_penalty = -2000 (if |vel_y| > 3.2)
```

**Design Rationale**: 
- Progress signals provide primary gradient during exploration
- Small per-step penalties discourage idling without suppressing learning
- Mode-specific bonuses exploit domain knowledge about optimal play patterns

---

## Algorithm Selection & Justification

### Why Dueling Double Q-Learning (DDQN)?

#### 1. **Advantage Over Simple DQN**

**Problem with Standard DQN**: Suffers from overestimation bias. When the network selects an action based on the online network's estimated Q-values, it often overestimates future rewards, leading to:
- Overconfident action policies
- Poor generalization to new states
- Instability during training

**DDQN Solution - Decoupled Action Selection**:
```python
# Standard DQN (biased)
next_q_standard = target_net(next_state).max(dim=1)

# Double DQN (unbiased estimator)
best_actions = online_net(next_state).argmax(1)  # Use online net for action selection
next_q_ddqn = target_net(next_state).gather(1, best_actions)  # Evaluate with target net
```

This separation reduces bias variance-reduction trade-off and improves sample efficiency - critical when game sessions are expensive (wall-clock time).

#### 2. **Advantage Over A3C (Actor-Critic)**

| Criterion | A3C | DDQN |
|-----------|-----|------|
| Sample Efficiency | Lower (requires >1 worker) | Higher (single-agent) |
| Wall-Clock Training | Slower (parallel overhead) | Faster (sequential) |
| Variance | Lower (parallel stability) | Requires target network stabilization |
| Memory Footprint | High (distributed state) | Lower (centralized buffer) |
| Convergence | Faster (asynchronous) | More stable (off-policy) |

**Decision**: For single-agent, wall-clock-constrained scenarios (human-supervised training), DDQN provides superior sample efficiency without distributed training overhead.

#### 3. **Advantage Over PPO (Policy Gradient)**

| Criterion | PPO | DDQN |
|-----------|-----|------|
| Sample Efficiency | Moderate (~80-90%) | Higher (~95-99%) |
| Hyperparameter Sensitivity | High (clip ratio, learning rate) | Moderate (mainly γ, ε) |
| Action Distribution | Continuous-friendly | Discrete-friendly |
| Exploration | Explicit entropy term | ε-greedy (simpler) |
| Convergence Stability | Better clipping prevents catastrophic failure | Requires target network updates |

**Decision**: Geometry Dash has a discrete, low-dimensional action space (2 actions) where Q-learning excels. PPO's policy gradient approach adds unnecessary complexity without proportional gains.

#### 4. **Advantage Over Simple Q-Learning**

Simple tabular Q-learning is infeasible:
- State space: 2^154 possible states (impossible to enumerate)
- Feature representation: Raw pixel-level state would require convolutional processing

Neural networks approximate Q-function: Q(s, a) → ℝ with generalization.

#### 5. **Dueling Architecture Specifically**

Standard DQN: Q(s, a) - single output per action
Dueling DQN: Q(s, a) = V(s) + A(s, a) - mean_a(A(s, a))

**Benefits**:
- **Value Stream** learns state quality independent of actions (generalizes across action choices)
- **Advantage Stream** learns action-relative value (identifies which actions are superior)
- Improved value estimation when many actions have similar Q-values
- Stronger signal for value-based exploration

**Empirical Result**: Dueling networks show 10-20% faster convergence on discrete control tasks.

---

## Curriculum Learning Strategy

### Motivation

Curriculum learning addresses **credit assignment and exploration challenges**:

**Challenge 1: Sparse Rewards**
- Agent receives meaningful reward only at 100% completion
- Intermediate sections provide 0 reward, causing RL algorithms to fail
- Solution: Train on progressively harder subsections, each providing dense rewards

**Challenge 2: Non-Stationary Policy Requirements**
- Optimal policy for "Triple Spike" (slice 3) is different from "First Ship" (slice 4)
- Physics change mid-level (gravity, controls)
- Solution: Specialize agents per section, reuse learned knowledge where physics are consistent

### Slice Definitions

"Stereo Madness" decomposed into 9 slices (see `curriculum/slice_definitions.json`):

```
Slice 1: 0-10%     | Cube     | Intro Jumps
Slice 2: 10-20%    | Cube     | Pre-Triple Spike
Slice 3: 20-31%    | Cube     | Triple Spike (high precision)
Slice 4: 30-47%    | Ship     | First Ship Section
Slice 5: 47-58%    | Cube     | Mid Level Cube
Slice 6: 58-68%    | Cube     | Mid Level Cube II
Slice 7: 68-77%    | Ship     | Second Ship Section
Slice 8: 77-86%    | Cube     | Clutterfunk section - Complex jumps
Slice 8: 86-100%   | ship     | Final Ship - Victory Lap
```

### Progression Criteria

**Promotion Requirement**:
- Minimum 700 episodes completed on current slice
- Success rate ≥ 70% over last 50 episodes

**Rationale**: 
- 700-episode minimum ensures statistical significance
- 70% threshold balances mastery (agent can handle most patterns) vs. exploration (continues learning corner cases)

### Policy Transfer Mechanism

**Same-Mode Transfer** (Cube → Cube):
```python
if prev_mode == current_mode:
    # Load weights from previous slice
    agent.online_net.load_state_dict(experts_cache[prev_slice_id])
    agent.target_net.load_state_dict(agent.online_net.state_dict())
    agent.epsilon = 0.5  # Resume with moderate exploration
```

**Mode-Switch Transfer** (Cube → Ship or Ship → Cube):
```python
if prev_mode != current_mode:
    # Fresh network for new physics
    agent.reset_network()  # Kaiming uniform initialization
    agent.epsilon = 1.0    # Full exploration
```

**Relay Race Navigation** (Checkpoint Seeding):
Before training a slice, agent uses ensemble of expert models to reach the slice's starting point:
```python
# Navigate from 0% to slice_start using previous experts
for expert_id in sorted_previous_slices:
    if expert_id < current_slice_id and expert_slice_id in position_range:
        agent.online_net.load_state_dict(experts_cache[expert_id])
        # Run episode until next expert's range
```

This ensures training always starts from slice's natural starting point, avoiding artificial task difficulty.

---

## Getting Started

### Prerequisites

- **Hardware**: GPU (NVIDIA with CUDA) recommended; CPU supported (all the result of the repo are result of cpu training (inter core i5))
- **OS**: Windows 10/11 (Geometry Dash + Geode mod)
- **Python**: 3.10+
- **Geometry Dash**: v2.2074 with Geode loader installed

### Installation Steps

#### 1. Install Geode Mod Loader

1. Download [Geode v4.10.0](https://github.com/geode-sdk/geode/releases)
2. Extract to Geometry Dash installation directory
3. Verify `Geometry Dash.exe` in same directory

#### 2. Install utridu Mod

```powershell
# From repository root
cd geode/mods/utridu
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
# Output: build/Release/utridu.geode
# Copy to Geometry Dash/geode/mods/
```

#### 3. Python Environment Setup

```bash
# From Stereo_Madness directory
python -m venv venv
venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium numpy keyboard
```

#### 4. Verify Shared Memory

1. Launch Geometry Dash
2. Enable geode mod (mod loader icon → utridu → enable)
3. Load "Stereo Madness" level
4. Press 'M' to toggle HUD - should display "RL: Memory Ready" (should be activated else it will be shown directly)

#### 5. Start Training

```bash
# From Stereo_Madness directory
python main.py
```

Monitor training loop output:
```
Ep 1    | Win% 5.1%   | % 5.1  | Reward -45.23  | Loss 23.984
Ep 2    | Win% 10.2%  | % 8.3  | Reward -8.12   | Loss 17.768
...
[Curriculum] Promoted to Slice 2: Pre-Triple Spike
```

---

## Results & Performance

### Benchmark Metrics

| Slice | Physics | Episodes to 70% | Peak Success Rate | 
|-------|---------|-----------------|-------------------|
| 1 | Cube | 1100 | 70% |
| 2 | Cube | 1300 | 70% |
| 3 | Cube | 2000 | 70% |
| 4 | Ship | 1000 | 70% |
| 5 | Cube | 3000 | 70% |
| 6 | Cube | - | 70% |
| 7 | Ship | - | 70% |
| 8 | Cube | 2000 | 70% |

**Total Training Time**: ~18-24 hours on cpu

### Key Observations

1. **Mode-Switch Overhead**: Ship slices (4) require ~30% less episodes than same-mode progression due to physics simplicity 
2. **Precision-Dependent Difficulty**: Slice 6 (Triple Spike) hardest cube section; requires precise jump timing
3. **Late-Game Scaling**: Episode step count increases linearly with level completion percentage
4. **Sample Efficiency**: ~ 6M total transitions to reach >70% on final slice

---

## Limitations & Future Work

### Current Limitations

#### 1. **Inference Latency**

**Issue**: Real-time inference adds ~15ms per frame
- Game runs at ~60 FPS (16.67ms/frame)
- Network forward pass: ~3-5ms
- Memory I/O + synchronization: ~8-12ms
- **Net Effect**: Occasional frame drops during decision-making
- *Result* ==> (Will work fine for the easy-mid-hard levels but for levels of diff > insane it will cause problems (impossible for the case of demons lvls))
  
**Mitigation**: Frame stacking (action repeats) and frame skipping reduce effective required latency.

**Future**: Quantization (INT8) and model pruning could reduce inference to <1ms.

#### 2. **Generalization**

**Issue**: Agent trains on Stereo Madness only; architecture not tested on other levels 
- Level-specific reward shaping may not transfer
- Object detection calibrated for Stereo Madness patterns
- Curriculum design requires manual slice definition per level

**Mitigation**: Object type classification is generic (spike, block, portal); reward shaping uses physics-agnostic signals.

**Future**: Meta-learning framework to automatically discover curriculum structure.

#### 3. **Mode-Switch Robustness**

**Issue**: Physics reset between Cube and Ship modes causes performance dips
- Ship mode requires different precision tuning than Cube
- Gravity inversion not explicitly modeled in state
- Landing mechanics fundamentally different

**Future**: Mode-aware neural architecture (separate heads with shared trunk).

#### 4. **Stochasticity Limitations**

**Issue**: Game engine is deterministic; no environmental noise
- Agent may overfit to precise state-action sequences 
- Real-world perturbations (lag spikes, input delays) not modeled
- No resilience to adversarial conditions

**Future**: Domain randomization (modify object positions, gravity, speeds) to increase robustness and expand the observation for the full game objects/mods

#### 5. **Reward Shaping Brittleness**

**Issue**: Hand-tuned expert reward functions require per-mode calibration (test and observe Agent behavior ==> adjust)
- Coefficients (progress_scale, jump_penalty) not learned automatically
- Suboptimal hyperparameters lead to poor exploration/exploitation trade-off 

**Future**: Inverse RL to learn reward function from human demonstrations. (or GAIL)

### Proposed Extensions

#### 1. **Multi-Level Training**

- Extend framework to other Geometry Dash levels 
- Implement automatic curriculum discovery via learning progress metrics
- Transfer learning across levels with similar physics patterns (back on track/Polargeist)

#### 2. **Imitation Learning Bootstrapping**

- Record human playthroughs 
- Pre-train agent via behavioral cloning
- Fine-tune via RL to exceed human performance

#### 3. **Model Ensemble & Uncertainty**

- Train multiple agents with different seeds
- Use ensemble disagreement for exploration confidence
- Reduces reliance on ε-greedy heuristic

#### 4. **Real-Time Adaptation**

- Detect mid-episode level modifications (e.g., moving objects)
- Adapt policy online using meta-RL (e.g., MAML)
- Enable transfer to user-created levels

#### 5. **Hardware Acceleration**

- Implement inference on GPU via TorchScript/ONNX
- Use CUDA streams for async memory I/O
- Reduce latency to <1ms for smoother gameplay

---

## File Structure Reference

```
Stereo_Madness/
├── main.py                         # Training orchestrator
├── config.py                       # Hyperparameter centralization
├── agents/
│   ├── ddqn.py                    # Dueling DQN networks & training
│   ├── replay_buffer.py           # Experience memory
│   ├── expert_cube.py             # Cube mode reward shaping
│   └── expert_ship.py             # Ship mode reward shaping
├── core/
│   ├── environment.py             # Gymnasium wrapper
│   ├── memory_bridge.py           # Shared memory synchronization
│   └── state_utils.py             # Feature normalization
├── curriculum/
│   ├── manager.py                 # Progression & slice management
│   └── slice_definitions.json     # Level decomposition
├── analytics/
│   ├── dashboard.py               # Training visualization
│   ├── plot_heatmap.py           # Performance heatmaps
│   └── plot_death_map.py         # Failure point analysis
├── checkpoints/
│   ├── slice_XX_current.pth      # Active training checkpoint
│   └── final_models/
│       └── slice_XX_model.pth    # Completed expert models
└── logs/
    ├── training_log.csv          # Episode metrics
    ├── death_log.csv             # Failure analysis
    └── training_meta.json        # Curriculum progress
```

---

## Contributing & Contact

For questions, issues, or extensions, consult the source code documentation in individual modules or contact the development team.

**Key References**:
-Playing Geometry Dash with Convolutional Neural Networks, Stanford University 
- Van Hasselt et al. (2015): "Deep Reinforcement Learning with Double Q-learning"
- Wang et al. (2016): "Dueling Network Architectures for Deep Reinforcement Learning"
- Curriculum Learning: Bengio et al. (2009), Graves et al. (2017)


