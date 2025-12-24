Algorithm & Technical Deep-Dive
===============================

This section provides rigorous analysis of why we chose Dueling Double Q-Learning and how it compares to alternatives.

Problem Formulation
-------------------

**Markov Decision Process (MDP)**

.. math::

   \langle S, A, P, R, \gamma \rangle

Where:

- :math:`S`: State space (154-dimensional feature vectors)
- :math:`A`: Action space (Discrete(2): {Release, Hold})
- :math:`P(s'|s,a)`: Transition dynamics (deterministic; game engine)
- :math:`R(s,a,s')`: Reward function (dense reward shaping)
- :math:`\gamma = 0.99`: Discount factor

**Goal**: Learn policy :math:`\pi^*: S \to A` that maximizes expected discounted return:

.. math::

   G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}

For Geometry Dash, we want to maximize cumulative progress and survival bonuses.

Core Algorithm: Double DQN
---------------------------

Deep Q-Learning estimates action-value function Q(s,a) using a neural network parameterized by weights θ.

**Standard DQN Update** (problematic):

.. math::

   \mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[
     \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2
   \right]

**Problem: Overestimation Bias**

The max operator :math:`\max_{a'} Q(s', a'; \theta^-)` introduces positive bias because:

1. The same network :math:`\theta^-` selects and evaluates the action
2. Errors in Q-estimation are correlated with max selection
3. This bias accumulates, leading to overconfident (suboptimal) policies

Mathematical intuition:

.. code-block:: text

   E[max(X, Y)] ≥ max(E[X], E[Y])  when X and Y are correlated

**Double DQN Solution**:

.. math::

   \mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[
     \left( r + \gamma Q(s', \underbrace{\arg\max_{a'} Q(s', a'; \theta)}_{\text{select via online}}, \theta^-) - Q(s, a; \theta) \right)^2
   \right]

**Key insight**: Use online network :math:`\theta` to **select** best action, but target network :math:`\theta^-` to **evaluate** it.

Implementation in PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Standard DQN (biased)
   online_q = online_net(state).gather(1, action)
   max_next_q = target_net(next_state).max(1)[0].unsqueeze(1)  # BIASED
   target_q = reward + gamma * max_next_q
   
   # Double DQN (unbiased)
   online_q = online_net(state).gather(1, action)
   best_actions = online_net(next_state).argmax(1, keepdim=True)  # Select via online
   max_next_q = target_net(next_state).gather(1, best_actions)   # Evaluate via target
   target_q = reward + gamma * max_next_q
   loss = F.mse_loss(online_q, target_q.detach())

**Empirical Impact**:
- Reduces Q-value overestimation by ~30-50% in discrete action spaces
- Improves sample efficiency by ~10-20%
- Particularly beneficial for precision-dependent tasks (like Geometry Dash)

Dueling Architecture
--------------------

Beyond Double DQN, we use a **Dueling Network** architecture that decomposes Q-values:

.. math::

   Q(s, a) = V(s) + A(s, a) - \text{mean}_{a'} A(s, a')

**Intuition**:

- :math:`V(s)`: State value (how good is this state, regardless of action?)
- :math:`A(s, a)`: Action advantage (how much better is action a than average?)

**Why beneficial?**

1. **Improved Value Estimation**: When multiple actions have similar Q-values, separately modeling V and A stabilizes learning
2. **Generalization**: The value stream learns features useful across all actions (e.g., "am I near an obstacle?")
3. **Faster Convergence**: Approximately 10-20% faster learning on discrete control tasks

**Architecture Diagram**:

.. code-block:: text

   Input (s) → FC(256) → FC(256) → [Value Stream]   → V(s)
                         ↓         [Advantage Stream] → A(s,a)
                         └─────────────────────────→ Q(s,a) = V + A - mean(A)

**PyTorch Implementation**:

.. code-block:: python

   class DuelingDQN(nn.Module):
       def __init__(self, input_dim, output_dim):
           super().__init__()
           self.fc1 = nn.Linear(input_dim, 256)
           self.fc2 = nn.Linear(256, 256)
           
           # Value stream
           self.value_stream = nn.Sequential(
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Linear(128, 1)
           )
           
           # Advantage stream
           self.advantage_stream = nn.Sequential(
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Linear(128, output_dim)
           )
       
       def forward(self, state):
           x = F.relu(self.fc1(state))
           x = F.relu(self.fc2(x))
           
           val = self.value_stream(x)
           adv = self.advantage_stream(x)
           
           # Recombination: subtract mean for normalization
           q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
           return q_vals

**Mean subtraction** ensures identifiability (prevents advantage bias from compensating value bias).

Why Not A3C?
-----------

**A3C (Asynchronous Advantage Actor-Critic)** trains multiple agents in parallel.

**Comparison Table**:

.. list-table:: Algorithm Comparison
   :widths: 30 35 35
   :header-rows: 1

   * - Criterion
     - A3C
     - DDQN
   * - Wall-Clock Time
     - Faster
     - Slower
   * - Memory Footprint
     - High
     - Low
   * - CPU Utilization
     - High
     - Mid
   * - Convergence
     - Stable
     - Stable

**Why DDQN wins for this project**:

1. **Single-Agent, Wall-Clock Constrained**: One CPU/agent is efficient; parallel overhead not justified
2. **Sample Efficiency**: DDQN's off-policy learning and experience replay achieve 60%+ action optimality
3. **Simplicity**: Easier to implement, debug, and analyze
4. **State-Space Alignment**: Discrete actions suit Q-learning better than policy gradients

**A3C would be better for**:
- Multi-worker distributed training
- Continuous action spaces
- Environments with high variance rewards

Why Not PPO?
-----------

**PPO (Proximal Policy Optimization)** uses policy gradients with clipped surrogate loss.

.. list-table:: Algorithm Comparison (PPO vs. DDQN)
   :widths: 30 35 35
   :header-rows: 1

   * - Criterion
     - PPO
     - DDQN
   * - Action Space
     - Continuous/Discrete
     - Discrete
   * - Hyperparameter Tuning
     - High
     - Moderate
   * - Exploration Control
     - Entropy
     - ε-Greedy
   * - Convergence Speed
     - Moderate
     - Fast
**Why DDQN wins**:

1. **Discrete Action Space**: Q-learning naturally suited; PPO's continuous relaxation unnecessary
2. **Low Variance Rewards**: PPO's advantage function estimation more complex with 2 actions
3. **Hyperparameter Sensitivity**: PPO requires tuning clip ratio, entropy coefficient, learning rate schedule
4. **Off-Policy Replay**: DDQN's experience replay provides better sample reuse

**Policy Gradient Mathematics** (for comparison):

.. math::

   \nabla J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla \log \pi_\theta(a|s) A^{\pi}(s,a) \right]

vs. **Q-Learning**:

.. math::

   \nabla J(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \nabla_\theta (r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2 \right]

Q-learning's use of max operator is more suitable for discrete spaces where clear optimal actions exist.

Exploration Strategy: Epsilon-Greedy
-------------------------------------

I use :math:`\epsilon`-greedy exploration: with probability :math:`\epsilon`, select random action; otherwise select greedy action.

**Decay Schedule**:

.. math::

   \epsilon(t) = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) e^{-t / \tau}

Where:

- :math:`\epsilon_{\text{start}} = 1.0`: Fully random at start
- :math:`\epsilon_{\text{end}} = 0.01`: 1% randomness at end
- :math:`\tau = 50000`: Time constant (steps to decay from start to ~37% of range)

**Why epsilon-greedy?**

1. **Simplicity**: Easy to implement and tune
2. **Effectiveness**: Sufficient exploration for discrete spaces
3. **Interpretability**: Clear trade-off between exploration (ε) and exploitation (1-ε)

**Alternatives considered**:

- **Boltzmann Exploration**: :math:`P(a|s) \propto e^{Q(s,a)/T}` - requires temperature tuning
- **Thompson Sampling**: Requires posterior distribution over Q-values - computationally expensive
- **Entropy Bonus** (like PPO): Encourages exploration but reduces sample efficiency

Epsilon-greedy best balances simplicity and effectiveness for discrete actions.

Experience Replay
-----------------

I maintain a circular buffer of transitions: :math:`(s, a, r, s', d) \sim \mathcal{D}`

**Benefits**:

1. **Break Correlation**: Sequential states are highly correlated; uniform sampling decorrelates
2. **Sample Reuse**: Each transition used multiple times (epochs)
3. **Reduce Variance**: Batching reduces gradient variance

**Buffer Implementation**:

.. code-block:: python

   class ReplayBuffer:
       def __init__(self, capacity):
           self.buffer = deque(maxlen=capacity)  # Circular FIFO
       
       def push(self, s, a, r, s_prime, done):
           self.buffer.append((s, a, r, s_prime, done))
       
       def sample(self, batch_size):
           batch = random.sample(self.buffer, batch_size)
           return zip(*batch)  # Unzip into separate arrays

**Key parameters**:

- Capacity: 50,000 transitions (~10 full training runs)
- Batch size: 64 (stable gradient estimation)
- Sampling: Uniform random (could use prioritized replay for improvements)

Target Network Synchronization
-------------------------------

The target network :math:`Q(s, a; \theta^-)` stabilizes learning by reducing moving target problem.

**Update frequency**: Every 1,000 online network updates

.. code-block:: python

   if steps_done % target_update == 0:
       target_net.load_state_dict(online_net.state_dict())

**Why necessary?**

If I used :math:`\theta` for both selection and evaluation, the target would change during optimization, causing instability.

**Trade-off**:

- **Too frequent** (every step): Target becomes moving target; high variance
- **Too infrequent** (every 10,000 steps): Outdated value estimates; slow convergence

1,000 steps found empirically optimal for this domain.

Loss Function & Optimization
-----------------------------

**Mean Squared Error (MSE) Loss**:

.. math::

   \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (Q_{\text{target},i} - Q_{\theta,i})^2

Where:

.. math::

   Q_{\text{target},i} = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-)

**Optimizer**: Adam (lr=0.0003)

- Adaptive learning rates per parameter
- Momentum for faster convergence
- Better than SGD+manual schedule for discrete control

**Why MSE?**

- Differentiable Q-value targets
- Stable gradient flow
- Standard in value-based RL

Temporal Difference (TD) Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prediction error:

.. math::

   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)

In our case:

.. math::

   \delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)

This measures how surprised we are by the reward (if |δ| large, our estimate was wrong).

Convergence Analysis
--------------------

**Convergence Guarantee** (under assumptions):

If:
- Experience replay (decorrelates samples)
- Target network (stabilizes target)
- Sufficient exploration (ε-greedy)

Then: Q-values converge to optimal Q^* in expectation.

**Proof sketch**: See Mnih et al. (2015) or Van Hasselt et al. (2015) for rigorous proofs.

**In practice** (Geometry Dash):

- Slice 1: Convergence in ~8 episodes
- Slice 3 (complex): Convergence in ~45 episodes
- Mode changes (Slice 4): Reset and re-converge in ~38 episodes

Complexity Analysis
-------------------

**Computational Complexity (CPU-Optimized)**:

* **State Collection**: $O(N)$ where $N=30$. Scan-time is negligible on CPU (~0.1ms).
* **Network Forward Pass**: $O(d_{in} \times d_{hid} \times d_{out}) \approx 8 \times 10^4$ FLOPs. 
  Executed in ~0.5–1ms on modern CPU (AVX-accelerated).
* **Gradient Computation**: $O(B \times \text{FLOPs})$ where $B=64$. 
  $\approx 5 \times 10^6$ operations. Executed in ~15–25ms on CPU.

**Space Complexity**:

- Online network: ~130K parameters × 4 bytes/float = 520KB
- Target network: ~130K parameters = 520KB
- Replay buffer: 50K × 154 × 8 bytes ≈ 62MB
- **Total**: ~63.5MB (negligible compared to GPU VRAM)

Practical Tuning Guidelines
---------------------------

**Learning Rate (LR)**

- **Too high** (>0.001): Divergence, loss oscillation
- **Too low** (<0.0001): Slow convergence, stuck in local minima
- **Sweet spot**: 0.0003 ✓

**Batch Size**

- **Too small** (<32): Noisy gradients, high variance
- **Too large** (>256): Slow training, computational waste, risk of local minima
- **Sweet spot**: 64 ✓

**Memory Size**

- **Too small** (<10K): High correlation, poor learning
- **Too large** (>100K): Memory waste, slow sampling
- **Sweet spot**: 50K ✓

**Target Update Frequency**

- **Too frequent** (<500): Moving target, high variance
- **Too infrequent** (>2000): Stale targets, slow convergence
- **Sweet spot**: 1000 ✓

For sensitivity analysis, see :doc:`../performance`.

References
----------

- Van Hasselt et al. (2015): "Deep Reinforcement Learning with Double Q-learning" - AAAI
- Wang et al. (2016): "Dueling Network Architectures for Deep Reinforcement Learning" - ICML
- Playing Geometry Dash with Convolutional Neural Networks, Stanford University

Next: :doc:`../curriculum` for how we handle the multi-phase training problem.
