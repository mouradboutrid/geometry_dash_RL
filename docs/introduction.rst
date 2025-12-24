Introduction
=============

Overview
--------

This documentation covers the Geometry Dash Reinforcement Learning Agent—a complete system for training an AI to autonomously master the "Stereo Madness" level in Geometry Dash v2.2074.

The project demonstrates modern deep reinforcement learning techniques applied to real-time action games, with particular emphasis on:

- **Sample efficiency** through Double DQN and experience replay
- **Curriculum design** for long-horizon sequential decision-making
- **Cross-language interoperability** for game engine integration
- **Mode-aware learning** to handle physics transitions

Problem Statement
-----------------

Geometry Dash is a notoriously difficult rhythm-platformer where players must time jumps to avoid obstacles with pixel-perfect precision. Training an RL agent to master this game is challenging due to:

1. **Sparse Rewards**: Only the final frame (100% completion) yields reward; intermediate progress provides no learning signal
2. **Long Horizons**: Stereo Madness requires 200-300 correct sequential actions
3. **Non-Stationary Physics**: Cube and Ship modes operate under different gravity and control mechanics
4. **Real-Time Constraints**: Decisions must be made within 16.67 ms per frame
5. **High Variance**: Stochastic exploration in such a precise domain easily leads to failure

Traditional Deep RL Methods Struggle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **DQN**: Suffers from overestimation bias, leading to overconfident policies
- **A3C**: Requires distributed workers; inefficient for single-agent, wall-clock-constrained scenarios
- **PPO**: Designed for continuous actions; discrete action spaces benefit from Q-learning
- **Simple Q-Learning**: State space (2^154 possibilities) is too large for tabular methods

Our Solution
^^^^^^^^^^^^

We employ a hierarchical approach combining algorithmic and curriculum innovations:

.. mermaid::

   graph LR
       A[Sparse Rewards] -->|Curriculum Learning| B[Dense Rewards per Slice]
       C[Long Horizons] -->|Slice Decomposition| D[Short Horizons per Task]
       E[Mode Transitions] -->|Policy Transfer| F[Mode-Aware Specialization]
       G[Overestimation Bias] -->|Double DQN| H[Stable Q-Learning]
       I[Exploration Complexity] -->|Epsilon Greedy| J[Simple but Effective]

Architecture Highlights
-----------------------

**Python Training Pipeline**

- PyTorch-based neural networks with dynamic computation graphs
- Gymnasium integration for standardized environment interface
- Modular reward shaping for domain-specific optimization
- Persistent checkpoint management for fault tolerance

**C++ Game Integration**

- Geode modding framework for Geometry Dash
- Named pipe synchronization for low-latency I/O
- Spinlock mutual exclusion for lock-free memory access
- Real-time object detection and state collection

**Curriculum System**

- 8-slice decomposition of Stereo Madness with automatic slice detection
- Advancement criteria based on rolling success metrics
- Expert model caching for relay race checkpoint seeding
- Mode-aware policy initialization and transfer learning

Key Innovations
---------------

1. **Decoupled Action Selection** (Double DQN)
   
   Standard DQN uses the same network to select and evaluate actions, leading to overestimation. Our approach:
   
   .. code-block:: python
   
      # Use online network to select best action
      best_actions = online_net(next_state).argmax(1)
      # Evaluate with target network for unbiased estimate
      next_q = target_net(next_state).gather(1, best_actions)

2. **Value-Advantage Decomposition** (Dueling Architecture)
   
   Rather than directly computing Q(s,a), we separately compute:
   - V(s): How good is this state?
   - A(s,a): How much better is this action than average?
   
   Then recombine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
   
   This improves generalization when many actions have similar Q-values.

3. **Expertise Specialization**
   
   Each curriculum slice maintains a specialized expert policy optimized for its local dynamics. When advancing, we transfer weights from the most recent expert with identical physics mode:
   
   .. code-block:: python
   
      if new_slice.mode == old_slice.mode:
          new_agent.load_weights(old_expert)
          new_agent.epsilon = 0.5  # Resume exploration
      else:
          new_agent.reset()  # Fresh policy for new physics
          new_agent.epsilon = 1.0

4. **Relay Race Navigation**
   
   Before training a new slice, use ensemble of previous experts to "fast-travel" to the slice's starting point, avoiding training on already-mastered sections.

Expected Outcomes
-----------------

Upon successful training:

- **Slice 1-3**: Agent masters cube platforming (0-31% completion)
- **Slice 4**: Agent learns ship stability mechanics (30-47%)
- **Slice 5-6**: Agent refines intermediate cube timing (47-68%)
- **Slice 7**: Agent masters second ship section (68-80%)
- **Slice 8**: Agent completes final cube sprint (80-100%)

Overall success rate: **70-75%** across random initializations and stochastic training seeds.

Document Organization
---------------------

This documentation is structured as follows:

1. **Getting Started**: Installation, environment setup, quick examples
2. **Core Concepts**: Architecture overview, algorithm deep-dive, curriculum strategy
3. **Implementation Details**: State design, reward functions, synchronization protocols
4. **Advanced Topics**: Performance analysis, known limitations, future directions
5. **API Reference**: Comprehensive module documentation with code examples
6. **Development**: Contribution guidelines, troubleshooting, changelog

Each section builds upon previous sections but can be read independently. For a quick start, see :doc:`quickstart`. For the complete picture, read sequentially.

Next Steps
----------

1. **New to the project?** → Start with :doc:`installation`
2. **Want to train an agent?** → See :doc:`quickstart`
3. **Interested in the algorithm?** → Read :doc:`algorithm`
4. **Need technical details?** → Check :doc:`architecture`
5. **Developing extensions?** → Review :doc:`contributing`
