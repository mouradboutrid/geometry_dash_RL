API Reference - Agents
======================

Complete API documentation for agent modules.

.. module:: agents

Agent (DDQN)
------------

.. automodule:: agents.ddqn
   :members:
   :undoc-members:
   :show-inheritance:

The ``Agent`` class implements Dueling Double Q-Learning with the following key methods:

.. py:class:: Agent(input_dim, output_dim, config, checkpoint_dir)

   Deep Q-Network agent with dueling architecture and Double DQN learning.

   :param int input_dim: Input dimension (154 for Geometry Dash)
   :param int output_dim: Output dimension (2 for jump/release)
   :param dict config: Configuration dict with keys:
       - device: torch device
       - lr: learning rate (default 0.0003)
       - gamma: discount factor (default 0.99)
       - batch_size: replay batch size (default 64)
       - target_update: steps between target network sync (default 1000)
       - epsilon_start: initial exploration rate (default 1.0)
       - epsilon_end: final exploration rate (default 0.01)
       - epsilon_decay: exploration decay steps (default 50000)
   :param str checkpoint_dir: Directory for saving/loading models

   .. py:method:: select_action(state, is_training=True) -> int

      Select action using epsilon-greedy exploration.

      :param ndarray state: Observation vector (154,)
      :param bool is_training: If True, apply epsilon-greedy; else greedy
      :return: Action index (0=Release, 1=Hold)

   .. py:method:: learn(memory) -> float

      Perform one learning step on batch from replay buffer.

      :param ReplayBuffer memory: Experience replay buffer
      :return: MSE loss value (float)

   .. py:method:: save(filename="best_model.pth")

      Save network weights to checkpoint.

      :param str filename: Checkpoint filename (saved in checkpoint_dir)

   .. py:method:: load(path)

      Load network weights from checkpoint.

      :param str path: Path to checkpoint file

   .. py:method:: reset_network()

      Re-initialize networks from scratch (for mode switches).

DuelingDQN Architecture
^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: DuelingDQN(nn.Module)

   Neural network with dueling architecture (value + advantage streams).

   Architecture:
   - Input: 154 features
   - FC1: 154 → 256, ReLU
   - FC2: 256 → 256, ReLU
   - Value stream: 256 → 128 → 1
   - Advantage stream: 256 → 128 → 2
   - Output: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

ReplayBuffer
------------

.. automodule:: agents.replay_buffer
   :members:
   :undoc-members:

.. py:class:: ReplayBuffer(capacity)

   Fixed-size circular buffer for experience replay.

   :param int capacity: Maximum number of transitions (default 50000)

   .. py:method:: push(state, action, reward, next_state, done)

      Store transition in buffer.

      :param ndarray state: Current state
      :param int action: Action taken
      :param float reward: Reward received
      :param ndarray next_state: Next state
      :param bool done: Episode termination flag

   .. py:method:: sample(batch_size) -> tuple

      Sample random batch from buffer.

      :param int batch_size: Size of batch
      :return: Tuple of (states, actions, rewards, next_states, dones) arrays
      :rtype: tuple

   .. py:method:: __len__() -> int

      Get current buffer size.

      :return: Number of stored transitions
      :rtype: int

Expert Modules
--------------

.. automodule:: agents.expert_cube
   :members:
   :undoc-members:

.. py:class:: CubeExpert

   Static utility class for cube mode reward shaping.

   .. py:staticmethod:: get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None) -> float

      Compute reward for cube mode.

      :param SharedState state: Current game state
      :param int action: Action taken (0 or 1)
      :param float prev_percent: Previous progress percentage
      :param float prev_dist_nearest_hazard: Previous hazard distance (optional)
      :param dict reward_context: Additional context (optional)
      :return: Scalar reward
      :rtype: float

      Reward components:
      - Progress: Δ% × 20.0 (main signal)
      - Step penalty: -0.0001
      - Jump penalty: -0.0005
      - Spam penalty: -0.001
      - Clearance bonus: +0.01
      - Landing bonus: +20

.. automodule:: agents.expert_ship
   :members:
   :undoc-members:

.. py:class:: ShipExpert

   Static utility class for ship mode reward shaping.

   .. py:staticmethod:: get_reward(state, action, prev_percent) -> float

      Compute reward for ship mode.

      :param SharedState state: Current game state
      :param int action: Action taken
      :param float prev_percent: Previous progress percentage
      :return: Scalar reward
      :rtype: float

      Reward components:
      - Progress: Δ% × 10.0
      - Survival: +0.05
      - Centering: Based on distance to y=235
      - Stability: -2000 if |vel_y| > 3.2
      - Precision: +3 if near hazard

Configuration
-------------

Key hyperparameters in ``config.py``:

.. code-block:: python

   # Algorithm
   GAMMA = 0.99              # Discount factor
   BATCH_SIZE = 64           # Replay batch size
   LR = 0.0003               # Learning rate
   MEMORY_SIZE = 50000       # Replay buffer capacity
   TARGET_UPDATE = 1000      # Target network sync frequency

   # Exploration
   EPSILON_START = 1.0       # Initial exploration rate
   EPSILON_END = 0.01        # Final exploration rate
   EPSILON_DECAY = 50000     # Decay schedule parameter

   # I/O
   INPUT_DIM = 154           # State dimension
   OUTPUT_DIM = 2            # Action dimension
   MEM_NAME = "GD_RL_Memory" # Shared memory name
   DEVICE = "cuda"           # torch device

See :doc:`../performance` for tuning guidance.
