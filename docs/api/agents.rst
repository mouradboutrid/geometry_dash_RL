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
   :param dict config: Configuration dict containing hyperparameters
   :param str checkpoint_dir: Directory for saving/loading models

   .. py:method:: select_action(state, is_training=True)

      Select action using epsilon-greedy exploration.

      :param ndarray state: Observation vector (154,)
      :param bool is_training: If True, apply epsilon-greedy; else greedy
      :return: Action index (0=Release, 1=Hold)
      :rtype: int

   .. py:method:: learn(memory)

      Perform one learning step on batch from replay buffer.

      :param ReplayBuffer memory: Experience replay buffer
      :return: MSE loss value
      :rtype: float

   .. py:method:: save(filename="best_model.pth")
   .. py:method:: load(path)
   .. py:method:: reset_network()

DuelingDQN Architecture
^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: DuelingDQN(nn.Module)

   Neural network with dueling architecture (value + advantage streams).

   **Architecture:**
   
   - **Input**: 154 features
   - **FC1**: 154 → 256, ReLU
   - **FC2**: 256 → 256, ReLU
   - **Value stream**: 256 → 128 → 1
   - **Advantage stream**: 256 → 128 → 2
   - **Output**: $Q(s,a) = V(s) + A(s,a) - \text{mean}(A(s,a))$

ReplayBuffer
------------

.. automodule:: agents.replay_buffer
   :members:
   :undoc-members:

.. py:class:: ReplayBuffer(capacity)

   Fixed-size circular buffer for experience replay.

   :param int capacity: Max transitions (default 50000)

   .. py:method:: push(state, action, reward, next_state, done)
   .. py:method:: sample(batch_size)
   .. py:method:: __len__()

Expert Modules
--------------

.. automodule:: agents.expert_cube
   :members:
   :undoc-members:

.. py:class:: CubeExpert

   Static utility class for cube mode reward shaping.

   .. py:staticmethod:: get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None)

      **Reward components:**
      
      - **Progress**: $\Delta\% \times 20.0$
      - **Step penalty**: -0.0001
      - **Jump penalty**: -0.0005
      - **Clearance bonus**: +0.01
      - **Landing bonus**: +20

.. automodule:: agents.expert_ship
   :members:
   :undoc-members:

.. py:class:: ShipExpert

   SOTA-aligned reward shaping for Ship mode (Stereo Madness). 
   Focuses on survival, forward progress, and vertical stability around **Y=235**.

   .. py:staticmethod:: get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None)

      Compute reward for ship mode with DDQN-safe clamping.

      :param SharedState state: Current game state
      :param int action: Action taken (0=Release, 1=Thrust)
      :param float prev_percent: Previous progress percentage
      :param float prev_dist_nearest_hazard: Previous hazard distance (optional)
      :param dict reward_context: Context containing ``prev_action`` and scales
      :return: Scalar reward clamped between [-10.0, 10.0]
      :rtype: float

      **Reward components:**
      
      - **Progress (Main)**: $\Delta\% \times 20.0$
      - **Stability**: Bonus up to +0.01 based on proximity to $Y=235$
      - **Thrust Penalty**: -0.0003 per thrust action
      - **Spam Penalty**: -0.0005 if action is a repeated thrust
      - **Clearance**: +0.005 for moving away from hazards within 30 units
      - **Death Penalty**: -5.0
      - **Step Penalty**: -0.0001

   .. py:staticmethod:: should_reset_weights(prev_mode)

      Returns True if switching from Cube (0) to Ship (1).

Configuration
-------------

.. code-block:: python

   # Algorithm
   GAMMA = 0.99              # Discount factor
   BATCH_SIZE = 64           # Replay batch size
   LR = 0.0003               # Learning rate
   MEMORY_SIZE = 50000       # Replay buffer capacity
   TARGET_UPDATE = 1000      # Target network sync frequency

   # I/O
   INPUT_DIM = 154           # State dimension
   OUTPUT_DIM = 2            # Action dimension
   DEVICE = "CPU"           # torch device
