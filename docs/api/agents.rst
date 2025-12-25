API Reference - Agents
======================

Complete API documentation for agent modules and learning components.

.. module:: agents

.. contents::
   :depth: 2
   :local:

Core Agent
==========

Agent (DDQN)
------------

The ``Agent`` class implements Dueling Double Q-Learning with the following key methods:

.. py:class:: Agent(input_dim, output_dim, config, checkpoint_dir)

   Deep Q-Network agent with dueling architecture and Double DQN learning.

   :param int input_dim: Input dimension (154 for Geometry Dash)
   :param int output_dim: Output dimension (2 for jump/release)
   :param dict config: Configuration dict containing hyperparameters
   :param str checkpoint_dir: Directory for saving/loading models

   .. py:method:: select_action(state, is_training=True)

      Select action using epsilon-greedy exploration during training.

      :param ndarray state: Observation vector of shape (154,)
      :param bool is_training: If True, apply epsilon-greedy; otherwise greedy policy
      :return: Action index (0=Release, 1=Hold)
      :rtype: int

   .. py:method:: learn(memory)

      Perform one learning step on batch from replay buffer.

      :param ReplayBuffer memory: Experience replay buffer
      :return: MSE loss value
      :rtype: float

   .. py:method:: save(filename="best_model.pth")

      Save trained model to checkpoint file.

   .. py:method:: load(path)

      Load model weights from checkpoint file.

   .. py:method:: reset_network()

      Reset all network weights.


DuelingDQN Architecture
-----------------------

Neural network implementation of the Dueling Deep Q-Network architecture:

.. py:class:: DuelingDQN(nn.Module)

   Neural network with dueling architecture (value + advantage streams).

   **Network Layers:**

   - **Input**: 154 features
   - **FC1**: 154 → 256 neurons (ReLU activation)
   - **FC2**: 256 → 256 neurons (ReLU activation)
   - **Value Stream**: 256 → 128 → 1
   - **Advantage Stream**: 256 → 128 → 2
   - **Output**: :math:`Q(s,a) = V(s) + A(s,a) - \text{mean}(A(s,a))`

   This architecture separately learns state values and action advantages, improving learning stability.

Memory Management
==================

ReplayBuffer
------------

Fixed-size circular buffer for storing and sampling experience transitions:

.. py:class:: ReplayBuffer(capacity)

   Efficient replay buffer for experience replay training.

   :param int capacity: Maximum number of transitions to store (default: 50,000)

   .. py:method:: push(state, action, reward, next_state, done)

      Add experience tuple to the replay buffer.

      :param ndarray state: State representation
      :param int action: Action taken
      :param float reward: Reward received
      :param ndarray next_state: Next state reached
      :param bool done: Episode termination flag

   .. py:method:: sample(batch_size)

      Sample a random batch of experiences from the buffer.

      :param int batch_size: Number of experiences to sample
      :return: Batch of (state, action, reward, next_state, done) tuples
      :rtype: tuple

   .. py:method:: __len__()

      Get the current number of transitions in the buffer.

      :return: Current buffer size
      :rtype: int

Expert Reward Shaping
=====================

Mode-specific reward shaping modules for different gameplay types:

CubeExpert
----------

Static utility class for cube mode reward shaping and calculation:

.. py:class:: CubeExpert

   Reward shaper for platforming and cube mode sequences.

   .. py:staticmethod:: get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None)

      Compute shaped reward for cube mode platforming.

      **Reward Components:**

      - **Progress**: :math:`\Delta\% \times 20.0` — Primary incentive for forward movement
      - **Step Penalty**: :math:`-0.0001` — Small cost per step
      - **Jump Penalty**: :math:`-0.0005` — Discourages unnecessary jumping
      - **Clearance Bonus**: :math:`+0.01` — Reward for maintaining distance from obstacles
      - **Landing Bonus**: :math:`+20` — Large reward for successful landings


ShipExpert
----------

State-of-the-art reward shaping for ship mode (Stereo Madness), optimized for survival and stability:

.. py:class:: ShipExpert

   Reward shaper for ship mode with focus on vertical stability around :math:`Y=235`.

   .. py:staticmethod:: get_reward(state, action, prev_percent, prev_dist_nearest_hazard=None, reward_context=None)

      Compute shaped reward for ship mode with safe clamping.

      :param SharedState state: Current game state
      :param int action: Action taken (0=Release, 1=Thrust)
      :param float prev_percent: Previous progress percentage
      :param float prev_dist_nearest_hazard: Previous distance to nearest hazard (optional)
      :param dict reward_context: Context dict with ``prev_action`` and reward scales
      :return: Scalar reward clamped to [-10.0, 10.0]
      :rtype: float

      **Reward Components:**

      - **Progress (Main)**: :math:`\Delta\% \times 20.0` — Primary learning signal
      - **Stability**: Up to :math:`+0.01` bonus based on proximity to :math:`Y=235`
      - **Thrust Penalty**: :math:`-0.0003` per thrust action applied
      - **Spam Penalty**: :math:`-0.0005` if repeating the previous thrust action
      - **Clearance Bonus**: :math:`+0.005` for moving away from hazards within 30 units
      - **Death Penalty**: :math:`-5.0` for collision/failure
      - **Step Penalty**: :math:`-0.0001` per timestep

   .. py:staticmethod:: should_reset_weights(prev_mode)

      Determine if network weights should be reset during mode switching.

      :param int prev_mode: Previous mode (0=Cube, 1=Ship)
      :return: True if switching from Cube to Ship
      :rtype: bool

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
