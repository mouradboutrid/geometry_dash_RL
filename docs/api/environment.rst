API Reference - Environment
===========================

Documentation for environment and state utilities.

.. module:: core

.. contents::
   :depth: 2
   :local:

Core Environment
================

GeometryDashEnv (Gymnasium)
---------------------------

.. automodule:: core.environment
   :members:
   :undoc-members:

.. py:class:: GeometryDashEnv(gym.Env)

   Gymnasium-compliant environment wrapper for Geometry Dash.

   Interfaces with the Geode mod via shared memory to provide:

   - **State Observations**: 154-dimensional normalized vectors
   - **Action Application**: Discrete action space (Release/Hold)
   - **Reward Calculation**: Dense, mode-specific reward signals
   - **Episode Management**: Reset and step functionality

   :param int frame_skip: Number of frames to repeat actions (default 4)
   :param int frame_stack: Number of frames to stack (default 2)

   .. py:attribute:: action_space

      Discrete(2): {0: Release, 1: Hold}

   .. py:attribute:: observation_space

      Box(shape=(308,)): Stacked observations (154 * 2)

   .. py:method:: reset() -> tuple

      Reset level and return initial observation.

      :return: Tuple of (observation, info)
      :rtype: tuple[ndarray, dict]

   .. py:method:: step(action, reward_context=None) -> tuple

      Execute action for ``frame_skip`` frames and return experience tuple.

      :param int action: Action to take (0=Release, 1=Hold)
      :param dict reward_context: Additional reward context (optional)
      :return: Tuple of (obs, reward, terminated, truncated, info)
      :rtype: tuple

      **Return Values:**

      - **obs**: Stacked observation array of shape (308,)
      - **reward**: Dense reward signal (float)
      - **terminated**: True if episode ended (death/completion)
      - **truncated**: Always False (no time limit in Geometry Dash)
      - **info**: Dict with 'percent' key (level completion percentage)

   .. py:method:: set_slice(slice_data)

      Set current curriculum slice (for reward context).

      :param dict slice_data: Slice definition from curriculum

   .. py:method:: _calculate_reward(state, action, is_dead, reward_context) -> float

      Compute reward using expert modules.

      :param SharedState state: Current game state
      :param int action: Action taken
      :param bool is_dead: Death flag
      :param dict reward_context: Reward configuration
      :return: Reward value
      :rtype: float

State Management
================

State Utilities
---------------

Utility functions for state processing and tensor conversion:

.. automodule:: core.state_utils
   :members:
   :undoc-members:

.. py:function:: normalize_state(state) -> ndarray

   Convert raw shared memory state to normalized observation vector.

   :param SharedState state: Raw state from C++ Geode mod
   :return: Normalized observation of shape (154,)
   :rtype: ndarray

   **Normalization Operations:**

   - **Velocity**: Divided by expected max (30 units/frame)
   - **Position**: Divided by playable area dimensions (900 pixels)
   - **Objects**: Computed relative to player and scaled by typical sizes
   - **Stability**: NaN/Inf values replaced with safe clamped values

   Example:

   .. code-block:: python

      from core.environment import GeometryDashEnv
      from core.state_utils import normalize_state

      env = GeometryDashEnv()
      obs, _ = env.reset()
      print(obs.shape)  # (308,)
      print(obs.dtype)  # float32

.. py:function:: to_tensor(obs) -> Tensor

   Convert numpy observation to PyTorch tensor on the configured device.

   :param ndarray obs: Observation array from environment
   :return: Tensor on device (CUDA or CPU)
   :rtype: torch.Tensor

Inter-Process Communication
============================

MemoryBridge (IPC)
------------------

Windows named memory interface for real-time IPC with Geode mod:

.. automodule:: core.memory_bridge
   :members:
   :undoc-members:

.. py:class:: MemoryBridge

   Windows named memory mapping interface for IPC with Geode mod.

   Provides thread-safe synchronization via spinlock flags:

   - **cpp_writing**: Lock flag set by C++ when writing state
   - **py_writing**: Lock flag set by Python when reading state

   .. py:method:: __init__()

      Connect to shared memory created by Geode mod.

      Raises RuntimeError if shared memory not found (Geometry Dash not running).

   .. py:method:: read_state() -> SharedState

      Read current game state from shared memory with synchronization.

      Implements busy-wait spinlock to ensure C++ finishes writing before reading.

      :return: Current shared memory state
      :rtype: SharedState

   .. py:method:: write_action(action)

      Write action command to shared memory.

      :param int action: Action code (0 or 1)

   .. py:method:: send_reset()

      Signal game to reset level.

      Waits 50ms for C++ to process reset command.

   .. py:method:: close()

      Close shared memory handle (cleanup).

.. py:class:: SharedState(ctypes.Structure)

   Struct matching the C++ SharedState definition in the Geode mod.

   **Synchronization Fields:**

   - **cpp_writing** (c_int): Lock flag set by C++ when writing
   - **py_writing** (c_int): Lock flag set by Python when reading

   **Player State:**

   - **player_x** (c_float): Player X position
   - **player_y** (c_float): Player Y position
   - **player_vel_x** (c_float): Computed velocity in X direction
   - **player_vel_y** (c_float): Velocity in Y direction from game
   - **player_rot** (c_float): Current rotation angle
   - **gravity** (c_int): Gravity direction (+1 or -1)
   - **is_on_ground** (c_int): Ground contact flag (0 or 1)
   - **player_mode** (c_int): Game mode (0=Cube, 1=Ship)
   - **player_speed** (c_float): Internal speed setting

   **Level State:**

   - **is_dead** (c_int): Death flag (collision detected)
   - **is_terminal** (c_int): Terminal state flag
   - **percent** (c_float): Level completion percentage (0-100)
   - **dist_nearest_hazard** (c_float): Distance to nearest spike obstacle
   - **dist_nearest_solid** (c_float): Distance to nearest solid block

   **Interaction Fields:**

   - **objects** (ObjectData * 30): Array of nearby object descriptors
   - **action_command** (c_int): Desired action (0=Release, 1=Hold)
   - **reset_command** (c_int): Reset level flag
   - **checkpoint_command** (c_int): Checkpoint activation flag

.. py:class:: ObjectData(ctypes.Structure)

   Individual object descriptor from the nearby objects array.

   **Fields:**

   - **dx** (c_float): Relative X distance from player
   - **dy** (c_float): Relative Y distance from player
   - **w** (c_float): Object bounding box width
   - **h** (c_float): Object bounding box height
   - **type** (c_int): Object type code (1=Spike, 2=Block, 5=Portal, -1=Empty/Inactive)

System Architecture
-------------------

Data flow from Geometry Dash to agent and back:

.. code-block:: text

   Raw C++ Game State (via shared memory)
         ↓
   MemoryBridge.read_state()
   [spinlock synchronization]
         ↓
   normalize_state()
   [scaling & NaN handling]
         ↓
   to_tensor()
   [numpy array → PyTorch tensor]
         ↓
   Agent Network.forward()
   [policy inference]
         ↓
   MemoryBridge.write_action()
   [spinlock synchronization]
         ↓
   Geode Mod Applies Action

Integration Example
-------------------

Complete example of environment integration with agent:

.. code-block:: python

   import torch
   from core.environment import GeometryDashEnv
   from agents.ddqn import Agent
   from config import *

   # Initialize environment
   env = GeometryDashEnv()
   env.frame_skip = 4
   env.frame_stack = 2

   # Initialize agent
   agent = Agent(INPUT_DIM, OUTPUT_DIM, config={
       'lr': 0.0003,
       'gamma': 0.99,
       'device': DEVICE,
   }, checkpoint_dir='checkpoints/')

   # Training loop
   obs, _ = env.reset()
   for step in range(1000):
       # Select and apply action
       action = agent.select_action(obs, is_training=True)
       next_obs, reward, done, _, info = env.step(action)
       
       # Log progress
       print(f"Step {step}: Level={info['percent']:.1f}%")

       # Reset or continue
       if done:
           obs, _ = env.reset()
       else:
           obs = next_obs

Troubleshooting
===============

MemoryBridge Connection Error
------------------------------

If initialization fails with:

.. code-block:: text

   RuntimeError: [Critical] Could not find Shared Memory 'GD_RL_Memory'.
   Make sure Geometry Dash is running with the Mod installed.

**Solutions:**

1. Verify Geometry Dash is running and not paused
2. Confirm Geode mod is installed and enabled
3. Load a level and wait for HUD to display "RL: Memory Ready"
4. Run Geometry Dash as Administrator
5. Restart Geometry Dash and retry connection

Invalid Observations (NaN Values)
----------------------------------

If normalized observations contain NaN or Inf values:

**Causes:**

- Invalid state values from shared memory
- Division by zero in normalization
- Uninitialized object positions

**Solutions:**

1. Verify reward function has no division by zero
2. Check object position clipping (ensure valid ranges)
3. Review ``np.nan_to_num`` clamping strategy
4. Add assertions to catch invalid states early

Observation Dimension Mismatch
-------------------------------

Ensure ``frame_skip`` and ``frame_stack`` settings produce expected dimensions:

.. code-block:: python

   expected_dim = INPUT_DIM * frame_stack  # 154 * 2 = 308
   obs, _ = env.reset()
   assert obs.shape[0] == expected_dim, f"Got {obs.shape[0]}, expected {expected_dim}"

See :doc:`../performance` for benchmark latencies.
