API Reference - Environment
===========================

Documentation for environment and state utilities.

.. module:: core

GeometryDashEnv (Gymnasium)
---------------------------

.. automodule:: core.environment
   :members:
   :undoc-members:

.. py:class:: GeometryDashEnv(gym.Env)

   Gymnasium environment wrapper for Geometry Dash.

   Interfaces with the Geode mod via shared memory to provide:
   - State observations (154-dimensional normalized vectors)
   - Action application (discrete: Release/Hold)
   - Reward calculation (dense, mode-specific)
   - Episode management (reset/step)

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

      Execute action for frame_skip frames and return experience.

      :param int action: Action to take (0 or 1)
      :param dict reward_context: Additional reward context (optional)
      :return: Tuple of (obs, reward, terminated, truncated, info)
      :rtype: tuple

      Returns:
      - obs: Stacked observation (308,)
      - reward: Dense reward signal (float)
      - terminated: True if episode ended
      - truncated: Always False (Geometry Dash has no time limit)
      - info: Dict with 'percent' key (level completion %)

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

State Utilities
---------------

.. automodule:: core.state_utils
   :members:
   :undoc-members:

.. py:function:: normalize_state(state) -> ndarray

   Convert raw shared memory state to normalized observation.

   :param SharedState state: Raw state from C++ mod
   :return: Normalized 154-dimensional vector
   :rtype: ndarray

   Normalization:
   - Velocity divided by expected max (30)
   - Position divided by playable area (900)
   - Objects relative to player and scaled by typical sizes
   - NaN/Inf replaced with clamped values

   Example:

   .. code-block:: python

      from core.environment import GeometryDashEnv
      from core.state_utils import normalize_state

      env = GeometryDashEnv()
      obs, _ = env.reset()
      print(obs.shape)  # (308,)
      print(obs.dtype)  # float32

.. py:function:: to_tensor(obs) -> Tensor

   Convert numpy observation to PyTorch tensor on configured device.

   :param ndarray obs: Observation array
   :return: Tensor on device (cuda or cpu)
   :rtype: torch.Tensor

MemoryBridge (IPC)
------------------

.. automodule:: core.memory_bridge
   :members:
   :undoc-members:

.. py:class:: MemoryBridge

   Windows named memory mapping interface for IPC with Geode mod.

   Provides synchronization via spinlock flags:
   - cpp_writing: Set by C++ when writing state
   - py_writing: Set by Python when reading state

   .. py:method:: __init__()

      Connect to shared memory created by Geode mod.

      Raises RuntimeError if shared memory not found (Geometry Dash not running).

   .. py:method:: read_state() -> SharedState

      Read current game state from shared memory.

      Implements spinlock to wait for C++ to finish writing.

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

   Struct matching C++ SharedState definition.

   Fields:

   - cpp_writing (c_int): Lock flag set by C++
   - py_writing (c_int): Lock flag set by Python
   - player_x (c_float): Player X position
   - player_y (c_float): Player Y position
   - player_vel_x (c_float): Computed velocity X
   - player_vel_y (c_float): Velocity Y from game
   - player_rot (c_float): Rotation angle
   - gravity (c_int): +1 or -1
   - is_on_ground (c_int): Ground contact flag
   - is_dead (c_int): Death flag
   - is_terminal (c_int): Terminal state flag
   - percent (c_float): Level completion (0-100)
   - dist_nearest_hazard (c_float): Distance to spike
   - dist_nearest_solid (c_float): Distance to block
   - player_mode (c_int): 0=Cube, 1=Ship
   - player_speed (c_float): Internal speed setting
   - objects (ObjectData * 30): Nearby objects array
   - action_command (c_int): Desired action (0=Release, 1=Hold)
   - reset_command (c_int): Reset level flag
   - checkpoint_command (c_int): Checkpoint flag

.. py:class:: ObjectData(ctypes.Structure)

   Individual object in nearby objects array.

   Fields:

   - dx (c_float): Relative X distance (to player)
   - dy (c_float): Relative Y distance
   - w (c_float): Object width
   - h (c_float): Object height
   - type (c_int): Object type (1=spike, 2=block, 5=portal, -1=empty)

Architecture
^^^^^^^^^^^^

.. code-block:: text

   Raw C++ State (via mmap)
         ↓
   MemoryBridge.read_state() (spinlock)
         ↓
   normalize_state() (scaling, NaN handling)
         ↓
   to_tensor() (numpy → torch)
         ↓
   Network.forward() (decision)
         ↓
   MemoryBridge.write_action() (spinlock)
         ↓
   Geode mod applies action

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from core.environment import GeometryDashEnv
   from agents.ddqn import Agent
   from config import *

   # Create environment
   env = GeometryDashEnv()
   env.frame_skip = 4
   env.frame_stack = 2

   # Create agent
   agent = Agent(INPUT_DIM, OUTPUT_DIM, config={
       'lr': 0.0003,
       'gamma': 0.99,
       'device': DEVICE,
   }, checkpoint_dir='checkpoints/')

   # Collect experience
   obs, _ = env.reset()
   for step in range(1000):
       action = agent.select_action(obs, is_training=True)
       next_obs, reward, done, _, info = env.step(action)
       print(f"Step {step}: Percent={info['percent']:.1f}%")

       if done:
           obs, _ = env.reset()
       else:
           obs = next_obs

Error Handling
--------------

**MemoryBridge Connection Error**

If you see:

.. code-block:: text

   RuntimeError: [Critical] Could not find Shared Memory 'GD_RL_Memory'.
   Make sure Geometry Dash is running with the Mod installed.

Solutions:

1. Verify Geometry Dash is running
2. Check utridu mod is enabled in Geode mod menu
3. Load a level (HUD must display "RL: Memory Ready")
4. Run Geometry Dash as Administrator

**NaN in Observations**

If normalization produces NaN:

1. Check reward function (avoid division by zero)
2. Clamp object positions (ensure valid ranges)
3. Use np.nan_to_num (currently in place)

**Frame Skip/Stack Mismatch**

Ensure observation dimension matches:

.. code-block:: python

   expected_dim = INPUT_DIM * frame_stack
   obs, _ = env.reset()
   assert obs.shape[0] == expected_dim

See :doc:`../performance` for benchmark latencies.
