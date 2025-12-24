API Reference
==============

Complete API documentation for all Geometry Dash RL modules.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   agents
   environment
   curriculum

.. toctree::
   :hidden:

   ../changelog

Module Overview
===============

**agents** - Reinforcement Learning Agent
   Core DDQN agent with dueling architecture, experience replay, and target network synchronization.

   - `Agent` - Main DDQN learner class
   - `DuelingDQN` - Network architecture with value and advantage streams
   - `ReplayBuffer` - Experience memory with uniform sampling
   - `CubeExpert` / `ShipExpert` - Mode-specific reward calculators

**environment** - Gymnasium Environment
   Wrapper around Geometry Dash via Geode mod integration.

   - `GeometryDashEnv` - gym.Env compatible interface
   - `MemoryBridge` - Windows IPC via named pipes
   - `SharedState` / `ObjectData` - Data structures (ctypes)
   - `normalize_state()` - State preprocessing function

**curriculum** - Progressive Training
   Curriculum learning manager with 8-slice decomposition.

   - `CurriculumManager` - Slice progression with automatic advancement
   - Slice definitions (curated checkpoints)
   - Expert caching for relay races

Configuration & Constants
=========================

See `config.py` for hyperparameters:

.. code-block:: python

   # Network
   INPUT_DIM = 154  # State observation dimension
   OUTPUT_DIM = 2   # Actions (Release, Hold)
   HIDDEN_DIM = 256

   # Learning
   LEARNING_RATE = 0.0003
   GAMMA = 0.99  # Discount factor
   TARGET_UPDATE = 1000  # Sync target network every N steps

   # Experience Replay
   MEMORY_SIZE = 50000
   BATCH_SIZE = 64

   # Exploration
   EPSILON_START = 1.0
   EPSILON_END = 0.01
   EPSILON_DECAY = 50000

   # Environment
   FRAME_SKIP = 4
   FRAME_STACK = 2
   DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

   # File Paths
   CHECKPOINT_DIR = 'checkpoints/'
   LOG_DIR = 'logs/'
   CURRICULUM_FILE = 'curriculum/slice_definitions.json'

Quick API Examples
==================

**Training an Agent**

.. code-block:: python

   from agents.ddqn import Agent
   from core.environment import GeometryDashEnv
   from config import *

   env = GeometryDashEnv()
   agent = Agent(INPUT_DIM, OUTPUT_DIM, device=DEVICE)

   for episode in range(1000):
       obs, _ = env.reset()
       done = False
       total_reward = 0

       while not done:
           action = agent.select_action(obs, is_training=True)
           next_obs, reward, done, _, _ = env.step(action)
           total_reward += reward

           agent.remember(obs, action, reward, next_obs, done)
           agent.learn()

           obs = next_obs

       print(f"Episode {episode}: Total Reward = {total_reward:.0f}")

**Loading a Checkpoint**

.. code-block:: python

   from agents.ddqn import Agent
   from config import *

   agent = Agent(INPUT_DIM, OUTPUT_DIM)
   agent.load('checkpoints/slice_01_current.pth')

   # Inference only
   obs, _ = env.reset()
   action = agent.select_action(obs, is_training=False)  # Deterministic
   next_obs, _, done, _, _ = env.step(action)

**Curriculum Progression**

.. code-block:: python

   from curriculum.manager import CurriculumManager
   from agents.ddqn import Agent

   curr = CurriculumManager()
   agent = Agent(INPUT_DIM, OUTPUT_DIM)

   while not curr.is_complete():
       current_slice = curr.get_current_slice()
       # ... training loop ...
       advanced, new_slice = curr.record_episode(success, reward, steps)

       if advanced:
           expert = curr.get_expert_for_relay()
           if expert:
               agent.load(expert)

See :doc:`../quickstart` for more examples.

Function Reference
==================

.. code-block:: python

   # agents/ddqn.py
   Agent(state_dim, action_dim, device='cpu')
   Agent.select_action(obs, is_training=True) -> int
   Agent.learn() -> dict  # Returns loss metrics
   Agent.remember(obs, action, reward, next_obs, done)
   Agent.save(path)
   Agent.load(path)
   Agent.to(device)  # Move to device

   # core/environment.py
   GeometryDashEnv()
   GeometryDashEnv.reset() -> (obs, info)
   GeometryDashEnv.step(action) -> (obs, reward, done, truncated, info)
   GeometryDashEnv.set_slice(slice_def)
   GeometryDashEnv.render()  # HUD visualization

   # core/state_utils.py
   normalize_state(raw_state) -> ndarray
   to_tensor(obs) -> Tensor

   # core/memory_bridge.py
   MemoryBridge()
   MemoryBridge.read_state() -> SharedState
   MemoryBridge.write_action(action)
   MemoryBridge.send_reset()

   # curriculum/manager.py
   CurriculumManager()
   CurriculumManager.get_current_slice() -> dict
   CurriculumManager.record_episode(success, reward, steps) -> (advanced, new_slice)
   CurriculumManager.get_expert_for_relay() -> str or None
   CurriculumManager.is_complete() -> bool
   CurriculumManager.get_progress() -> float
   CurriculumManager.save()
   CurriculumManager.get_stats() -> dict

Data Structures
===============

**SharedState** (C++ struct, ctypes)

.. code-block:: python

   class SharedState(ctypes.Structure):
       _fields_ = [
           ('cpp_writing', c_int),
           ('py_writing', c_int),
           ('player_x', c_float),
           ('player_y', c_float),
           ('player_vel_x', c_float),
           ('player_vel_y', c_float),
           ('player_rot', c_float),
           ('gravity', c_int),  # +1 or -1
           ('is_on_ground', c_int),  # 0 or 1
           ('is_dead', c_int),
           ('percent', c_float),  # 0-100
           ('player_mode', c_int),  # 0=Cube, 1=Ship
           ('objects', ObjectData * 30),
           ('dist_nearest_hazard', c_float),
           ('dist_nearest_solid', c_float),
           ('action_command', c_int),
           ('reset_command', c_int),
       ]

**ObjectData** (ctypes)

.. code-block:: python

   class ObjectData(ctypes.Structure):
       _fields_ = [
           ('dx', c_float),  # Relative X
           ('dy', c_float),  # Relative Y
           ('w', c_float),   # Width
           ('h', c_float),   # Height
           ('type', c_int),  # 1=spike, 2=block, 5=portal, -1=empty
       ]

**Episode Info** (returned by step())

.. code-block:: python

   info = {
       'percent': 25.5,  # Level completion %
       'mode': 0,  # 0=Cube, 1=Ship
       'hazard_dist': 15.2,  # Nearest spike
       'solid_dist': 8.5,  # Nearest block
       'reward_components': {  # Broken down rewards
           'progress': 2.5,
           'penalty': -0.0001,
           'bonus': 0.0,
       }
   }

Error Handling
==============

**RuntimeError: Could not find Shared Memory**

Indicates Geometry Dash not running or mod not enabled.

.. code-block:: python

   try:
       env = GeometryDashEnv()
   except RuntimeError as e:
       print("GD not running. Start it with utridu mod enabled.")

**ValueError: Invalid action**

Action must be in {0, 1}.

.. code-block:: python

   action = env.action_space.sample()  # Safe random action
   obs, reward, done, _, _ = env.step(action)

**NaN in Observations**

Rare case if reward calculation fails (divide by zero).

.. code-block:: python

   obs, _ = env.reset()
   if np.isnan(obs).any():
       print("Warning: NaN in observation")
       obs = np.nan_to_num(obs)  # Replace with 0

See :doc:`../limitations` for known issues and mitigations.

Full Module Documentation
==========================

.. toctree::
   :maxdepth: 3

   agents
   environment
   curriculum

See also:
- :doc:`../algorithm` for algorithmic details
- :doc:`../architecture` for system design
- :doc:`../performance` for benchmarks
- :doc:`../changelog` for version history
