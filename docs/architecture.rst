Architecture Overview
=====================

This document describes the complete system architecture, including Python training pipeline, C++ game integration, and cross-language synchronization.

High-Level System Design
------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                 GEOMETRY DASH RL SYSTEM OVERVIEW                │
   └─────────────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────────────┐
   │                   PYTHON TRAINING LAYER                         │
   │  ┌────────────────────────────────────────────────────────────┐ │
   │  │ GDAgentOrchestrator (main.py)                              │ │
   │  │  - Episode loop management                                 │ │
   │  │  - Curriculum progression logic                            │ │
   │  │  - Expert model caching & relay navigation                 │ │
   │  └────────────────────────────────────────────────────────────┘ │
   │                              ↓↑                                 │
   │  ┌────────────────────────────────────────────────────────────┐ │
   │  │ Module Ecosystem                                           │ │
   │  │  ┌──────────┬─────────────┬────────────┬──────────────┐    │ │
   │  │  │  Agent   │ Environment │ Curriculum │ Replay       │    │ │
   │  │  │  (DDQN)  │ (Gymnasium) │ (Manager)  │ Buffer       │    │ │
   │  │  └──────────┴─────────────┴────────────┴──────────────┘    │ │
   │  └────────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────────┘
                                   ↓↑
   ┌─────────────────────────────────────────────────────────────────┐
   │           SHARED MEMORY INTERFACE (mmap + spinlock)             │
   │  - Windows named pipe: "GD_RL_Memory"                           │
   │  - Size: 1024 bytes (struct alignment)                          │
   │  - Synchronization: volatile int flags (cpp_writing, py_writing)│
   │  - Latency: <1ms for I/O, spinlock contention minimal           │
   └─────────────────────────────────────────────────────────────────┘
                                   ↓↑
   ┌─────────────────────────────────────────────────────────────────┐
   │                   C++ GEODE MOD LAYER                           │
   │  (utridu/src/main.cpp)                                          │
   │  ┌────────────────────────────────────────────────────────────┐ │
   │  │ MyPlayLayer::rl_loop (60 Hz)                               │ │
   │  │  - State collection & normalization                        │ │
   │  │  - Object detection & distance computation                 │ │
   │  │  - Reward signal preparation                               │ │
   │  │  - Action injection (spinlock + transition logic)          │ │
   │  └────────────────────────────────────────────────────────────┘ │
   └─────────────────────────────────────────────────────────────────┘
                                    ↓↑
   ┌─────────────────────────────────────────────────────────────────┐
   │              GEOMETRY DASH ENGINE (Cocos2D)                     │
   │  - Frame-by-frame physics simulation                            │
   │  - Player state management (velocity, collision detection)      │
   │  - Object management (obstacles, portals, gravity switches)     │
   │  - Rendering & game loop (60 FPS)                               │
   └─────────────────────────────────────────────────────────────────┘

Python Training Pipeline
------------------------

**Main Components**

The ``GDAgentOrchestrator`` class orchestrates the complete training flow:

.. code-block:: python

   class GDAgentOrchestrator:
       def __init__(self):
           self.env = GeometryDashEnv()          # Gymnasium wrapper
           self.manager = CurriculumManager()    # Progression control
           self.memory = ReplayBuffer(50K)       # Experience storage
           self.agent = Agent(...)               # DDQN learner
           self.experts_cache = {}               # Previous expert models
       
       def train(self):
           while True:
               episode = 0
               while not self.manager.should_promote():
                   # Collect experience
                   obs, _ = self.env.reset()
                   while not terminated:
                       action = self.agent.select_action(obs)
                       obs, reward, terminated, _, info = self.env.step(action)
                       self.memory.push(obs, action, reward, next_obs, terminated)
                   
                   # Train on batch
                   loss = self.agent.learn(self.memory)
                   
                   # Track progress
                   success = info['percent'] >= slice_end
                   self.manager.update(success, steps)
               
               # Promoted! Save expert and move to next slice

Component Breakdown
^^^^^^^^^^^^^^^^^^^

**1. GeometryDashEnv (Gymnasium Wrapper)**

.. code-block:: python

   class GeometryDashEnv(gym.Env):
       def __init__(self):
           self.bridge = MemoryBridge()  # IPC connection
           self.action_space = Discrete(2)  # Jump or release
           self.observation_space = Box(shape=(154,))  # Normalized state
       
       def step(self, action):
           # Frame skipping loop (skip_frames=4)
           for _ in range(self.frame_skip):
               self.bridge.write_action(action)
               raw_state = self.bridge.read_state()
               reward = self._calculate_reward(...)
           
           # Frame stacking (stack_frames=2)
           obs = self._stack_frames(raw_state)
           return obs, reward, terminated, _, info

Key responsibilities:

- Connect to shared memory via MemoryBridge
- Apply frame skipping (repeat actions 4x) for temporal abstraction
- Stack frames (2 consecutive observations) for temporal context
- Normalize raw state to standardized vector
- Calculate dense rewards
- Detect episode termination

**2. Agent (Dueling DQN)**

.. code-block:: python

   class Agent:
       def __init__(self, input_dim, output_dim, config):
           self.online_net = DuelingDQN(input_dim, output_dim)
           self.target_net = DuelingDQN(input_dim, output_dim)
           self.optimizer = torch.optim.Adam(...)
       
       def select_action(self, state, is_training=True):
           # Epsilon-greedy exploration
           if training and random() < epsilon:
               return random_action()
           else:
               return self.online_net(state).argmax().item()
       
       def learn(self, memory):
           # Sample batch
           batch = memory.sample(batch_size=64)
           
           # Double DQN loss computation
           current_q = self.online_net(batch.state).gather(1, batch.action)
           best_actions = self.online_net(batch.next_state).argmax(1)
           next_q = self.target_net(batch.next_state).gather(1, best_actions)
           target_q = batch.reward + gamma * (1 - batch.done) * next_q
           
           # Gradient descent
           loss = F.mse_loss(current_q, target_q)
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()

Core functionality:

- Epsilon-greedy action selection with decay schedule
- Double DQN learning with target network synchronization
- Checkpoint save/load for resumption

**3. ReplayBuffer (Experience Memory)**

.. code-block:: python

   class ReplayBuffer:
       def __init__(self, capacity=50000):
           self.buffer = deque(maxlen=capacity)  # FIFO circular buffer
       
       def push(self, s, a, r, s_prime, done):
           self.buffer.append((s, a, r, s_prime, done))
       
       def sample(self, batch_size):
           batch = random.sample(self.buffer, batch_size)
           return unzip(batch)  # Returns (states, actions, rewards, next_states, dones)

Design notes:

- Circular deque (FIFO) automatically discards old experiences
- Uniform random sampling decorrelates temporal sequences
- 50K transitions accommodate ~10 full training runs per slice

**4. CurriculumManager (Progression Control)**

.. code-block:: python

   class CurriculumManager:
       def __init__(self):
           self.slices = load_json("slice_definitions.json")  # 8 slices
           self.slice_idx = 0
           self.wins_window = []  # Rolling 50-episode window
       
       def update(self, episode_success, steps):
           self.wins_window.append(1 if episode_success else 0)
           return sum(self.wins_window) / len(self.wins_window)
       
       def should_promote(self):
           # Criteria: ≥20 episodes AND ≥70% win rate
           return len(self.wins_window) >= 20 and rate >= 0.70
       
       def advance_slice(self):
           self.slice_idx += 1
           self.wins_window = []  # Reset metrics for new slice

Responsibilities:

- Track curriculum progress across 9 slices
- Determine promotion eligibility
- Reset statistics between slices
- Persist progress to JSON for resumption

Cross-Language Synchronization
-------------------------------

**Shared Memory Protocol**

The core synchronization mechanism uses Windows named memory mapping:

.. code-block:: cpp

   // C++ (Geode Mod)
   struct SharedState {
       volatile int cpp_writing;    // 1 when C++ writes
       volatile int py_writing;     // 1 when Python reads
       // ... state fields ...
       int action_command;          // Action from Python
   };

   // Python (MemoryBridge)
   class SharedState(ctypes.Structure):
       _fields_ = [
           ("cpp_writing", ctypes.c_int),
           ("py_writing", ctypes.c_int),
           # ... state fields ...
           ("action_command", ctypes.c_int),
       ]

**Spinlock Mutual Exclusion**

.. code-block:: text

   Timeline (t in milliseconds)
   
   t=0   : C++ writes state (cpp_writing=1), Python idle
   t=0.5 : C++ finishes (cpp_writing=0), releases lock
   t=1   : Python reads (py_writing=1), C++ waits
   t=1.5 : Python finishes (py_writing=0), releases lock
   t=2   : C++ resumes (next frame)
   ...
   
   Contention: ~0.5ms per operation on modern hardware
   Safety timeout: 5000 iterations (~2ms wall clock) to prevent deadlock

**Data Flow per Frame**

.. code-block:: text

   Frame T (Game at 60 FPS = 16.67ms per frame)
   
   [C++ Game Loop]
   │
   ├─→ Collect player state (position, velocity, rotation)
   │
   ├─→ Scan 30 nearest objects (obstacles, hazards, portals)
   │
   ├─→ Compute reward signals (nearest hazard distance, etc.)
   │
   ├─→ Write to shared memory (spinlock: wait if Python writing)
   │   - All floats/ints copied to memory region
   │   - Action command read (from previous frame's decision)
   │
   └─→ Apply action to player (press/release jump button)
   
   [Python Training Loop] (asynchronous, not frame-locked)
   │
   ├─→ Read state from shared memory (spinlock: wait if C++ writing)
   │
   ├─→ Normalize observation (scale/clip values)
   │
   ├─→ Network inference: state → action (3-5ms on GPU)
   │
   ├─→ Store in replay buffer
   │
   ├─→ Sample and train on batch (1-3ms per step)
   │
   └─→ Write action command to shared memory
       (next frame will apply this action)

**Synchronization Guarantees**

1. **No Corruption**: Spinlock ensures only one reader/writer at a time
2. **No Deadlock**: Timeout prevents infinite waits (safety=5000 iterations)
3. **Minimal Latency**: Shared memory I/O <1ms; spinlock contention rare (<5%)
4. **Causal Ordering**: Actions propagate in order (FIFO)

State Design
------------

**Raw State Collection** (C++, from game engine)

.. code-block:: cpp

   // Player properties
   pSharedMem->player_x = player.position.x;
   pSharedMem->player_y = player.position.y;
   pSharedMem->player_vel_x = (x_curr - x_prev) / dt;  // Computed velocity
   pSharedMem->player_vel_y = player.yVelocity;
   pSharedMem->player_rot = player.rotation;
   pSharedMem->gravity = player.is_upside_down ? -1 : 1;
   pSharedMem->is_on_ground = player.is_on_ground;
   pSharedMem->is_dead = player.is_dead;
   
   // Level progress
   pSharedMem->percent = (player.x / level.length) * 100.0;
   
   // Nearby objects (30 max)
   for (int i=0; i<nearby_objects.size(); i++) {
       pSharedMem->objects[i].dx = obj.x - player.x;  // Relative position
       pSharedMem->objects[i].dy = obj.y - player.y;
       pSharedMem->objects[i].w = obj.width;
       pSharedMem->objects[i].h = obj.height;
       pSharedMem->objects[i].type = category_int(obj.category);
   }

**Normalization** (Python, before network input)

.. code-block:: python

   def normalize_state(raw_state):
       # Player features (4)
       player_data = [
           raw_state.player_vel_y / 30.0,      # Max velocity ~30
           raw_state.player_y / 900.0,         # Playable height ~900
           float(raw_state.is_on_ground),      # Boolean → {0, 1}
           float(raw_state.player_mode),       # {0=Cube, 1=Ship}
       ]
       
       # Object features (30 × 5 = 150)
       obj_features = []
       for i in range(30):
           obj = raw_state.objects[i]
           obj_features.extend([
               obj.dx / 1000.0,    # Visibility range
               obj.dy / 300.0,     # Vertical range
               obj.w / 50.0,       # Typical object width
               obj.h / 50.0,       # Typical object height
               float(obj.type) / 10.0  # Object ID
           ])
       
       # Combine and sanitize
       full_state = np.array(player_data + obj_features, dtype=np.float32)
       full_state = np.nan_to_num(full_state, nan=0.0, posinf=1.0, neginf=-1.0)
       return full_state

**Rationale for normalization**:

- **Velocity**: Assume max ~30 pixels/frame in Geometry Dash physics
- **Position**: Level height is ~900 pixels
- **Object distance**: Visibility/interaction range ~1000 pixels ahead
- **NaN handling**: Game glitches occasionally produce invalid floats; clamping prevents propagation

C++ Integration (Geode Mod)
---------------------------

**Geode Framework Integration**

The mod hooks into Geometry Dash's ``PlayLayer`` class via Cocos2D:

.. code-block:: cpp

   class $modify(MyPlayLayer, PlayLayer) {
       // MyPlayLayer extends PlayLayer; Geode injects at runtime
       
       struct Fields {
           CCLabelTTF* m_statusLabel;      // HUD display
           CCDrawNode* m_debugDrawNode;    // Visual debugging
           float m_lastX;                  // For velocity computation
           float m_lastPercent;            // For stuck detection
           int m_stuckFrames;              // Frame counter
       };
       
       bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
           // Initialize shared memory
           hMapFile = CreateFileMappingA(...);
           pSharedMem = (SharedState*)MapViewOfFile(...);
           
           // Schedule RL loop
           this->schedule(schedule_selector(MyPlayLayer::rl_loop));
           return true;
       }
       
       void rl_loop(float dt) {
           // Called every frame (60 FPS)
           // Collect state, apply action
       }
   };

**Death Detection Logic**

The mod implements sophisticated death detection:

.. code-block:: cpp

   bool engineDead = m_player1->m_isDead;
   
   // Stuck frame detection (prevents infinite loops)
   if (percent_change < 0.0001 && current_percent > 0.5 && !engineDead) {
       m_fields->m_stuckFrames++;
   } else {
       m_fields->m_stuckFrames = 0;
   }
   bool isStuckDead = (m_fields->m_stuckFrames > 30);  // >0.5 sec stuck
   
   pSharedMem->is_dead = engineDead || isStuckDead;

**Action Application**

.. code-block:: cpp

   int cmd = pSharedMem->action_command;  // 0=Release, 1=Hold
   
   if (cmd == 1) {
       if (!m_fields->m_isHolding) {
           m_player1->pushButton(PlayerButton::Jump);  // Button press
           m_fields->m_isHolding = true;
       }
   } else {
       if (m_fields->m_isHolding) {
           m_player1->releaseButton(PlayerButton::Jump);  // Button release
           m_fields->m_isHolding = false;
       }
   }

State Transition Logic:
- Only trigger button press/release on state **change**
- Avoids repeatedly pressing already-pressed button
- Reduces input noise and improves reliability

Performance Characteristics
---------------------------

**Latency Breakdown (per frame)**

.. code-block:: text

   16.67 ms (60 FPS frame budget)
   
   ├─ C++ State Collection: 0.3 ms
   │  └─ Scan 30 objects, compute distances, write to memory
   │
   ├─ Shared Memory Spinlock: 0.1 ms (avg)
   │  └─ Wait for Python if reading; usually uncontended
   │
   ├─ Python Network Inference: 3-5 ms
   │  └─ Forward pass through DDQN network
   │
   ├─ Shared Memory Write: 0.1 ms
   │  └─ Write action command to memory
   │
   └─ Total: ~3.5-5.5 ms (21-33% of frame budget)
      Leaves 11-13 ms for training on background thread (if implemented)

Extensibility Points
--------------------

1. **New Action Spaces**: Modify ``OUTPUT_DIM`` in config.py; extend DuelingDQN output layer
2. **Alternative Algorithms**: Replace DuelingDQN with other networks; update Agent.learn()
3. **Multi-Level Support**: Create separate slice_definitions.json per level; adapt state normalization
4. **Parallel Training**: Implement async RL with multiple game instances (requires separate shared memory regions)

See :doc:`../contributing` for extension guidelines.

Next: :doc:`../curriculum` for the progression strategy.
