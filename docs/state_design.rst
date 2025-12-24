State Design & Feature Engineering
===================================

Detailed explanation of state representation and input normalization.

Raw State Collection
--------------------

The C++ Geode mod collects 154 features per frame:

.. code-block:: cpp

   struct SharedState {
       // Player Physics (6 floats + 5 ints)
       float player_x, player_y;              // Position
       float player_vel_x, player_vel_y;      // Velocity
       float player_rot;                      // Rotation
       int gravity;                           // +1 or -1
       int is_on_ground;
       int is_dead;
       int is_terminal;
       int player_mode;                       // 0=Cube, 1=Ship
       float player_speed;                    // Internal speed setting
       
       // Level Progress
       float percent;                         // 0-100% completion
       
       // Reward Signals
       float dist_nearest_hazard;
       float dist_nearest_solid;
       
       // Nearby Objects (30 max)
       ObjectData objects[30];                // 5 floats each
   };

**Reasoning**:
- Player physics: Essential for understanding state
- Velocity: Crucial for jump timing
- Nearest hazard/solid: Explicit reward signal (no need to infer)
- Objects: Context for decision-making

For full implementation details, see :doc:`../architecture`.

Feature Normalization
---------------------

Raw values scaled to appropriate ranges before network input.

.. code-block:: python

   def normalize_state(state):
       """
       Convert raw C++ struct to normalized numpy array (154,).
       """
       # Player Features (4)
       player_data = [
           state.player_vel_y / 30.0,         # Max velocity ~30
           state.player_y / 900.0,            # Playable height
           float(state.is_on_ground),         # {0.0, 1.0}
           float(state.player_mode)           # {0.0, 1.0}
       ]
       
       # Object Features (30 * 5 = 150)
       obj_features = []
       for i in range(30):
           obj = state.objects[i]
           obj_features.extend([
               obj.dx / 1000.0,               # Visibility distance
               obj.dy / 300.0,                # Vertical range
               obj.w / 50.0,                  # Typical width
               obj.h / 50.0,                  # Typical height
               float(obj.type) / 10.0         # Object type ID
           ])
       
       # Combine
       full_state = np.array(player_data + obj_features, dtype=np.float32)
       
       # Handle NaN/Inf (graceful degradation)
       full_state = np.nan_to_num(
           full_state,
           nan=0.0,      # Missing values → neutral
           posinf=1.0,   # Out-of-range → max
           neginf=-1.0   # Out-of-range → min
       )
       
       return full_state

**Normalization Rationale**:

| Feature | Scale | Rationale |
|---------|-------|-----------|
| vel_y / 30 | [-1, 1] | Max upward/downward velocity |
| player_y / 900 | [0, 1] | Level height bound |
| is_on_ground | {0, 1} | Discrete flag |
| player_mode | {0, 1} | Discrete mode |
| dx / 1000 | [0, 1] | Visibility range (1000 px ahead) |
| dy / 300 | [-1, 1] | Vertical range (±300 px) |
| w, h / 50 | [0, 2] | Typical obstacle size |
| type / 10 | [0, 1] | Arbitrary object type |

**Why Normalize?**

1. **Neural Network Stability**: Inputs in [0, 1] or [-1, 1] prevent exploding gradients
2. **Learning Speed**: Gradient magnitudes balanced across features
3. **Generalization**: Normalized inputs reduce overfitting to specific ranges

**Validation**:

.. code-block:: python

   # After normalization, check statistics
   obs = normalize_state(raw_state)
   print(f"Mean: {obs.mean():.4f}")     # Should be ~0
   print(f"Std:  {obs.std():.4f}")      # Should be ~0.3-0.5
   print(f"Min:  {obs.min():.4f}")      # Should be around -1
   print(f"Max:  {obs.max():.4f}")      # Should be around +1

Frame Stacking
--------------

Observations are stacked (repeated) to provide temporal context.

.. code-block:: python

   class GeometryDashEnv(gym.Env):
       def __init__(self):
           self.frame_stack = 2  # Stack 2 observations
           self._frame_buffer = deque(maxlen=2)
           
       def step(self, action):
           # ... apply action ...
           obs_single = normalize_state(raw_state)
           
           # Initialize buffer on first observation
           if len(self._frame_buffer) == 0:
               for _ in range(self.frame_stack):
                   self._frame_buffer.append(obs_single.copy())
           else:
               self._frame_buffer.append(obs_single)
           
           # Stack frames along dimension 0
           obs_stacked = np.concatenate(list(self._frame_buffer), axis=0)
           # Shape: (308,) = 2 * 154
           
           return obs_stacked, reward, terminated, truncated, info

**Why Stack?**

1. **Temporal Context**: Agent knows recent state history
2. **Velocity Computation**: (pos_t - pos_{t-1}) approximates velocity
3. **Reaction Time**: Agent can plan based on state trajectory
4. **Mitigate Occlusion**: If object appears one frame, agent sees 2 frames

**Frame Skip Trade-off**:

.. code-block:: text

   Frame Skip = 1 (No skipping):
   ├─ Resolution: Every frame decision (60 Hz)
   ├─ Latency: Network must respond in 16.7 ms
   ├─ Exploration: High-frequency updates
   └─ Downside: Cannot keep up with GPU; training slow
   
   Frame Skip = 4 (Current):
   ├─ Resolution: Every 67 ms decision
   ├─ Latency: Network can respond in 64 ms budget
   ├─ Exploration: Agent effectively plans 200+ ms ahead
   └─ Benefit: Smooth, feasible training
   
   Frame Skip = 10 (Too high):
   ├─ Latency: Very loose; agent lags
   ├─ Planning Horizon: Too long; hard to learn fine-grained control
   └─ Result: Agent misses opportunities

**Current**: Frame skip = 4, Frame stack = 2 (good balance)

Object Feature Engineering
--------------------------

30 nearest obstacles represented as 150 features.

**Object Filtering** (C++):

.. code-block:: cpp

   for (GameObject* go : m_objects) {
       CCRect objRect = go->getObjectRect();
       float dx = objRect.getMinX() - playerRect.getMaxX();
       float dy = objRect.getMidY() - playerRect.getMidY();
       
       // Only nearby objects
       if (dx < -50.0f || dx > 800.0f) continue;
       if (abs(dy) > 200.0f) continue;
       
       // Skip decoration
       if (category == "decoration") continue;
       
       // Add to list
       nearby.push_back({dx, dy, w, h, category});
   }

**Sorted Order**: Objects sorted by dx (distance ahead) so agent learns to process left-to-right.

**Python Representation**:

.. code-block:: python

   # For each of 30 objects
   for i in range(30):
       obj = state.objects[i]
       if obj.type == -1:  # Empty slot
           features = [0, 0, 0, 0, 0]  # Zero padding
       else:
           features = [
               obj.dx / 1000.0,        # Where is it (horizontal)
               obj.dy / 300.0,         # How high/low
               obj.w / 50.0,           # How wide
               obj.h / 50.0,           # How tall
               float(obj.type) / 10.0  # What type
           ]
       obj_features.extend(features)

**Object Types**:

.. code-block:: text

   Type 0: Unknown / Decoration (ignored)
   Type 1: Spike / Hazard (deadly)
   Type 2: Solid Block (traversable)
   Type 5: Portal (mode switch)
   Type -1: Empty / Padding (no object)

State Augmentation (Optional)
-----------------------------

Additional features that could improve learning (not currently used):

.. code-block:: python

   # Time-to-collision (estimated)
   ttc = obj.dx / max(player.vel_x, 0.1)
   
   # Object density (cluster detection)
   objects_per_window = count_objects_in_range(dx=[0, 200])
   
   # Gravity awareness
   gravity_direction = player.gravity  # +1 or -1
   
   # Recent history (velocity delta)
   vel_change = player.vel_y - prev_vel_y
   
   # Target corridor (for Ship mode)
   target_y = 235.0
   distance_to_center = abs(player_y - target_y)

**Potential Benefits**:
- Better predictions of collision events
- Mode-specific features
- Implicit knowledge of physics

**Trade-off**: Increases input dimension (154 → 170+), slower training. Not justified currently; included for reference.

State Statistics (Empirical)
----------------------------

From 1000 random frames during training:

.. code-block:: text

   Feature                    Mean        Std         Min         Max
   ────────────────────────────────────────────────────────────────
   vel_y / 30                  0.15       0.45       -0.98       0.98
   player_y / 900              0.35       0.20        0.01       0.99
   is_on_ground               0.62        0.49        0.00       1.00
   player_mode                 0.38       0.48        0.00       1.00
   ────────────────────────────────────────────────────────────────
   object dx / 1000           0.40       0.35        0.00       0.99
   object dy / 300            -0.02      0.45       -0.98       0.98
   object w / 50              0.30       0.35        0.00       0.95
   object h / 50              0.40       0.40        0.00       0.99
   object type / 10           0.15       0.20        0.00       1.00

**Insights**:
- Velocity well-distributed (good feature coverage)
- Position slightly skewed toward bottom (gravity pulls down)
- Objects spread across viewing window (good utilization)
- Mode 38% ship, 62% cube (reflects slice distribution)

See :doc:`../architecture` for how these features flow through the network.
