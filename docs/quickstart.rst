Quickstart Guide
================

Get a training agent running in 5 minutes.

Minimum Setup (5 min)
---------------------

**Prerequisites**: Geometry Dash v2.2074, Geode installed, Python 3.10+

1. **Install Python dependencies**

   .. code-block:: bash

      cd Stereo_Madness
      pip install torch gymnasium numpy keyboard

2. **Build C++ mod**

   .. code-block:: bash

      cd geode/mods/utridu/build
      cmake -G "Visual Studio 17 2022" -A x64 ..
      cmake --build . --config Release

3. **Run training**

   .. code-block:: bash

      cd Stereo_Madness
      python main.py

Done! Your agent begins training on Slice 1.

Monitor Training
----------------

The training loop outputs metrics every episode:

.. code-block:: text

   Ep 1    | Win% -%     | % -  | Reward -  | Loss -
   Ep 2    | Win% -%     | % -  | Reward -  | Loss -
   Ep 3    | Win% -%     | % -  | Reward -  | Loss -
   ...
   Ep X    | Win% 70.0%  | % 10.3 | Reward 1100  | Loss 8.9
   [Curriculum] Promoted to Slice 2: Pre-Triple Spike

**Metrics explained:**

- ``Win%``: Success rate over last 50 episodes
- ``%``: Level completion percentage achieved this episode
- ``Reward``: Total episode reward (progress + bonuses - penalties)
- ``Loss``: Mean squared error between predicted and target Q-values

**Expected timeline:**

- Slice 1: 800, 1000 episodes to 70% success
- Slice 2-3: 1000, 6000 episodes each
- Slice 4 (ship mode): 1000 episodes (physics reset)
- Slice 5-8: - episodes each
- **Total**: 24-30 hours on CPU (i5)

Check Checkpoints
-----------------

Training automatically saves progress every 50 episodes:

.. code-block:: bash

   # Current training checkpoint (overwritten)
   checkpoints/slice_01_current.pth
   checkpoints/slice_02_current.pth
   ...

   # Final expert models (saved when promoted)
   checkpoints/final_models/slice_01_model.pth
   checkpoints/final_models/slice_02_model.pth
   ...

Resume from Checkpoint
----------------------

If training interrupts, simply run again:

.. code-block:: bash

   python main.py

The system automatically:
- Loads curriculum progress from ``logs/training_meta.json``
- Resumes from last slice
- Reloads previous expert weights
- Continues epsilon decay schedule

Manual Resume (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^

To resume from a specific checkpoint:

.. code-block:: python

   from Stereo_Madness.main import GDAgentOrchestrator
   
   orchestrator = GDAgentOrchestrator()
   # orchestrator.agent.load("checkpoints/slice_02_current.pth")
   orchestrator.train()

Analyze Training Logs
---------------------

Logs are saved to ``logs/`` directory:

**Training metrics**

.. code-block:: bash

   # CSV with episode metrics
   logs/training_log.csv
   # Columns: episode, slice_id, success, percent, reward, loss, epsilon

**Death analysis**

.. code-block:: bash

   # CSV with failure points
   logs/death_log.csv
   # Columns: episode, slice_id, death_percent, distance_to_completion

**Curriculum metadata**

.. code-block:: bash

   # JSON with progress checkpoint
   logs/training_meta.json
   # {"slice_idx": 2, "total_steps": 6000000}

Plot Results
^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Load training log
   df = pd.read_csv('logs/training_log.csv')
   
   # Plot success rate over time
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 2, 1)
   plt.plot(df['episode'], df['success'].rolling(50).mean())
   plt.xlabel('Episode')
   plt.ylabel('Success Rate (50-ep rolling avg)')
   plt.grid(True)
   
   plt.subplot(1, 2, 2)
   plt.plot(df['episode'], df['reward'])
   plt.xlabel('Episode')
   plt.ylabel('Episode Reward')
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

Run Trained Agent (Inference)
-----------------------------

After training completes, test the agent on a new game:

**Option 1: Evaluate trained model**

.. code-block:: python

   from Stereo_Madness.main import GDAgentOrchestrator
   import torch
   
   orchestrator = GDAgentOrchestrator()
   orchestrator.agent.load("checkpoints/final_models/slice_08_model.pth")
   
   # Run one episode without training
   obs, _ = orchestrator.env.reset()
   done = False
   total_reward = 0
   
   while not done:
       action = orchestrator.agent.select_action(obs, is_training=False)
       obs, reward, done, _, info = orchestrator.env.step(action)
       total_reward += reward
   
   print(f"Final percent: {info['percent']:.1f}%")
   print(f"Total reward: {total_reward:.2f}")

**Option 2: Ensemble evaluation**

.. code-block:: python

   # Test with all trained experts
   import os
   
   final_models_dir = "checkpoints/final_models"
   results = {}
   
   for filename in os.listdir(final_models_dir):
       if filename.endswith(".pth"):
           model_path = os.path.join(final_models_dir, filename)
           orchestrator.agent.load(model_path)
           
           # Run 10 episodes
           successes = 0
           for _ in range(10):
               obs, _ = orchestrator.env.reset()
               done = False
               while not done:
                   action = orchestrator.agent.select_action(obs, is_training=False)
                   obs, _, done, _, info = orchestrator.env.step(action)
               
               if info['percent'] >= 100.0:
                   successes += 1
           
           results[filename] = successes / 10.0
           print(f"{filename}: {results[filename]*100:.0f}% success")

Customize Hyperparameters
-------------------------

Edit ``config.py`` to tune training:

**Conservative training** (slower, more stable)

.. code-block:: python

   GAMMA = 0.99           # Standard
   BATCH_SIZE = 128       # Larger batch = more stable
   LR = 0.00015           # Lower learning rate
   TARGET_UPDATE = 2000   # Sync target net less often

**Aggressive training** (faster, less stable, I did not test it !)

.. code-block:: python

   GAMMA = 0.95           # Lower discount = focus on short-term
   BATCH_SIZE = 32        # Smaller batch = more varied gradients
   LR = 0.0005            # Higher learning rate
   TARGET_UPDATE = 500    # Sync target net more often

**Exploration tuning**

.. code-block:: python

   EPSILON_START = 1.0    # Full exploration initially
   EPSILON_END = 0.01     # Minimal exploration after decay
   EPSILON_DECAY = 100000 # Decay over more steps = slower exploration

For detailed parameter guidance, see :doc:`../performance`.

Common Customizations
---------------------

**Train on specific slice**

.. code-block:: python

   # In main.py, modify GDAgentOrchestrator.__init__:
   self.manager = CurriculumManager()
   self.manager.slice_idx = 2  # Start at Slice 3
   self.current_slice = self.manager.get_current_slice()

**Disable curriculum (single-slice training)**

.. code-block:: python

   # In main.py train() loop:
   # Replace: if self.manager.should_promote(): ...
   # With: if False: ...

**Adjust reward shaping**

.. code-block:: python

   # In environment.py step() method:
   reward_context = {
       'progress_scale': 10.0,      # Reduce progress bonus
       'jump_penalty': 0.001,       # Increase jump penalty
       'clearance_bonus': 0.02,     # Increase hazard avoidance reward
   }
   reward = self.env._calculate_reward(state, action, is_dead, reward_context)

**Change frame stacking/skipping**

.. code-block:: python

   # In main.py GDAgentOrchestrator.__init__:
   self.env.frame_skip = 2    # Skip every 2 frames (default: 4)
   self.env.frame_stack = 4   # Stack 4 frames (default: 2)

Next Steps
----------

-  **Training started?** Monitor progress via log files
-  **Want to understand the algorithm?** Read :doc:`../algorithm`
-  **Curious about architecture?** See :doc:`../architecture`
-  **Analyzing performance?** Check :doc:`../performance`
-  **Troubleshooting issues?** See :doc:`../installation` troubleshooting section
-  **Extending the project?** Review :doc:`../contributing`

Common Questions
----------------

**Q: Can I run this on CPU?**

A: Yes, That exactly why i use GEODE/Shared Memory instead of CNN 
**Q: How long until the agent completes the level?**

A: 24-30 hours of wall-clock training time on CPU. (less for succes rate < 70% but sacrifice precesion)

**Q: What if I stop training mid-slice?**

A: Progress automatically saves. Just run ``python main.py`` again to resume.

**Q: Can I train multiple agents in parallel?**

A: No (would require separate shared memory regions per agent). Future extension planned.

**Q: How do I evaluate on other levels?**

A: Requires manual slice definition and expert reward shaping. See :doc:`../future_work`.
