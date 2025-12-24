Installation & Setup
====================

This guide covers environment setup, dependencies, and verification procedures.

Prerequisites
-------------

**System Requirements:**

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA compute capability 5.2+ (recommended; CPU fallback supported)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 2GB free for checkpoints and logs

**Software Requirements:**

- Python 3.10 or higher
- Geometry Dash v2.2074
- Geode mod loader v4.10.0+
- Visual Studio 2022 (for C++ compilation)
- CMake 3.20+

Step 1: Install Geometry Dash & Geode
--------------------------------------

1. **Ensure Geometry Dash v2.2074 is installed**

   .. code-block:: powershell

      # Verify via Steam or direct installation
      dir "C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash"

2. **Download Geode Loader**

   Visit: https://github.com/geode-sdk/geode/releases

   Download: ``geode-installer-v4.10.0-win.exe``

3. **Install Geode**

   .. code-block:: powershell

      # Run installer with administrator privileges
      # Select Geometry Dash installation directory
      # Verify by checking: gd_install_dir/Geometry Dash.exe

4. **Verify Installation**

   Launch Geometry Dash. You should see the Geode mod loader icon in the bottom-right corner.

Step 2: Build & Install utridu Mod
-----------------------------------

The ``utridu`` mod handles real-time state collection and action injection.

**2.1: Navigate to mod directory**

.. code-block:: powershell

   cd "D:\geometry_dash_RL\geode\mods\utridu"

**2.2: Create build directory**

.. code-block:: powershell

   mkdir build
   cd build

**2.3: Generate Visual Studio project**

.. code-block:: powershell

   cmake -G "Visual Studio 17 2022" -A x64 ..

**2.4: Compile Release build**

.. code-block:: powershell

   cmake --build . --config Release

**2.5: Install mod file**

.. code-block:: powershell

   # Copy compiled .geode file to Geometry Dash mod directory
   copy "Release\utridu.geode" "$GD_INSTALL_DIR\geode\mods\"

**2.6: Verify mod loads**

1. Launch Geometry Dash
2. Click Geode icon → Mods → Enable ``utridu``
3. Load any level (no data persists at this point)
4. Press 'M' to toggle HUD
5. You should see: ``RL: Memory Ready (Press M to Toggle HUD)``

.. warning::

   If you see ``ERR: CreateFile Failed! (Run Admin)``, run Geometry Dash as administrator.

Step 3: Python Environment Setup
---------------------------------

**3.1: Create virtual environment**

.. code-block:: powershell

   cd "D:\geometry_dash_RL\Stereo_Madness"
   python -m venv venv
   .\venv\Scripts\Activate.ps1

**3.2: Upgrade pip**

.. code-block:: powershell

   python -m pip install --upgrade pip setuptools wheel

**3.3: Install PyTorch with CUDA support**

For RTX 30/40 series (CUDA 11.8):

.. code-block:: powershell

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For CPU-only (slower):

.. code-block:: powershell

   pip install torch torchvision torchaudio

Verify PyTorch installation:

.. code-block:: powershell

   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

**3.4: Install project dependencies**

.. code-block:: powershell

   pip install gymnasium numpy keyboard

**3.5: (Optional) Install development tools**

For documentation, testing, and analysis:

.. code-block:: powershell

   pip install sphinx sphinx-rtd-theme sphinx-tabs tensorboard pandas matplotlib seaborn

Full dependency list: ``requirements.txt``

.. code-block:: powershell

   pip install -r requirements.txt

Step 4: Configuration
---------------------

**4.1: Verify shared memory parameters**

Edit ``config.py``:

.. code-block:: python

   # --- SHARED MEMORY ---
   MEM_NAME = "GD_RL_Memory"
   MEM_SIZE_BYTES = 1024

**Must match C++ struct size** in ``src/main.cpp``:

.. code-block:: cpp

   struct SharedState {
       volatile int cpp_writing;
       volatile int py_writing;
       float player_x, player_y, player_vel_x, player_vel_y, player_rot;
       int gravity, is_on_ground, is_dead, is_terminal;
       float percent, dist_nearest_hazard, dist_nearest_solid;
       int player_mode;
       float player_speed;
       ObjectData objects[30];
       int action_command, reset_command, checkpoint_command;
   };

**4.2: Configure hyperparameters**

Edit ``config.py`` to adjust training behavior:

.. code-block:: python

   # --- HYPERPARAMETERS ---
   GAMMA = 0.99                # Discount factor
   BATCH_SIZE = 64             # Experience batch size
   LR = 0.0003                 # Network learning rate
   MEMORY_SIZE = 50000         # Replay buffer capacity
   TARGET_UPDATE = 1000        # Target network sync interval
   EPSILON_START = 1           # Initial exploration rate
   EPSILON_END = 0.01          # Final exploration rate
   EPSILON_DECAY = 50000       # Decay schedule

See :doc:`../performance` for tuned recommendations.

Step 5: Verification & Testing
-------------------------------

**5.1: Run smoke test**

.. code-block:: powershell

   cd "D:\geometry_dash_RL\Stereo_Madness"
   python -c "
   import torch
   from config import *
   from core.environment import GeometryDashEnv
   from agents.ddqn import Agent
   
   print(f'✓ PyTorch version: {torch.__version__}')
   print(f'✓ CUDA available: {torch.cuda.is_available()}')
   print(f'✓ Device: {DEVICE}')
   
   # Test environment connection (requires game running)
   try:
       env = GeometryDashEnv()
       print(f'✓ Environment connected')
       print(f'✓ Action space: {env.action_space}')
       print(f'✓ Observation space: {env.observation_space}')
   except Exception as e:
       print(f'✗ Environment failed: {e}')
   "

**5.2: Pre-training checklist**

Before starting training, ensure:

- [ ] Geometry Dash v2.2074 installed
- [ ] Geode mod loader running (icon visible)
- [ ] utridu mod enabled (visible in mod list)
- [ ] Python venv activated
- [ ] PyTorch installed with CUDA support
- [ ] All dependencies installed: ``pip list | grep -E "torch|gymnasium|numpy|keyboard"``
- [ ] Shared memory struct size matches (1024 bytes)
- [ ] config.py hyperparameters reviewed
- [ ] Checkpoints directory exists: ``checkpoints/``
- [ ] Logs directory exists: ``logs/``

**5.3: Launch training**

Once all checks pass:

.. code-block:: powershell

   python main.py

You should see output like:

.. code-block:: text

   [System] Loading All Expert Models into RAM...
   [Curriculum] Starting fresh from Slice 1: Intro - Simple Cube jumps
   [Training] Starting Slice 1 Mastery...
   Ep 1    | Win% 5.1%   | % 12.5  | Reward 45.23  | Loss 0.0234
   Ep 2    | Win% 10.2%  | % 18.3  | Reward 38.12  | Loss 0.0189
   ...

Troubleshooting
---------------

**"Could not find Shared Memory 'GD_RL_Memory'"**

- [ ] Verify Geometry Dash is running
- [ ] Verify utridu mod is enabled
- [ ] Load a level (menu doesn't initialize memory)
- [ ] Press 'M' in-game to toggle HUD; should display memory status

**"ERR: CreateFile Failed! (Run Admin)"**

- [ ] Right-click Geometry Dash → Run as Administrator
- [ ] Or: Whitelist in antivirus software

**"CUDA out of memory"**

- [ ] Reduce BATCH_SIZE in config.py (default: 64 → try 32)
- [ ] Reduce MEMORY_SIZE (default: 50000 → try 20000)
- [ ] Switch to CPU: Set DEVICE = torch.device("cpu")

**"Low FPS in game during training"**

- [ ] Normal behavior during inference (~15ms latency per frame)
- [ ] Reduce frame_skip value in main.py (tradeoff: slower learning)
- [ ] Upgrade GPU or switch to CPU (much slower training)

**"Agent not learning; losses not decreasing"**

- [ ] Check reward shapes in expert_cube.py / expert_ship.py
- [ ] Verify input normalization in state_utils.py (check for NaN propagation)
- [ ] Reduce learning rate (LR = 0.0001)
- [ ] Increase batch size for more stable gradients (BATCH_SIZE = 128)

**"Training crashes with 'stuck level' error**

- [ ] Verify level auto-checkpoint disabled (should be in init)
- [ ] Check C++ stuck-frame detection threshold (default: 30 frames / 0.5s)
- [ ] Ensure no manual pause during training

Environment Variables
---------------------

Optional environment variables for advanced configuration:

.. code-block:: powershell

   # Force CPU training (ignores GPU)
   [Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "-1", "User")
   
   # Reduce cuDNN overhead for small models
   [Environment]::SetEnvironmentVariable("CUDNN_BENCHMARK", "0", "User")
   
   # Disable TensorFloat-32 for deterministic training
   [Environment]::SetEnvironmentVariable("CUDA_LAUNCH_BLOCKING", "1", "User")

Next Steps
----------

- **Ready to train?** → See :doc:`quickstart`
- **Want to understand the algorithm?** → Read :doc:`../algorithm`
- **Curious about architecture?** → Check :doc:`../architecture`
- **Encountering more issues?** → Open a GitHub issue
