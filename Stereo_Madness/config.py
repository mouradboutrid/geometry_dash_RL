import os
import torch

# --- PROJECT SETTINGS ---
PROJECT_NAME = "GD_RL_Agent"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PATHS (Dynamic based on structure) ---
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CURRICULUM_FILE = os.path.join(BASE_DIR, "curriculum", "slice_definitions.json")

# Log Files
TRAIN_LOG = os.path.join(LOG_DIR, "training_log.csv")
DEATH_LOG = os.path.join(LOG_DIR, "death_log.csv")
META_FILE = os.path.join(LOG_DIR, "training_meta.json")

# --- SHARED MEMORY ---
MEM_NAME = "GD_RL_Memory"
MEM_SIZE_BYTES = 1024  # Ensure this matches your C++ struct size roughly

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMETERS ---
GAMMA = 0.99                # Discount Factor (Future reward importance)
BATCH_SIZE = 64             # Replay Buffer Batch Size
LR = 0.0003                 # Learning Rate
MEMORY_SIZE = 50000         # Max Transitions in Buffer
TARGET_UPDATE = 1000        # Steps between Target Net updates

# --- EXPLORATION (Epsilon Greedy) ---
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 50000       # Decay over approx 50000 frames

# --- INPUT SHAPE ---
# 4 Player Vars + (30 Objects * 5 Vars) = 154 inputs
INPUT_DIM = 154
OUTPUT_DIM = 2 # (Hold or Release)