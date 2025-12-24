import numpy as np
import torch
from config import INPUT_DIM, DEVICE

def normalize_state(state):
    """
    Converts the raw C++ SharedState into a normalized Numpy array 
    ready for the Neural Network.
    
    Output Shape: (154,)
    """
    # Player Physics (4 features)
    # I assume standard GD physics ranges for normalization (Based on my runs in the game !)
    player_data = [
        state.player_vel_y / 30.0,   # Max Y vel is approx 20-30
        state.player_y / 900.0,      # Max height is approx 800-900
        float(state.is_on_ground),   # 0.0 or 1.0
        float(state.player_mode)     # 0.0 (Cube) or 1.0 (Ship)
    ]
    
    # Object Data (30 objects * 5 features = 150 features)
    obj_features = []
    for i in range(30):
        obj = state.objects[i]
        
        # If object is empty/far away, these will be default values
        obj_features.extend([
            obj.dx / 1000.0,         # Distance X (0 to ~1000)
            obj.dy / 300.0,          # Distance Y (-200 to +200)
            obj.w / 50.0,            # Width
            obj.h / 50.0,            # Height
            float(obj.type) / 10.0   # Object Type ID scaled down
        ])
        
    # Combine
    full_state = np.array(player_data + obj_features, dtype=np.float32)
    
    # Safety Check: Replace NaNs or Infs if game glitched
    full_state = np.nan_to_num(full_state, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return full_state

def to_tensor(obs):
    """Quick helper to convert numpy obs to PyTorch Tensor on GPU"""
    return torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
