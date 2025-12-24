import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from agents.base_dqn import DuelingDQN
from config import DEVICE, CHECKPOINT_DIR
import os

def visualize_decision_boundary(slice_id=1):
    model_path = os.path.join(CHECKPOINT_DIR, f"slice_{slice_id:02d}_current.pth")
    if not os.path.exists(model_path):
        print("Model not found. Train for at least 50 episodes first!")
        return

    # Load Model
    model = DuelingDQN().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create Search Space (Distance to Spike vs vertical Velocity)
    # I simulate 100x100 different game states
    dist_range = np.linspace(0, 500, 100) # 0 to 500 units away
    vel_range = np.linspace(-15, 15, 100)  # -15 to +15 vertical velocity
    
    heatmap = np.zeros((100, 100))

    with torch.no_grad():
        for i, dist in enumerate(dist_range):
            for j, vel in enumerate(vel_range):
                # Build a dummy state (154 inputs)
                state = np.zeros(154)
                state[0] = vel / 30.0    # Normalized Vel Y
                state[1] = 0.5           # Fixed Y height
                state[4] = dist / 1000.0 # Normalized Dist to Spike 1
                
                state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                
                # Get Q-Values
                q_values = model(state_t)
                # Probability of choosing 'Jump' (Action 1)
                probs = torch.softmax(q_values, dim=1)
                heatmap[j, i] = probs[0, 1].item() # Store probability of Jump

    # Plotting
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap, xticklabels=20, yticklabels=20, cmap="icefire", cbar_kws={'label': 'Jump Probability'})
    
    # Labeling
    ax.set_title(f"Brain Decision Boundary: Slice {slice_id}", fontsize=15, pad=20)
    ax.set_xlabel("Distance to Hazard (Units)", fontsize=12)
    ax.set_ylabel("Vertical Velocity (Upward <-> Downward)", fontsize=12)
    
    # Fix Ticks
    plt.xticks(np.linspace(0, 100, 5), np.linspace(0, 500, 5).astype(int))
    plt.yticks(np.linspace(0, 100, 5), np.linspace(-15, 15, 5).astype(int))

    plt.tight_layout()
    plt.show()

visualize_decision_boundary(slice_id=1)
