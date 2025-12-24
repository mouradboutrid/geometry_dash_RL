import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import time
import config
from config import TRAIN_LOG, PROJECT_NAME

def animate(i):
    # Check if data exists
    if not os.path.exists(TRAIN_LOG):
        print("Waiting for training logs...")
        return

    try:
        # Read Data
        # Expected columns: Episode, Reward, Epsilon, Slice
        data = pd.read_csv(TRAIN_LOG)
        if data.empty: return
    except:
        return # Handle file read collisions smoothly

    # Clear Plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # PLOT 1: Learning Curve (Raw Reward)
    # Shows the raw performance per episode. High variance is normal.
    ax1.plot(data['Episode'], data['Reward'], color='#00ff00', linewidth=0.8, alpha=0.8)
    ax1.set_title("Total Reward per Episode", color='white', fontsize=10)
    ax1.set_facecolor('#1e1e1e')
    ax1.grid(color='#333', linestyle='--')

    # PLOT 2: Exploration Decay
    # Shows the Epsilon value dropping toward 0 (Pure AI).
    ax2.plot(data['Episode'], data['Epsilon'], color='#00ccff', linewidth=1.5)
    ax2.set_title("Epsilon (Exploration Rate)", color='white', fontsize=10)
    ax2.set_facecolor('#1e1e1e')
    ax2.grid(color='#333', linestyle='--')

    # PLOT 3: Curriculum Progression
    # Step chart showing which Slice the agent is currently playing.
    ax3.step(data['Episode'], data['Slice'], where='post', color='#ff9900', linewidth=2)
    ax3.set_title("Curriculum Level (Slice ID)", color='white', fontsize=10)
    ax3.set_facecolor('#1e1e1e')
    ax3.set_yticks(range(1, 10)) # Slices 1-9
    ax3.grid(color='#333', linestyle='--')

    # PLOT 4: Moving Average (Trend)
    # The most important line. Shows if the agent is actually getting smarter.
    if len(data) > 50:
        rolling_avg = data['Reward'].rolling(window=50).mean()
        ax4.plot(data['Episode'], rolling_avg, color='#ff00ff', linewidth=1.5)
    ax4.set_title("50-Episode Moving Average (Trend)", color='white', fontsize=10)
    ax4.set_facecolor('#1e1e1e')
    ax4.grid(color='#333', linestyle='--')

# MATPLOTLIB SETUP
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
fig.suptitle(f"{PROJECT_NAME} - Real-time Training Hub", fontsize=16, color='white', weight='bold')

# Create 2x2 Grid
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# Start Animation (Update every 2000ms = 2 seconds)
ani = animation.FuncAnimation(fig, animate, interval=2000)


print("Starting Dashboard... (Ensure training is running)")
plt.tight_layout()
plt.show()
