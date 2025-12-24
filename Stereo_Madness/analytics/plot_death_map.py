import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import DEATH_LOG, PROJECT_NAME

def generate_death_heatmap():
    if not os.path.exists(DEATH_LOG):
        print("No death logs found yet. Keep training!")
        return

    # Load Data
    df = pd.read_csv(DEATH_LOG)
    if df.empty: return

    # Setup Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Density (The "Glow")
    # This creates a smooth 'mountain' showing where deaths cluster
    sns.kdeplot(data=df, x='Percent', fill=True, color='#ff0033', alpha=0.5, linewidth=2)
    
    # Plot Rug (Individual Death Points)
    # Tiny lines at the bottom for every single crash
    sns.rugplot(data=df, x='Percent', color='#ff0033', alpha=.2)

    # Highlight "The Gauntlet" (Stereo Madness Triple Spikes ~65-70% & 80-90%)
    danger_zones = [
        (13, 15, "First Spikes"),
        (64, 68, "Triple Spike 1"),
        (82, 88, "Final Gauntlet")
    ]
    
    for start, end, label in danger_zones:
        ax.axvspan(start, end, color='yellow', alpha=0.1)
        ax.text((start+end)/2, ax.get_ylim()[1]*0.9, label, 
                color='yellow', ha='center', fontsize=9, fontweight='bold')

    # Formatting
    ax.set_title(f"{PROJECT_NAME}: Fail Point Distribution", fontsize=15, color='white', pad=20)
    ax.set_xlabel("Level Progress (%)", fontsize=12, color='#aaaaaa')
    ax.set_ylabel("Death Density", fontsize=12, color='#aaaaaa')
    ax.set_xlim(0, 100)
    ax.grid(axis='y', color='#333333', linestyle='--')
    
    # Remove top/right spines
    sns.despine()

    plt.tight_layout()
    
    # Save a static copy for your records
    save_path = "logs/death_heatmap_latest.png"
    plt.savefig(save_path)
    print(f"Heatmap updated and saved to: {save_path}")
    plt.show()

generate_death_heatmap()
