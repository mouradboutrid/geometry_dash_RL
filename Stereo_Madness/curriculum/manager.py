import json
import os
import numpy as np
from config import CURRICULUM_FILE, META_FILE, CHECKPOINT_DIR

class CurriculumManager:
    def __init__(self):
        # Load the Slice Definitions
        with open(CURRICULUM_FILE, 'r') as f:
            self.slices = json.load(f) 
            
        # Initialize Metrics
        self.slice_idx = 0       # Current index (0 to 8)
        self.wins_window = []    # Rolling window of last 50 episodes (0=Fail, 1=Success)
        self.total_steps = 0
        self.best_rate_current_slice = 0.0
        
        # Create Directories
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
        
        # Resume Progress if Meta file exists
        if os.path.exists(META_FILE):
            self.load_state()
        else:
            print(f"[Curriculum] Starting fresh from Slice 1: {self.slices[0]['description']}")

    def get_current_slice(self):
        """Returns the dictionary for the active slice."""
        return self.slices[self.slice_idx]

    def update(self, won_episode, steps_taken):
        """
        Called after every episode to update statistics.
        Returns the current rolling success rate (0.0 to 1.0).
        """
        self.total_steps += steps_taken
        
        # Update Window
        self.wins_window.append(1 if won_episode else 0)
        if len(self.wins_window) > 50:
            self.wins_window.pop(0) # Keep size at 50
            
        # Calculate Rate
        current_rate = 0.0
        if len(self.wins_window) > 0:
            current_rate = sum(self.wins_window) / len(self.wins_window)
            
        # Track 'High Score' for this slice
        if current_rate > self.best_rate_current_slice:
            self.best_rate_current_slice = current_rate
            
        return current_rate

    def should_promote(self):
        """
        The Gatekeeper: Returns True if agent has mastered the slice.
        Criteria: At least 20 episodes played AND >70% win rate. (Faster training .... :))
        """
        if len(self.wins_window) < 20: 
            return False 
        
        current_rate = sum(self.wins_window) / len(self.wins_window)
        return current_rate >= 0.70

    def advance_slice(self):
        """
        Moves the index to the next slice and resets slice-specific metrics.
        Returns True if advanced, False if game is finished.
        """
        if self.slice_idx < len(self.slices) - 1:
            self.slice_idx += 1
            new_slice = self.slices[self.slice_idx]
            
            # Reset Metrics for the new challenge
            self.wins_window = []
            self.best_rate_current_slice = 0.0
            
            # Save immediately so progress isn't lost
            self.save_state()
            
            print(f"[Curriculum] Promoted to Slice {new_slice['id']}: {new_slice['description']}")
            return True
        else:
            print("[Curriculum] Level Complete! No more slices.")
            return False

    def save_state(self):
        """Saves current progress to JSON."""
        data = {
            "slice_idx": self.slice_idx,
            "total_steps": self.total_steps
        }
        with open(META_FILE, 'w') as f:
            json.dump(data, f)

    def load_state(self):
        """Loads progress from JSON."""
        try:
            with open(META_FILE, 'r') as f:
                data = json.load(f)
                self.slice_idx = data.get("slice_idx", 0)
                self.total_steps = data.get("total_steps", 0)
                
            curr = self.slices[self.slice_idx]
            print(f"[Curriculum] Resumed at Slice {curr['id']} ({curr['description']})")
        except Exception as e:
            print(f"[Curriculum] Error loading save file: {e}. Starting fresh.")
