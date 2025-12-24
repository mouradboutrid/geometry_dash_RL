import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from collections import deque

from core.memory_bridge import MemoryBridge
from core.state_utils import normalize_state
from config import INPUT_DIM
from agents.expert_cube import CubeExpert
from agents.expert_ship import ShipExpert  

class GeometryDashEnv(gym.Env):
    def __init__(self):
        # configurable frame-skip and frame-stack (keep defaults 1 to preserve backward compat)
        self.frame_skip = 1
        self.frame_stack = 1

        super(GeometryDashEnv, self).__init__()
        
        # Connect to the game
        self.bridge = MemoryBridge()
        
        # Action: 0 = Release, 1 = Hold/Jump
        self.action_space = spaces.Discrete(2)
        
        # Observation: single-frame or stacked frames
        obs_shape = (INPUT_DIM * self.frame_stack,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        # State tracking
        self.current_slice = None
        self.prev_percent = 0.0
        self.prev_dist_nearest_hazard = None
        self.steps_in_episode = 0
        self.prev_action = None
        # frame buffer used when frame_stack > 1
        self._frame_buffer = deque(maxlen=self.frame_stack)

    def set_slice(self, slice_data):
        self.current_slice = slice_data

    def step(self, action, reward_context=None):
        """Apply `action` for `frame_skip` frames, accumulate rewards, and return a stacked observation.

        Returns observation (stacked), total_reward, terminated, truncated, info
        """
        self.steps_in_episode += 1
        reward_context = reward_context or {}

        total_reward = 0.0
        terminated = False
        truncated = False
        last_raw = None

        for f in range(self.frame_skip):
            # send action and advance one frame
            self.bridge.write_action(action)
            raw_state = self.bridge.read_state()
            last_raw = raw_state

            # calculate reward for this intermediate frame
            # include prev_action in context for jump penalties
            ctx = dict(reward_context)
            ctx['prev_action'] = self.prev_action
            frame_reward = self._calculate_reward(raw_state, action, raw_state.is_dead, ctx)
            total_reward += frame_reward

            # update trackers for next frame's delta computations
            self.prev_percent = raw_state.percent
            self.prev_dist_nearest_hazard = getattr(raw_state, "dist_nearest_hazard", None)

            # check termination mid-skip
            if raw_state.is_dead or (self.current_slice and raw_state.percent >= self.current_slice['end']):
                terminated = raw_state.is_dead or True
                break

        # update prev_action after the repeated frames
        self.prev_action = action

        # build stacked observation from last_raw
        if last_raw is None:
            last_raw = self.bridge.read_state()
        obs_single = normalize_state(last_raw)

        # initialize buffer on first use
        if len(self._frame_buffer) == 0:
            for _ in range(self.frame_stack):
                self._frame_buffer.append(obs_single.copy())
        else:
            self._frame_buffer.append(obs_single.copy())

        obs = np.concatenate(list(self._frame_buffer), axis=0)

        return obs, total_reward, terminated, truncated, {"percent": last_raw.percent}

    def _calculate_reward(self, state, action, is_dead, reward_context=None):
        reward_context = reward_context or {}

        # --- 0. Terminal State Rewards ---
        if is_dead:
            # Use dynamic death penalty from context if available, otherwise default
            return reward_context.get("death_penalty", -100.0)
        if self.current_slice and state.percent >= self.current_slice['end']:
            return 1000.0  # Jackpot for finishing the slice
        
        # --- 1. Mode-specific expert reward ---
        if state.player_mode == 0:  # Cube
            reward = CubeExpert.get_reward(
                state,
                action,
                prev_percent=self.prev_percent,
                prev_dist_nearest_hazard=self.prev_dist_nearest_hazard,
                reward_context=reward_context
            )
        elif state.player_mode == 1:  # Ship
            reward = ShipExpert.get_reward(
                state,
                action,
                prev_percent=self.prev_percent
                # Note: Not passing context to ship expert as it's not requested
            )
        else:
            reward = 0.0
        
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_in_episode = 0
        
        # Tell game to reset
        self.bridge.send_reset()
        time.sleep(0.1)  # wait for respawn
        
        raw_state = self.bridge.read_state()
        self.prev_percent = raw_state.percent
        self.prev_dist_nearest_hazard = getattr(raw_state, "dist_nearest_hazard", None)

        obs_single = normalize_state(raw_state)
        # reset frame buffer
        self._frame_buffer.clear()
        for _ in range(self.frame_stack):
            self._frame_buffer.append(obs_single.copy())

        obs = np.concatenate(list(self._frame_buffer), axis=0)
        return obs, {}
