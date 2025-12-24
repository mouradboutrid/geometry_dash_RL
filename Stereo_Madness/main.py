import torch
import keyboard
import time
import os

# MY MODULES IMPORTS
from config import *
from core.environment import GeometryDashEnv
from agents.ddqn import Agent
from agents.replay_buffer import ReplayBuffer
from curriculum.manager import CurriculumManager


class GDAgentOrchestrator:
    def __init__(self):
        # ENV
        self.env = GeometryDashEnv()
        self.env.frame_skip = 4
        self.env.frame_stack = 2

        # CURRICULUM
        self.manager = CurriculumManager()
        self.current_slice = self.manager.get_current_slice()
        self.env.set_slice(self.current_slice)

        # MEMORY
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # AGENT
        agent_config = {
            'device': DEVICE,
            'lr': LR,
            'gamma': GAMMA,
            'batch_size': BATCH_SIZE,
            'target_update': TARGET_UPDATE,
            'epsilon_start': EPSILON_START,
            'epsilon_end': EPSILON_END,
            'epsilon_decay': EPSILON_DECAY,
        }
        self.agent = Agent(INPUT_DIM, OUTPUT_DIM, agent_config, CHECKPOINT_DIR)

        # EXPERTS
        self.final_models_dir = os.path.join(CHECKPOINT_DIR, "final_models")
        self.experts_cache = self._load_experts_to_ram()

        # RELAY
        self._bridge_to_training_zone()
        
    # CURRICULUM ACCESS (ROBUST)
    def _get_all_slices(self):
        for attr in ['slices', '_slices', 'curriculum']:
            if hasattr(self.manager, attr):
                slices = getattr(self.manager, attr)
                if isinstance(slices, list) and slices:
                    break
        else:
            raise AttributeError("CurriculumManager does not expose slices")

        out = []
        for s in slices:
            out.append({
                'id': int(s['id']),
                'start': float(s['start']),
                'end': float(s['end']),
                'mode': s.get('mode', 0)
            })

        out.sort(key=lambda x: x['start'])
        return out

    # EXPERT LOADING
    def _load_experts_to_ram(self):
        cache = {}
        if not os.path.exists(self.final_models_dir):
            return cache

        print("\n[System] Loading All Expert Models into RAM...")
        for fname in os.listdir(self.final_models_dir):
            if fname.startswith("slice_") and fname.endswith("_model.pth"):
                try:
                    sid = int(fname.split("_")[1])
                    path = os.path.join(self.final_models_dir, fname)
                    cache[sid] = torch.load(path, map_location=DEVICE)
                    print(f"   -> RAM LOADED: Expert {sid}")
                except Exception as e:
                    print(f"   -> Failed loading {fname}: {e}")
        return cache

    # RELAY BRIDGE
    def _bridge_to_training_zone(self):
        sid = self.current_slice['id']

        if sid == 1:
            self._load_current_progress()
            return

        target_pct = max(0.0, self.current_slice['start'] - 0.5)

        print("\n" + "═" * 60)
        print(f" RELAY RACE ACTIVE | TARGET: {target_pct:.1f}%")
        print(f" Experts Available: {sorted(self.experts_cache.keys())}")
        print("═" * 60)

        if not self._run_relay_navigation(target_pct):
            raise RuntimeError("[Bridge] FATAL: Relay failed")

        print("[Bridge] Checkpoint marked successfully.")
        self._load_current_progress()

    def _run_relay_navigation(self, target_percent):
        slice_list = self._get_all_slices()

        print("[Relay] Curriculum map:")
        for s in slice_list:
            print(f"   Slice {s['id']}: {s['start']} → {s['end']}")

        for attempt in range(1, 16):
            obs, _ = self.env.reset()
            active_expert = None

            while True:
                # ACT
                action = self.agent.select_action(obs, is_training=False)
                obs, _, terminated, truncated, info = self.env.step(action)

                current_pos = float(info.get("percent", 0.0))

                # EXPERT SWITCH (AFTER STEP)
                correct_expert = None
                for s in slice_list:
                    if s['start'] <= current_pos < s['end']:
                        correct_expert = s['id']
                        break

                if (
                    correct_expert is not None
                    and correct_expert != active_expert
                    and correct_expert in self.experts_cache
                    and correct_expert < self.current_slice['id']
                ):
                    self.agent.online_net.load_state_dict(
                        self.experts_cache[correct_expert]
                    )
                    active_expert = correct_expert
                    print(f"[Relay] {current_pos:.1f}% → Expert {active_expert}")

                # SUCCESS
                if current_pos >= target_percent:
                    keyboard.press('w')
                    time.sleep(0.15)
                    keyboard.release('w')
                    time.sleep(1.2)
                    return True

                if terminated or truncated:
                    break

            print(
                f"[Relay] Attempt {attempt} failed at "
                f"{info.get('percent', 0.0):.1f}%"
            )

        return False

    # MODE-AWARE POLICY INITIALIZATION
    def _load_current_progress(self):
        sid = self.current_slice['id']
        path = os.path.join(CHECKPOINT_DIR, f"slice_{sid:02d}_current.pth")

        # Resume existing checkpoint
        if os.path.exists(path):
            self.agent.load(path)
            print(f"[System] Resumed Slice {sid}")
            return

        # Get all slices
        all_slices = self._get_all_slices()

        # Determine previous slice with SAME mode
        prev_same_mode = None
        for s in reversed(all_slices):
            if s["id"] < sid and s["mode"] == self.current_slice["mode"]:
                prev_same_mode = s
                break

        # Initialize network
        if prev_same_mode is None:
            print(
                f"[System] Slice {sid}: new mode {self.current_slice['mode']} → fresh policy"
            )
            self.agent.reset_network()
            self.agent.epsilon = 1.0
        else:
            pid = prev_same_mode["id"]
            print(
                f"[System] Slice {sid}: transfer from Slice {pid} (same mode)"
            )
            self.agent.online_net.load_state_dict(
                self.experts_cache[pid]
            )
            self.agent.target_net.load_state_dict(
                self.agent.online_net.state_dict()
            )
            self.agent.epsilon = 0.5

    # TRAINING LOOP
    def train(self):
        print(
            f"\n[Training] Starting Slice {self.current_slice['id']} Mastery..."
        )
        episode = 0

        try:
            while True:
                episode += 1
                obs, _ = self.env.reset()
                last_loss = 0.0
                total_reward = 0.0  # Track total reward

                while True:
                    action = self.agent.select_action(obs, is_training=True)
                    next_obs, reward, terminated, _, info = self.env.step(action)

                    self.memory.push(
                        obs, action, reward, next_obs, float(terminated)
                    )

                    loss = self.agent.learn(self.memory)
                    if loss is not None:
                        last_loss = loss

                    obs = next_obs
                    total_reward += reward  # Accumulate reward

                    if terminated:
                        break

                win_rate = self.manager.update(
                    info['percent'] >= self.current_slice['end'], 0
                )

                # Print including total reward
                print(
                    f"Ep {episode:<4} | "
                    f"Win% {win_rate*100:>5.1f}% | "
                    f"% {info['percent']:>5.1f} | "
                    f"Reward {total_reward:>7.2f} | "
                    f"Loss {last_loss:.4f}"
                )

                if self.manager.should_promote():
                    self._save_expert_final()

                    if self.manager.advance_slice():
                        self.current_slice = self.manager.get_current_slice()
                        self.env.set_slice(self.current_slice)
                        self.experts_cache = self._load_experts_to_ram()
                        self._bridge_to_training_zone()
                    else:
                        break

                if episode % 50 == 0:
                    self.agent.save(
                        filename=f"slice_{self.current_slice['id']:02d}_current.pth"
                    )

        except KeyboardInterrupt:
            self.agent.save(
                filename=f"slice_{self.current_slice['id']:02d}_current.pth"
            )

    # SAVE FINAL EXPERT
    def _save_expert_final(self):
        sid = self.current_slice['id']
        os.makedirs(self.final_models_dir, exist_ok=True)
        path = os.path.join(
            self.final_models_dir, f"slice_{sid:02d}_model.pth"
        )
        torch.save(self.agent.online_net.state_dict(), path)
        print(f"[System] Expert {sid} Saved.")

GDAgentOrchestrator().train()
