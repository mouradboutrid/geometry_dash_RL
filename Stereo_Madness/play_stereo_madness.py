import torch
import os
from config import *
from core.environment import GeometryDashEnv
from agents.ddqn import Agent
from curriculum.manager import CurriculumManager


class StereoMadnessPlayer:
    def __init__(self):
        self.env = GeometryDashEnv()
        self.env.frame_skip = 4
        self.env.frame_stack = 2

        self.manager = CurriculumManager()
        self.slice_list = self._get_all_slices()

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
        self.agent.online_net.eval()

        self.final_models_dir = os.path.join(CHECKPOINT_DIR, "final_models")
        self.models = self._load_all_models()

        if not self.models:
            raise RuntimeError("No expert models found!")


    def _get_all_slices(self):
        """Get all curriculum slices in sorted order."""
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

    def _load_all_models(self):
        models = {}
        if not os.path.exists(self.final_models_dir):
            return models
        for fname in sorted(os.listdir(self.final_models_dir)):
            if fname.startswith("slice_") and fname.endswith("_model.pth"):
                try:
                    sid = int(fname.split("_")[1])
                    path = os.path.join(self.final_models_dir, fname)
                    models[sid] = torch.load(path, map_location=DEVICE)
                except Exception:
                    pass
        return models

    def _get_slice_at_percent(self, percent):
        for s in self.slice_list:
            if s['start'] <= percent < s['end']:
                return s
        return None

    def play(self):
        while True:
            obs, _ = self.env.reset()
            active_expert = None
            try:
                while True:
                    with torch.no_grad():
                        action = self.agent.select_action(obs, is_training=False)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    current_pos = float(info.get("percent", 0.0))

                    correct_expert = None
                    for s in self.slice_list:
                        if s['start'] <= current_pos < s['end']:
                            correct_expert = s['id']
                            break

                    if (
                        correct_expert is not None
                        and correct_expert != active_expert
                        and correct_expert in self.models
                    ):
                        self.agent.online_net.load_state_dict(self.models[correct_expert])
                        self.agent.online_net.eval()
                        active_expert = correct_expert

                    if current_pos >= 100.0:
                        return

                    if terminated or truncated:
                        break

            except KeyboardInterrupt:
                return





player = StereoMadnessPlayer()
player.play()

