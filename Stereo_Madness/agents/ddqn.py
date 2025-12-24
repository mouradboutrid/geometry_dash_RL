import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature Extractor (Shared)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Value Stream (V)
        # Estimates: "How good is this state?" (regardless of action)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (A)
        # Estimates: "How much better is Action X than the average action?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        
        # Combine: Q = V + (A - mean(A))
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals

class Agent:
    def __init__(self, input_dim, output_dim, config, checkpoint_dir):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dim = output_dim
        self.checkpoint_dir = checkpoint_dir
        
        # Networks
        self.online_net = DuelingDQN(input_dim, output_dim).to(self.device)
        self.target_net = DuelingDQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config['lr'])
        self.loss_fn = nn.MSELoss()
        
        # Exploration
        self.epsilon = config['epsilon_start']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.steps_done = 0

    def select_action(self, state, is_training=True):
        """Epsilon-Greedy Action Selection"""
        if is_training:
            # Decay epsilon
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            
            if random.random() < self.epsilon:
                return random.randrange(self.output_dim)

        # Greedy Action (Exploitation)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax().item()

    def learn(self, memory):
        """Double DQN Update Step"""
        if len(memory) < self.config['batch_size']:
            return None # Not enough samples yet

        state, action, reward, next_state, done = memory.sample(self.config['batch_size'])
        
        # Convert to Tensor
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Current Q(s, a)
        curr_q = self.online_net(state).gather(1, action)
        
        # Max Action from Online Net (Double DQN trick)
        next_actions = self.online_net(next_state).argmax(1, keepdim=True)
        
        # Q-Value from Target Net using that action
        next_q = self.target_net(next_state).gather(1, next_actions)
        
        # Bellman Equation
        target_q = reward + (1 - done) * self.config['gamma'] * next_q
        
        # Optimize
        loss = self.loss_fn(curr_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Target Network periodically
        if self.steps_done % self.config['target_update'] == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return loss.item()

    def save(self, filename="best_model.pth"):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.online_net.state_dict(), path)
        print(f"[Agent] Saved model to {path}")

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.online_net.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.online_net.load_state_dict(checkpoint) # for backwards compatibility
            self.target_net.load_state_dict(self.online_net.state_dict())
            print(f"[Agent] Loaded weights from {path}")
        else:
            print(f"[Agent] Warning: No model found at {path}")
    def reset_network(self):
        """
        Re-initialize the online and target networks from scratch.
        Used when starting a new physics mode (e.g., first Ship slice).
        """
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
        self.online_net.apply(init_weights)
        self.target_net.load_state_dict(self.online_net.state_dict())
