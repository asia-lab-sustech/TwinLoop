# src/agents/dqn_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import threading
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from queue import Queue, Empty

from .base_agent import BaseAgent

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(42)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
class SimpleDuelingDQN(nn.Module):
    def __init__(self, state_dim=52, action_dim=21, hidden_dim=256):
        super().__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        
        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        val = self.v_head(features)
        adv = self.a_head(features)
        return val + adv - adv.mean(-1, keepdim=True)

class DQNAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.dqn_config = config.get('dqn', {})
        self.network_config = config.get('network', {})
        self.loss_callback = None
        
        self.device = DEVICE

        self.expected_state_dim = self.dqn_config.get('state_dim', 52)
        self.expected_action_dim = self.dqn_config.get('action_dim', 17)

        self.state_dim = None
        self.action_dim = None
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.scheduler = None

        self.gamma = self.dqn_config.get('gamma', 0.95)
        self.learning_rate = self.dqn_config.get('learning_rate', 0.0005)
        self.batch_size = self.dqn_config.get('batch_size', 32)
        self.min_memory_size = self.dqn_config.get('min_memory_size', 64)

        self.train_frequency = self.dqn_config.get('train_frequency', 4)
        memory_capacity = self.dqn_config.get('memory_capacity', 1000)
        self.memory = ReplayBuffer(capacity=memory_capacity)

        self.sample_counter = 0
        self.train_step = 0
        self.losses = deque(maxlen=100)

        self.tau       = self.dqn_config.get('tau_start', 1.0) 
        self.tau_min   = self.dqn_config.get('tau_end', 0.1) 
        self.tau_decay = self.dqn_config.get('tau_decay', 0.9) 

        self.sync_alpha = float(self.dqn_config.get('sync_alpha', 0.2))
        
    def _build_network(self, action_dim: int) -> nn.Module:
        net = SimpleDuelingDQN(
            state_dim=self.state_dim if self.state_dim else self.expected_state_dim,
            action_dim=action_dim
        )
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        return net.to(self.device)
    
    def build_network(self, state_dim, action_dim):
        self.state_dim = state_dim
        return self._build_network(action_dim)

    def _create_networks(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = self._build_network(action_dim)
        self.target_network = self._build_network(action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.95)

    def select_action(self, state: np.ndarray, valid_actions: List[str],
                      exploration: bool = True,
                      task_id: Any = -1) -> Tuple[int, str]:
        with self.lock:
            if self.q_network is None:
                self._create_networks(state.shape[0], len(valid_actions))

            processed = self.preprocess_state(state, valid_actions)
            state_t = torch.FloatTensor(processed).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_t).squeeze().cpu().numpy()

            q_values = np.atleast_1d(q_values).copy()

            valid_indices = []
            for i, name in enumerate(valid_actions):
                if i < len(q_values) and self.validate_action(i, name, valid_actions):
                    valid_indices.append(i)
                elif i < len(q_values):
                    q_values[i] = -1e10 

            if not valid_indices:
                action_idx, action_name = 0, valid_actions[0]
                self.store_pending_task(task_id, state, action_idx, valid_actions)
                return action_idx, action_name

            if exploration and self.training:
                valid_qs = np.array([float(q_values[idx]) for idx in valid_indices], dtype=np.float32)
                shifted = valid_qs - np.max(valid_qs)
                tau = self.tau
                exp_qs = np.exp(shifted / tau)
                probs = exp_qs / (np.sum(exp_qs) + 1e-8)
                chosen = np.random.choice(len(valid_indices), p=probs)
                action_idx = valid_indices[chosen]
            else:
                # best_q = -float('inf')
                # action_idx = valid_indices[0]
                # for idx in valid_indices:
                #     if q_values[idx] > best_q:
                #         best_q = q_values[idx]
                #         action_idx = idx
                valid_qs = np.array([float(q_values[idx]) for idx in valid_indices], dtype=np.float32)
                shifted = valid_qs - np.max(valid_qs)
                tau = self.tau_min
                exp_qs = np.exp(shifted / tau)
                probs = exp_qs / (np.sum(exp_qs) + 1e-8)
                chosen = np.random.choice(len(valid_indices), p=probs)
                action_idx = valid_indices[chosen]
            action_name = valid_actions[action_idx]
            self.store_pending_task(task_id, state, action_idx, valid_actions)
            return action_idx, action_name

    def learn_from_experience(self, state, action, reward, next_state,
                              done, valid_actions):
        if not self.training or self.q_network is None:
            return
        
        self.memory.add(
            np.array(state, dtype=np.float32), action, reward,
            np.array(next_state, dtype=np.float32), done)
        self.sample_counter += 1

        if len(self.memory) < self.min_memory_size:
            return
        if self.sample_counter % self.train_frequency != 0:
            return
        
        with self.lock:
            self._train_on_batch()

    def _train_on_batch(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_a = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_a)
            target_q = rewards + self.gamma * next_q * (~dones).float()

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 8.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.train_step += 1

        soft_tau = 0.01
        for tp, lp in zip(self.target_network.parameters(),
                          self.q_network.parameters()):
            tp.data.copy_(soft_tau * lp.data + (1 - soft_tau) * tp.data)
            
        if self.loss_callback:
            self.loss_callback(loss.item())
            
        if self.tau > self.tau_min:
            self.tau *= self.tau_decay

    def get_model_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.q_network is None:
            return None
        with self.lock:
            buffer_data = list(self.memory.buffer)  

            return {
                'q_network':      {k: v.clone().cpu() for k, v in self.q_network.state_dict().items()},
                'target_network': {k: v.clone().cpu() for k, v in self.target_network.state_dict().items()},
                'optimizer':      self.optimizer.state_dict(),
                'buffer':         buffer_data,
            }   

    def set_model_weights(self, weights: Dict[str, torch.Tensor], sync_type: str = "sync"):
        if weights is None:
            return
        with self.lock:
            if self.q_network is None:
                self._create_networks(self.expected_state_dim, self.expected_action_dim)    

            if 'q_network' in weights and isinstance(weights['q_network'], dict):
                q_w = {k: v.to(self.device) for k, v in weights['q_network'].items()}
                t_w = {k: v.to(self.device) for k, v in weights['target_network'].items()}
            else:
                q_w = {k: v.to(self.device) for k, v in weights.items()}
                t_w = q_w   

            self.q_network.load_state_dict(q_w)
            self.target_network.load_state_dict(t_w)    

            if 'optimizer' in weights:
                try:
                    self.optimizer.load_state_dict(weights['optimizer'])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                except Exception as e:
                    print(f"[Agent] optimizer state restore failed (shape mismatch?): {e}") 

            if 'buffer' in weights and weights['buffer']:
                    self.memory.buffer.clear()
                    for transition in weights['buffer']:
                        self.memory.buffer.append(transition)
                        
    def set_loss_callback(self, fn):
        self.loss_callback = fn
