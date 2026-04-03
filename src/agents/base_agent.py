from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import threading

class BaseAgent(ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.lock = threading.RLock()

        self.state_config  = config.get('state_space', {})
        self.action_config = config.get('action_space', {})
        self.reward_config = config.get('reward', {})

        self.pending_tasks: Dict[Any, Dict] = {}

        self.training      = True
        self.episode_count = 0
        self.step_count    = 0

    @abstractmethod
    def build_network(self, state_dim: int, action_dim: int) -> nn.Module:
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: List[str],
                      exploration: bool = True, task_id: Any = -1) -> Tuple[int, str]:
        pass

    @abstractmethod
    def learn_from_experience(self, state: np.ndarray, action: int,
                              reward: float, next_state: np.ndarray,
                              done: bool, valid_actions: List[str]):
        pass

    @abstractmethod
    def get_model_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def set_model_weights(self, weights: Dict[str, torch.Tensor], sync_type: str = "sync"):
        pass

    def store_pending_task(self, task_id: Any, state: np.ndarray,
                           action: int, valid_actions: List[str]):
        if task_id != -1:
            self.pending_tasks[task_id] = {
                'state': state,
                'action': action,
                'valid_actions': valid_actions
            }

    def update_with_reward(self, reward: float,
                           next_state: np.ndarray = None,
                           done: bool = False, task_id: Any = -1):
        with self.lock:
            if task_id not in self.pending_tasks:
                print(f"[WARN] task_id={task_id} not in pending_tasks")
                return

            task_data       = self.pending_tasks.pop(task_id)
            last_state      = task_data['state']
            last_action     = task_data['action']
            last_valid_acts = task_data['valid_actions']

            next_state_for_calc = (next_state if next_state is not None
                                   else last_state)

            self.learn_from_experience(
                state=last_state,
                action=last_action,
                reward=reward,
                next_state=next_state_for_calc,
                done=done,
                valid_actions=last_valid_acts
            )
            self.step_count += 1

    def preprocess_state(self, state: np.ndarray,
                         valid_actions: List[str]) -> np.ndarray:
        return state.astype(np.float32)

    def validate_action(self, action_idx: int, action_name: str,
                        valid_actions: List[str]) -> bool:
        return action_name in valid_actions

    def get_agent_info(self) -> Dict[str, Any]:
        return {
            'agent_id':      self.agent_id,
            'type':          self.config.get('type', 'unknown'),
            'step_count':    self.step_count,
            'pending_tasks': len(self.pending_tasks)
        }