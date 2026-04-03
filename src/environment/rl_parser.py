import math
import re
import numpy as np
from typing import Dict, List, Tuple

def _extract_server_num(name: str) -> int:
    m = re.search(r'\d+', name)
    return int(m.group()) if m else 0

class RLParser:
    def __init__(self):
        self.MAX_TASK_SIZE   = 5e7
        self.MAX_TASK_CYCLES = 5e9

        self.MAX_LOCAL_PROC  = 8e8
        self.MAX_SERVER_PROC = 3.6e9
        self.MAX_LOAD_CYCLES_LOCAL = 1e9
        self.MAX_LOAD_CYCLES = 1e9
        self.MAX_RATE_BPS    = 8e6
        self.MAX_SERVERS     = 16

    def parse_state_message(self, data: Dict) -> Tuple[np.ndarray, List[str], Dict]:
        """
        将来自环境的原始物理字典转换为 DQN 所需的归一化张量。
        """
        servers   = data.get("servers", [])
        task_info = data.get("task_info", {})
        car_meta  = data.get("car", {})

        state_features = []

        # ================================================================
        # Part 1: car & task features 
        # ================================================================
        # 1. local processing rate
        local_proc = float(car_meta.get("processing_rate", 0.0))
        state_features.append(local_proc / self.MAX_LOCAL_PROC)

        # 2. local load (cycles)
        local_load = float(car_meta.get("local_load", 0.0))
        state_features.append(local_load / self.MAX_LOAD_CYCLES_LOCAL)

        # 3. task input size (bits)
        input_size_bits = float(task_info.get("input_size", 0.0))
        state_features.append(input_size_bits / self.MAX_TASK_SIZE)

        # 4. task required cycles
        cycles = float(task_info.get("demand", 0.0))
        state_features.append(cycles / self.MAX_TASK_CYCLES)

        # ================================================================
        # Part 2: servers 
        # ================================================================
        sorted_servers = sorted(servers, key=lambda x: _extract_server_num(x.get("name", "")))

        for i, server in enumerate(sorted_servers):
            if i >= self.MAX_SERVERS:
                break
                
            # server load
            current_load = float(server.get("current_load", 0.0))
            state_features.append(current_load / self.MAX_LOAD_CYCLES)
            
            # server processing rate
            proc = float(server.get("processing_rate", 0.0))
            state_features.append(proc / self.MAX_SERVER_PROC)
            
            # uplink rate
            uplink_bps = float(server.get("uplink_rate", 0.0))
            state_features.append(uplink_bps / self.MAX_RATE_BPS)

        # Padding for missing servers
        pad_length = self.MAX_SERVERS - len(sorted_servers)
        for _ in range(pad_length):
            state_features.extend([0.0, 0.0, 0.0])

        state_vector = np.array(state_features, dtype=np.float32)

        # ================================================================
        # Part 3: action space
        # ================================================================
        valid_actions = ["0"]
        for i in range(len(sorted_servers)):
            valid_actions.append(str(i + 1))

        return state_vector, valid_actions, task_info

    def parse_reward_message(self, data: Dict) -> float:
        latency = float(data.get("latency", 0.0))
        return -latency