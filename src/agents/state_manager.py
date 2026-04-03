import numpy as np
from typing import Dict, List, Tuple, Optional


class StateManager:
    
    def __init__(self, rl_parser):
        self.rl_parser = rl_parser

    def build_state_from_veins(self, veins_data):
        return self.rl_parser.parse_state_message(veins_data)