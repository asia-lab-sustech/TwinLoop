from typing import Dict, Any, Type, Union
from .base_agent import BaseAgent
from .dqn_agent import DQNAgent

class AgentFactory:
    _agent_registry = {
        'dqn': DQNAgent,
    }

    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: Type[BaseAgent]):
        cls._agent_registry[agent_type] = agent_class

    @classmethod
    def create_agent(cls, agent_id: str, config: Union[Dict[str, Any], Any]) -> BaseAgent:
        if hasattr(config, 'type'):
            agent_type = config.type.lower()
            config_dict = cls._agent_config_to_dict(config)
        else:
            agent_type = config.get('type', 'dqn').lower()
            config_dict = config

        if agent_type not in cls._agent_registry:
            raise ValueError(f"Unknown type: {agent_type}")

        agent_class = cls._agent_registry[agent_type]
        return agent_class(agent_id, config_dict)

    @classmethod
    def _agent_config_to_dict(cls, agent_config) -> Dict[str, Any]:
        config_dict = {
            'id': agent_config.id,
            'type': agent_config.type,
            'state_space': {
                'dimensions': agent_config.state_space.dimensions,
                'features': agent_config.state_space.features,
            },
            'action_space': {
                'dimensions': agent_config.action_space.dimensions,
                'mapping': agent_config.action_space.mapping,
            },
            'reward': agent_config.reward,
            'dqn': agent_config.dqn,
            'network': agent_config.network
        }
        return config_dict