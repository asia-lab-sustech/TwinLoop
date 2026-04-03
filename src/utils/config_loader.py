import yaml
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class StateSpaceConfig:
    dimensions: int
    features: List[str]
    normalization: Dict[str, List[float]]
    fixed_features: Optional[List[str]] = None
    dynamic_features: Optional[List[str]] = None
    construction: Optional[Dict[str, Any]] = None


@dataclass
class ActionSpaceConfig:
    dimensions: int
    mapping: Dict[int, str]
    base_actions: Optional[List[str]] = None
    dynamic_actions: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    id: str
    type: str
    state_space: StateSpaceConfig
    action_space: ActionSpaceConfig
    reward: Dict[str, float]
    connectivity: Optional[Dict[str, Any]] = None
    dqn: Optional[Dict[str, Any]] = None
    network: Optional[Dict[str, Any]] = None


class ConfigLoader:
    """
    配置加载器
    """

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # 从项目根目录查找configs文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            self.config_dir = os.path.join(project_root, 'configs')
        else:
            self.config_dir = config_dir

        self._cache = {}
        print(f"配置目录: {self.config_dir}")

    def load_agent_config(self, agent_id: str) -> AgentConfig:
        """加载指定智能体的配置"""
        filename = f"agents/{agent_id}.yaml"
        raw_config = self._load_yaml(filename)

        # 转换为类型安全对象
        agent_config = raw_config["agent"]

        # 处理可选字段
        state_space_data = agent_config["state_space"]
        action_space_data = agent_config["action_space"]

        return AgentConfig(
            id=agent_config["id"],
            type=agent_config["type"],
            state_space=StateSpaceConfig(
                dimensions=state_space_data.get("dimensions", 0),
                features=state_space_data.get("features", []),
                normalization=state_space_data.get("normalization", {}),
                fixed_features=state_space_data.get("fixed_features"),
                dynamic_features=state_space_data.get("dynamic_features"),
                construction=state_space_data.get("construction")
            ),
            action_space=ActionSpaceConfig(
                dimensions=action_space_data.get("dimensions", 0),
                mapping=action_space_data.get("mapping", {}),
                base_actions=action_space_data.get("base_actions"),
                dynamic_actions=action_space_data.get("dynamic_actions"),
                constraints=action_space_data.get("constraints")
            ),
            reward=agent_config.get("reward", {}),
            connectivity=agent_config.get("connectivity"),
            dqn=agent_config.get("dqn"),
            network=agent_config.get("network")
        )

    def load_all_agent_configs(self) -> List[AgentConfig]:
        """加载所有智能体配置"""
        agent_configs = []
        agents_dir = os.path.join(self.config_dir, "agents")

        if not os.path.exists(agents_dir):
            print(f"警告: 智能体配置目录不存在: {agents_dir}")
            return agent_configs

        for filename in os.listdir(agents_dir):
            if filename.endswith(".yaml") and filename.startswith("agent_"):
                agent_id = filename.replace(".yaml", "")
                try:
                    agent_config = self.load_agent_config(agent_id)
                    agent_configs.append(agent_config)
                except Exception as e:
                    print(f"警告: 加载智能体配置 {filename} 失败: {e}")

        return agent_configs

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载YAML文件并缓存"""
        if filename in self._cache:
            return self._cache[filename]

        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ValueError(f"配置文件为空或格式错误: {filepath}")

            self._cache[filename] = config
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"YAML解析错误 in {filepath}: {e}")
        except Exception as e:
            raise ValueError(f"读取配置文件错误 {filepath}: {e}")


# 全局配置加载器实例
global_config_loader = ConfigLoader()


def get_config_loader() -> ConfigLoader:
    """获取全局配置加载器实例"""
    return global_config_loader