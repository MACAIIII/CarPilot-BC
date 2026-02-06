import yaml
import os
import numpy as np

class YamlConfig:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def __getattr__(self, item):
        # 允许通过 cfg.paths['data_dir'] 这种方式访问
        if item in self._config:
            return self._config[item]
        raise AttributeError(f"No such config: {item}")

# 初始化配置单例
# 自动定位项目根目录下的 yaml 文件
_current_dir = os.path.dirname(os.path.abspath(__file__))
_yaml_path = os.path.join(_current_dir, "../configs/config.yaml")
cfg = YamlConfig(_yaml_path)


class ActionSmoother:
    def __init__(self, alpha=None, action_dim=3):
        # 从 YAML 的 collection 层级读取 smooth_alpha
        self.alpha = alpha if alpha is not None else cfg.collection['smooth_alpha']
        self.current_action = np.zeros(action_dim)
        print(f"DEBUG: 平滑器启动 | 配置文件 Alpha: {self.alpha}")

    def smooth(self, target_action):
        self.current_action = self.alpha * target_action + (1 - self.alpha) * self.current_action
        return self.current_action

    def reset(self):
        self.current_action = np.zeros(3)
