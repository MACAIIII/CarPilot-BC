import yaml
import os
import numpy as np

class YamlConfig:
    def __init__(self):
        # 强制使用绝对路径锁定项目根目录
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        config_path = os.path.join(project_root, "configs", "config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ 找不到配置文件: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            # 直接存储为原生 Python 字典
            self._config = yaml.safe_load(f)

    # 通过 property 暴露各模块字典
    @property
    def data(self): return self._config.get('data', {})
    
    @property
    def paths(self): return self._config.get('paths', {})
    
    @property
    def collection(self): return self._config.get('collection', {})
    
    @property
    def train(self): return self._config.get('train', {})

# 实例化
cfg = YamlConfig()

class ActionSmoother:
    def __init__(self, alpha=None, action_dim=3):
        # 使用原生字典取值：cfg.collection['xxx']
        self.alpha = alpha if alpha is not None else cfg.collection['smooth_alpha']
        self.current_action = np.zeros(action_dim)

    def smooth(self, target_action):
        self.current_action = self.alpha * target_action + (1 - self.alpha) * self.current_action
        return self.current_action

    def reset(self):
        self.current_action = np.zeros(3)
