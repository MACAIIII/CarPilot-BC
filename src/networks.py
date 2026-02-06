import torch 
import torch.nn as nn
from torchvision import models
import sys
import os

# 接入工程体系
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import cfg

class CarPilotNet(nn.Module):
    def __init__(self, seq_len=None, action_dim=None):
        super(CarPilotNet, self).__init__()

        # 优先使用传入参数，否则读取 config.yaml
        self.seq_len = seq_len if seq_len else cfg.train['seq_len']
        self.action_dim = action_dim if action_dim else len(cfg.data['columns']) - 1

        # 1. 空间特征提取 (Backbone)
        # ResNet18 权重：IMAGENET1K_V1 提供了极佳的基础视觉特征提取能力
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512 
        
        # 2. 时序特征建模 (Transformer Encoder)
        # 擅长处理帧与帧之间的运动趋势
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. 回归控制器 (MLP Controller)
        self.controller = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim), 
            nn.Tanh()  # 确保输出范围严格在 [-1, 1]
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, 3, img_size, img_size]
        b, t, c, h, w = x.shape
        
        # 合并 Batch 和 Time 维度进行特征提取
        x = x.view(b * t, c, h, w)
        features = self.backbone(x) # [b*t, 512, 1, 1]
        
        # 还原时间维度：[b, t, 512]
        features = features.view(b, t, self.feature_dim)

        # Transformer 序列建模
        out = self.transformer(features) # [b, t, 512]
        
        # 取最后一个时间步作为决策依据（Many-to-One）
        last_feature = out[:, -1, :]

        # 输出动作：[转向, 油门, 刹车]
        action = self.controller(last_feature)

        return action

# ==========================================
# 测试模型
# ==========================================
if __name__ == "__main__":
    # 从配置初始化模型
    model = CarPilotNet()
    
    # 模拟输入: Batch=8, Seq=5, RGB, 64x64
    img_size = cfg.train.get('img_size', 64)
    sample_input = torch.randn(8, cfg.train['seq_len'], 3, img_size, img_size) 
    
    # 前向传播
    sample_output = model(sample_input)
    
    print(f"🚀 模型加载成功！")
    print(f"输入形状: {sample_input.shape}")
    print(f"输出动作形状: {sample_output.shape}") 
    print(f"首个动作样本: \n{sample_output[0].detach().numpy()}")
