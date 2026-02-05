import torch 
import torch.nn as nn
from torchvision import models

"""
模型架构：ResNet18 (空间特征) + Transformer (时序特征) -> 连续动作回归

1. 用 ResNet18 来提取深层空间特征：
   - 作为特征提取器（Backbone），将 64x64 的图像转为 512 维的特征向量。
   - 相比分类，回归任务需要 ResNet 捕捉更细微的视觉变化（如车身与赛道边缘的微小夹角）。

2. 用 Transformer 来处理时间序列特征：
   - 捕捉 5 帧连续图像间的运动趋势。
   - 擅长处理序列数据，可以捕捉专家驾驶中“提前切弯”等长距离依赖关系。

3. 控制器（Controller）：
   - 输出维度为 3：[转向, 油门, 刹车]。
   - 激活函数使用 Tanh：将输出限制在 [-1, 1] 之间，完美适配赛车控制范围。
"""

class CarPilotNet(nn.Module):
    def __init__(self, seq_len=5, action_dim=3):
        super(CarPilotNet, self).__init__()

        # 使用预训练的 ResNet18 作为特征提取器
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 去掉最后的全连接层 (fc)，保留到全局平均池化层 (avgpool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = 512 # resnet18 输出维度为 512
        
        # Transformer 配置
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8, # 多头注意力机制
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 回归控制器
        self.controller = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim), # 输出 3 个值：转向，油门，刹车
            nn.Tanh()  # 关键：输出范围在 -1 到 1 之间，适合连续数值控制
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, 3, 64, 64]
        b, t, c, h, w = x.shape
        
        # 1. 空间特征提取：将 batch 和 time 维度合并处理 [b*t, c, h, w]
        x = x.view(b * t, c, h, w)
        features = self.backbone(x) # 输出形状 [b*t, 512, 1, 1]
        
        # 2. 还原时间维度：[b, t, 512]
        features = features.view(b, t, self.feature_dim)

        # 3. 时序特征建模
        out = self.transformer(features) # [b, t, 512]
        
        # 4. 决策：取最后一个时间步的特征 [b, 512]
        last_feature = out[:, -1, :]

        # 5. 输出动作：[b, 3]
        action = self.controller(last_feature)

        return action

if __name__ == "__main__":
    # 测试模型架构
    model = CarPilotNet(seq_len=5, action_dim=3)
    # 模拟输入: 8个样本, 5帧时序, 3通道, 64x64分辨率
    sample_input = torch.randn(8, 5, 3, 64, 64) 
    sample_output = model(sample_input)
    
    print("输出形状:", sample_output.shape)  # 应该是 (8, 3)
    print("动作示例 (Steer, Throttle, Brake):")
    print(sample_output[0].detach().numpy()) # 打印第一个样本的动作连续值
