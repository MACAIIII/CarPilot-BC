import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import sys
from torchvision import transforms

# 接入工程体系
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import cfg

"""
dataset的目的其实是把数据按张量Tensor的格式读取出来，方便后续的训练
在此我们需要的数据是连续5张图像以及对应的动作标签
所以在__getitem__中我们需要读取连续5张图像并堆叠成一个张量返回
为防止idx越界，我们用滑动窗口的方式读取数据

这一串数字是imagenet的数据集的统计值，用于图像归一化
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

主函数用来测试Dataset是否正确工作
batch_imgs.shape返回的张量形状（8.5.3.64.64）表示 batch_size=8, seq_len=5, channels=3, height=64, width=64
batch_actions.shape返回的张量形状（8.3）表示 batch_size=8, 每个动作有3个值（转向，油门，刹车）
"""

class CarRacingDataset(Dataset):
    def __init__(self, csv_file=None, img_dir=None, augment=True):
        """
        :param augment: 是否开启镜像增强（数据量翻倍，平衡左右转）
        """
        # 如果不传参，默认读取 config.yaml 里的清洗后数据
        self.csv_file = csv_file if csv_file else os.path.join(cfg.paths['csv_dir'], "actions_cleaned.csv")
        self.img_dir = img_dir if img_dir else cfg.paths['frame_dir']
        
        # 从配置中读取超参数
        self.seq_len = cfg.train['seq_len']
        self.img_size = cfg.train['img_size']
        self.augment = augment

        # 加载数据
        self.data = pd.read_csv(self.csv_file)
        # 基础样本数：总行数 - 序列长度 + 1
        self.base_len = len(self.data) - self.seq_len + 1
        
        # 定义转换流 (归一化使用 ImageNet 标准)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # 开启增强则数据量翻倍
        return self.base_len * 2 if self.augment else self.base_len
    
    def __getitem__(self, idx):
        # 1. 确定原始起始索引和是否翻转
        if idx < self.base_len:
            start_idx = idx
            do_flip = False
        else:
            start_idx = idx - self.base_len
            do_flip = True

        imgs = []

        # 2. 读取连续的 seq_len 张图片构成时间序列
        for i in range(self.seq_len):
            current_idx = start_idx + i
            # 注意：这里使用你在 CSV 中定义的列名 'img_path'
            img_name = self.data.iloc[current_idx][cfg.data['columns'][0]]
            full_path = os.path.join(self.img_dir, img_name)
            
            img = cv2.imread(full_path)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {full_path}")
                
            # OpenCV 是 BGR，必须转 RGB 喂给模型
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)

            if do_flip:
                # 水平翻转：img 是 [C, H, W]，dims=[2] 表示翻转宽
                img = torch.flip(img, dims=[2])
                
            imgs.append(img)

        # 堆叠图片: [seq_len, C, H, W] -> 例如 [5, 3, 64, 64]
        stacked_imgs = torch.stack(imgs)

        # 3. 获取动作标签（取序列最后一帧的动作）
        target_idx = start_idx + self.seq_len - 1
        steering = self.data.iloc[target_idx]['steering']
        throttle = self.data.iloc[target_idx]['throttle']
        brake = self.data.iloc[target_idx]['brake']

        if do_flip:
            steering = -steering # 镜像图像后，转向也要反向

        action = torch.tensor([steering, throttle, brake], dtype=torch.float32)

        return stacked_imgs, action

# ==========================================
# 测试函数
# ==========================================
if __name__ == "__main__":
    # 初始化 Dataset
    dataset = CarRacingDataset(augment=True)
    print(f"✅ 数据集加载成功! 原始有效样本: {dataset.base_len}, 增强后总量: {len(dataset)}")

    # 初始化 DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 取一个 batch 测试
    batch_imgs, batch_actions = next(iter(dataloader))
    
    print("\n--- Batch Info ---")
    print(f"Batch images shape: {batch_imgs.shape}")  # (8, 5, 3, 64, 64)
    print(f"Batch actions shape: {batch_actions.shape}") # (8, 3)
    print(f"Sample Action (Steering): {batch_actions[0][0].item():.4f}")
