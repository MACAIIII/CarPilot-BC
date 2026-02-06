import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from torchvision import transforms

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
    def __init__(self, csv_file, img_dir, seq_len=5, augment=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.augment = augment  # 将参数存入实例
        # 严格计算可用样本数：例如100帧取5帧，可用起始索引为0到95，共96个样本
        self.base_len = len(self.data) - self.seq_len + 1
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        # 根据是否增强返回对应长度
        return self.base_len * 2 if self.augment else self.base_len
    
    def __getitem__(self, idx):
        # 决定起始位置和是否翻转
        if idx < self.base_len:
            start_idx = idx
            do_flip = False
        else:
            start_idx = idx - self.base_len
            do_flip = True

        imgs = []

        # 核心修复：基于 start_idx 进行滑动窗口读取
        for i in range(self.seq_len):
            current_idx = start_idx + i
            frame_id = os.path.join(self.img_dir, self.data.iloc[current_idx]['frame_id'])
            
            img = cv2.imread(frame_id)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {frame_id}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)

            if do_flip:
                # dims=[2] 对应 [C, H, W] 中的 W (水平翻转)
                img = torch.flip(img, dims=[2])
                
            imgs.append(img)

        # 堆叠5张图成一个张量 [5, 3, 64, 64]
        stacked_imgs = torch.stack(imgs)

        # 动作标签取这连续5张中最后一帧的动作
        actual_idx = start_idx + self.seq_len - 1
        steering = self.data.iloc[actual_idx]['steering']
        throttle = self.data.iloc[actual_idx]['throttle']
        brake = self.data.iloc[actual_idx]['brake']

        if do_flip:
            steering = -steering

        action = torch.tensor([steering, throttle, brake], dtype=torch.float32)

        return stacked_imgs, action

            
if __name__ == "__main__":
    # 增加 augment=True 测试增强逻辑
    dataset = CarRacingDataset(csv_file="data/actions.csv", img_dir="data/frames", seq_len=5, augment=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch_imgs, batch_actions in dataloader:
        print("Batch images shape:", batch_imgs.shape)  # Expected: (batch_size, seq_len, 3, 64, 64)
        print("Batch actions shape:", batch_actions.shape)  # Expected: (batch_size, 3)
        # 看看第一组数据的转向值
        print("Example Steering Action:", batch_actions[0][0].item())
        break
