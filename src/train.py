import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys

# 接入工程体系，仅用于获取路径和配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataloader import CarRacingDataset 
from src.networks import CarPilotNet
from src.utils import cfg

def get_args():
    parser = argparse.ArgumentParser(description="CarRacing-Regression Training")
    # 保留你原有的命令行参数，方便手动指定
    parser.add_argument("--resume", default="", type=str, help="预训练模型路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return parser.parse_args()

def train(args):
    device = torch.device(args.device)
    
    # --- 1. 结构化目录处理 (遵照你的要求: result/models 和 result/logs) ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 保持实验 ID 命名风格
    exp_id = f"{timestamp}_Regression_lr{cfg.train.get('lr', 1e-4)}_seq{cfg.train.get('seq_len', 5)}"
    
    # 路径完全自动化
    base_result = cfg.paths.get('result_dir', './result')
    log_dir = os.path.join(base_result, "logs", exp_id)
    model_dir = os.path.join(base_result, "models", exp_id)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    print(f"回归模式训练启动！日志保存至: {log_dir}")

    # 3. 加载数据集 (路径从 cfg 读取)
    try:
        # 这里会去 config.yaml 找 csv_path 和 frame_dir
        train_set = CarRacingDataset(augment=True)
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.train.get('batch_size', 64), 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True if args.device == "cuda" else False
        )
        print(f"数据加载成功，总样本数 (含增强): {len(train_set)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 4. 初始化模型 (保持你原有的初始化方式)
    model = CarPilotNet(
        seq_len=cfg.train.get('seq_len', 5), 
        action_dim=3
    ).to(device)
    
    start_epoch = 0
    # --- 关键：加载你之前的权重 ---
    if args.resume and os.path.exists(args.resume):
        print(f"加载预训练模型: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device) 
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
    
    # 5. 损失函数与优化器 (完全保留你的原始选择)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.get('lr', 1e-4), weight_decay=1e-4)
    
    # 6. 学习率调度器：余弦退火
    epochs = cfg.train.get('epochs', 60)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    start_time = time.time()

    # --- 训练循环 ---
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (imgs, true_actions) in enumerate(train_loader):
            imgs = imgs.to(device)
            true_actions = true_actions.to(device) 

            # 前向传播
            pred_actions = model(imgs) 
            
            # 计算 Loss
            loss = criterion(pred_actions, true_actions)

            # 反向传播
            optimizer.zero_grad()

            if i % 100 ==0:
                print(f"DUBUG - PRED:{pred_actions[0].detach().cpu().numpy()}, TRUE:{true_actions[0].detach().cpu().numpy()}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train_step', loss.item(), step)

        # 每个 Epoch 统计
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

        # --- 7. 保存模型到指定的 models 文件夹 ---
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss
        }
        
        # 存一个最新的用于恢复
        latest_path = os.path.join(model_dir, "latest_model.pth")
        torch.save(checkpoint_data, latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(model_dir, "best_model.pth")
            torch.save(checkpoint_data, save_path)
            print(f"★ 发现更优模型 (Loss: {best_loss:.6f})，已保存。")

        scheduler.step()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"训练完成！总耗时: {total_time} | 最小 Loss: {best_loss:.6f}")
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)
