import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataloader import CarRacingDataset
from src.networks import CarPilotNet
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="CarRacing-Regression Training")
    
    parser.add_argument("--resume", default="", type=str, help="预训练模型路径")
    # --- 路径配置 ---
    # 建议此处路径指向你的专家数据集
    parser.add_argument("--csv-path", default="./data/actions.csv", type=str, help="标注文件CSV的路径")
    parser.add_argument("--img-dir", default="./data/frames/", type=str, help="存放图片帧的文件夹路径")
    parser.add_argument("--output-dir", default="./result", type=str, help="保存日志和模型的根目录")
    
    # --- 训练参数 ---
    parser.add_argument("--batch-size", default=64, type=int, help="批次大小")
    parser.add_argument("--epochs", default=60, type=int, help="训练轮数")
    parser.add_argument("--lr", default=1e-4, type=float, help="初始学习率")
    parser.add_argument("--seq-len", default=5, type=int, help="时序长度")
    
    # --- 设备 ---
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    
    return parser.parse_args()

def train(args):
    device = torch.device(args.device)
    
    # 1. 初始化 TensorBoard 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{timestamp}_Regression_lr{args.lr}_seq{args.seq_len}"
    log_dir = os.path.join(args.output_dir, experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"回归模式训练启动！日志保存至: {log_dir}")

    # 2. 加载数据集 (回归版 dataset 返回 imgs 和 action 向量)
    try:
        train_set = CarRacingDataset(args.csv_path, args.img_dir, seq_len=args.seq_len, augment=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"数据加载成功，总样本数 (含增强): {len(train_set)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 3. 初始化模型
    model = CarPilotNet(seq_len=args.seq_len, action_dim=3).to(device)
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"加载预训练模型: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device) 
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # 4. 损失函数与优化器
    # 回归任务首选 SmoothL1Loss (Huber Loss)，它比 MSE 对离散噪音更鲁棒
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 5. 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float('inf')
    start_time = time.time()

    # --- 训练循环 ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (imgs, true_actions) in enumerate(train_loader):
            imgs = imgs.to(device)
            true_actions = true_actions.to(device) # [batch, 3]

            # 前向传播
            pred_actions = model(imgs) # [batch, 3]
            
            # 计算 Loss (直接计算两个向量之间的误差)
            loss = criterion(pred_actions, true_actions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train_step', loss.item(), step)

        # 每个 Epoch 结束后的逻辑
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

        # 6. 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss
            }
            save_path = os.path.join(log_dir, "best_model.pth")
            torch.save(checkpoint, save_path)
            print(f"★ 发现更优模型 (Loss: {best_loss:.6f})，已保存。")

        scheduler.step()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"训练完成！总耗时: {total_time} | 最小 Loss: {best_loss:.6f}")
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)
