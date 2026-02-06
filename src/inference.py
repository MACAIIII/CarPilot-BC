import gymnasium as gym
import torch
import numpy as np
import os
import sys
from torchvision import transforms
from collections import deque

# 接入工程体系
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.networks import CarPilotNet
from src.utils import cfg

def inference(model_id=None):
    """
    :param model_id: 具体的实验 ID（如 20260206-154500...），若不指定则搜索最新的 best_model
    """
    # 1. 初始化设备与配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = cfg.train.get('seq_len', 5)
    img_size = cfg.train.get('img_size', 64)
    
    # --- 自动定位模型路径 ---
    base_model_dir = os.path.join(cfg.paths.get('result_dir', './result'), "models")
    
    if model_id:
        model_path = os.path.join(model_id, "best_model.pth")
    else:
        # 自动寻找 models 文件夹下最近一次生成的 best_model
        try:
            all_exps = sorted(os.listdir(base_model_dir))
            model_path = os.path.join(base_model_dir, all_exps[-1], "best_model.pth")
        except Exception:
            print("❌ 错误：在 result/models 下找不到任何训练好的模型！")
            return

    # 2. 初始化环境 (render_mode="human" 用于观察)
    env = gym.make("CarRacing-v3", render_mode="human")
    
    # 3. 加载模型
    # 模型会根据 cfg 自动配置架构
    model = CarPilotNet().to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return

    print(f"🚗 正在唤醒 AI 驾驶员...")
    print(f"📄 使用权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. 预处理定义 (必须与训练时完全一致)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    state, _ = env.reset()
    frame_queue = deque(maxlen=seq_len)

    print("🏁 比赛开始！AI 已接管控制台。")
    
    with torch.no_grad():
        try:
            while True:
                # A. 图像预处理
                frame = transform(state)
                frame_queue.append(frame)

                if len(frame_queue) < seq_len:
                    # 队列未满时，给一个空动作 [转向, 油门, 刹车]
                    state, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))
                    continue

                # B. 构造输入 Tensor [1, seq_len, 3, H, W]
                input_tensor = torch.stack(list(frame_queue), dim=0).unsqueeze(0).to(device)

                # C. 模型推理
                action_out = model(input_tensor) 
                action_np = action_out.cpu().numpy().flatten()

                # D. 后处理与安全截断
                # Steering: [-1, 1], Throttle: [0, 1], Brake: [0, 1]
                final_steer = float(action_np[0])
                final_throttle = np.clip(float(action_np[1]), 0.0, 1.0)
                final_brake = np.clip(float(action_np[2]), 0.0, 1.0)

                # --- 策略：防止 AI 止步不前 ---
                # 如果检测到油门太小且没踩刹车，可以给个基础怠速
                if final_throttle < 0.05 and final_brake < 0.1:
                    final_throttle = 0.1

                action = np.array([final_steer, final_throttle, final_brake], dtype=np.float32)
                
                # E. 环境步进
                state, reward, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    state, _ = env.reset()
                    frame_queue.clear()
                    print("🔄 赛道重置...")
                    
        except KeyboardInterrupt:
            print("\n🛑 手动停止推理。")
        finally:
            env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="指定模型 ID (例如 20260206-154500...)")
    args = parser.parse_args()
    
    inference(model_id=args.id)
