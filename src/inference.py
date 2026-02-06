import gymnasium as gym
import torch
import numpy as np
from src.networks import CarPilotNet
from torchvision import transforms
from collections import deque

# 1. 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 记得替换为你训练出的回归版模型路径
MODEL_PATH = "result/20260205-183011_Regression_lr0.0001_seq5/best_model.pth" 
SEQ_LEN = 5

# 2. 预处理 (保持与训练时严格一致)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def drive():
    # 初始化环境
    # render_mode="human" 方便你实时观察 AI 的走线是否顺滑
    env = gym.make("CarRacing-v3", render_mode="human")
    
    # 初始化回归模型 (action_dim=3)
    model = CarPilotNet(seq_len=SEQ_LEN, action_dim=3).to(DEVICE)
    
    # 加载权重
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}")
        return

    print(f"正在加载专家级回归模型: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state, _ = env.reset()
    frame_queue = deque(maxlen=SEQ_LEN)

    print("AI 专家驾驶员已就位，准备开始连续控制...")
    
    with torch.no_grad():
        while True:
            # 1. 图像预处理与时序队列填充
            frame = transform(state)
            frame_queue.append(frame)

            if len(frame_queue) < SEQ_LEN:
                # 初始帧不足时，车辆静止
                state, _, _, _, _ = env.step(np.array([0, 0, 0]))
                continue

            # 构造模型输入 [1, 5, 3, 64, 64]
            input_tensor = torch.stack(list(frame_queue), dim=0).unsqueeze(0).to(DEVICE)

            # 2. 模型推理 (直接输出 Tanh 激活后的连续值 [-1, 1])
            action_out = model(input_tensor) # shape: [1, 3]
            action_np = action_out.cpu().numpy().flatten()

            # 3. 后处理：将模型输出映射到环境要求的物理含义
            # action_np[0]: 转向 (Steering)，范围 [-1, 1]，无需处理
            # action_np[1]: 油门 (Throttle)，模型输出可能带负值，需截断到 [0, 1]
            # action_np[2]: 刹车 (Brake)，同理需截断到 [0, 1]
            
            final_steer = float(action_np[0])
            final_throttle = max(0.0, float(action_np[1]))
            final_brake = max(0.0, float(action_np[2]))

            # 4. 专家级微操补偿 (可选)
            # 如果你发现 AI 还是太温柔，可以稍微放大油门系数
            # final_throttle = min(1.0, final_throttle * 1.2)

            action = np.array([final_steer, final_throttle, final_brake], dtype=np.float32)
            
            # 执行动作
            state, reward, terminated, truncated, _ = env.step(action)

            # 调试：实时查看 AI 的脚法（转向, 油门, 刹车）
            # print(f"AI 控制 -> 转向: {final_steer:.2f} | 油门: {final_throttle:.2f} | 刹车: {final_brake:.2f}")

            if terminated or truncated:
                state, _ = env.reset()
                frame_queue.clear()

if __name__ == "__main__":
    import os
    drive()
