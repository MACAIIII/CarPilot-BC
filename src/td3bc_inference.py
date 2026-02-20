import gymnasium as gym
import torch
import numpy as np
import os
import sys
import cv2
from collections import deque
import d3rlpy

# 1. 接入工程体系
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import cfg

def inference_td3bc(model_path=None):
    # --- A. 初始化配置 ---
    # 显式指定设备编号修复 d3rlpy 的解析错误
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    
    seq_len = 5  
    img_size = cfg.train.get('img_size', 64)
    
    # --- B. 环境准备 ---
    env = gym.make("CarRacing-v3", render_mode="human")
    
    # --- C. 构造算法实例并初始化 ---
    config = d3rlpy.algos.TD3PlusBCConfig(
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    )
    # 创建算法对象
    algo = config.create(device=device)

    # 建立模型结构：输入 15通道(5帧*3)，64x64
    observation_shape = (seq_len * 3, img_size, img_size) 
    action_size = 3
    algo.create_impl(observation_shape, action_size)

    # --- D. 加载权重 ---
    if model_path is None:
        model_dir = os.path.join(cfg.paths.get('result_dir', './result'), "td3bc_models")
        model_path = os.path.join(model_dir, "td3bc_final.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return

    print(f"📦 正在加载权重: {model_path} 到设备: {device}")
    algo.load_model(model_path)

    # --- E. 推理循环 ---
    obs, info = env.reset()
    frame_queue = deque(maxlen=seq_len)

    print("🚀 TD3+BC 推理启动！")
    
    try:
        while True:
            # 1. 预处理图像：Gym 输出 RGB
            img = cv2.resize(obs, (img_size, img_size))
            img = img.transpose(2, 0, 1) # [C, H, W]
            
            frame_queue.append(img)

            if len(frame_queue) == seq_len:
                # 拼接成 [15, 64, 64]
                stacked_obs = np.concatenate(list(frame_queue), axis=0)
                
                # predict 期望 [Batch, C, H, W]
                # d3rlpy 会自动处理归一化
                action = algo.predict(np.expand_dims(stacked_obs, axis=0))[0]

                # 动作处理
                steer = float(action[0])
                throttle = np.clip(float(action[1]), 0.0, 1.0)
                brake = np.clip(float(action[2]), 0.0, 1.0)

                # 基础逻辑
                if throttle < 0.05 and brake < 0.1:
                    throttle = 0.1
                
                real_action = np.array([steer, throttle, brake], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(real_action)
            else:
                obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))

            if terminated or truncated:
                obs, info = env.reset()
                frame_queue.clear()

    except KeyboardInterrupt:
        print("\n🛑 停止推理。")
    finally:
        env.close()

if __name__ == "__main__":
    inference_td3bc()