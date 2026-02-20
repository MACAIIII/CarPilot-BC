import os
import sys
import numpy as np
import cv2
import torch
import d3rlpy
from tqdm import tqdm

# 1. 接入你原有的工程体系获取路径配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import cfg

def calculate_reward(steer, throttle, brake, img=None, prev_steer=None):
    """
    极简奖励：只解决直道高频抖动，保持原有驾驶风格
    """
    # 1. 基础奖励：保持原有行为克隆的奖励（鼓励前进）
    reward = throttle * 1.0  # 简单的油门奖励
    
    # 2. 核心：惩罚转向抖动（高频变化）
    if prev_steer is not None:
        steer_change = abs(steer - prev_steer)
        # 惩罚快速转向变化，但允许慢速调整
        if steer_change > 0.2:  # 变化超过0.1开始惩罚
            reward -= steer_change * 0.5  # 变化越大惩罚越重
    
    # 3. 防停滞（轻微）
    if throttle < 0.05 and brake < 0.05:
        reward -= 0.2
    
    # 裁剪到合理范围 [-3, 3]
    return reward


def load_dataset_for_td3bc():
    """
    手动实现 Frame Stacking，将数据转换为 [15, 64, 64] 的堆叠张量
    """
    import pandas as pd
    csv_path = os.path.join(cfg.paths['csv_dir'], "actions_cleaned.csv")
    img_dir = cfg.paths['frame_dir']
    
    data = pd.read_csv(csv_path)
    img_size = cfg.train.get('img_size', 64)
    n_frames = 5  # 对应你的 seq_len

    observations = []
    actions = []
    rewards = []
    terminals = []
    prev_steer = 0.0
    print(f"📦 正在处理 1600 帧小数据集 (手动堆叠 n_frames={n_frames})...")
    
    # 从第 n_frames 帧开始，确保前方有足够的帧进行堆叠
    for i in tqdm(range(n_frames - 1, len(data))):
        # 提取并堆叠最近的 5 帧
        stack = []
        for j in range(n_frames):
            idx = i - (n_frames - 1) + j
            img_name = data.iloc[idx]['frame_id']
            img = cv2.imread(os.path.join(img_dir, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            # d3rlpy 期望像素通道在前: [C, H, W]
            img = img.transpose(2, 0, 1) 
            stack.append(img)
        
        # 将 5 帧拼接在一起 [3*5, 64, 64] -> [15, 64, 64]
        stacked_img = np.concatenate(stack, axis=0)
        observations.append(stacked_img)
        
        # 动作取序列最后一帧
        actions.append([
            data.iloc[i]['steering'],
            data.iloc[i]['throttle'],
            data.iloc[i]['brake']
        ])
        
        steer = data.iloc[i]['steering']
        throttle = data.iloc[i]['throttle']
        brake = data.iloc[i]['brake']

                # 基础数据：当前动作
        steer = data.iloc[i]['steering']
        throttle = data.iloc[i]['throttle']

        # 辅助数据：上一帧和下一帧（离线学习可以“预知未来”）
        prev_steer = data.iloc[i-1]['steering'] if i > 0 else steer
        # 观察下一帧专家是怎么做的，如果下一帧专家没转弯，AI现在转弯就要扣分
        next_expert_steer = data.iloc[i+1]['steering'] if i < len(data)-1 else steer

        # ---------------------------------------------------------
        # 1. 基础动力奖励：鼓励有速度，但不要盲目地板油
        # ---------------------------------------------------------
        reward = throttle * 1.0

        # ---------------------------------------------------------
        # 2. 惩罚“进弯过早” (Anti-Early-Turning)
        # ---------------------------------------------------------
        # 如果当前 AI 想大打方向盘（abs(steer)大），但专家在下一帧其实是直行
        # 说明 AI 在“抢跑”进弯，必须重罚
        if abs(steer) > 0.3 and abs(next_expert_steer) < 0.1:
            reward -= 2.0 * (abs(steer) - abs(next_expert_steer))

        # ---------------------------------------------------------
        # 3. 惩罚“转向晃动” (Smoothing Penalty)
        # ---------------------------------------------------------
        # 强制要求相邻两帧转向连续。这个值越大，画龙越少。
        jitter = abs(steer - prev_steer)
        reward -= jitter * 3.0 

        # ---------------------------------------------------------
        # 4. 模拟 Gym 赛道进度奖励 (Directional Reward)
        # ---------------------------------------------------------
        # 既然我们无法 step，就用专家数据的“油门”作为正向引导
        # 专家踩油门的地方通常是赛道正确方向。
        if throttle > 0.5 and abs(steer) < 0.2:
            reward += 0.5 # 鼓励在直道加速
        
        # prev_steer = steer  # 更新上一帧转向

        rewards.append(reward)
        terminals.append(0)

    # 标记最后一个样本为结束
    terminals[-1] = 1

    # 创建 d3rlpy 数据集
    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations, dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
    )
    return dataset

def train_td3_bc():
    # 1. 加载数据
    dataset = load_dataset_for_td3bc()

    # 2. TD3+BC 算法配置
    # 由于数据只有 1600 帧，alpha 设为 2.5 或更高（如 5.0）来防止模型跑偏
    config = d3rlpy.algos.TD3PlusBCConfig(
        actor_learning_rate=3e-5,    # 极低学习率保护原有权重
        critic_learning_rate=3e-4,   
        batch_size=64,               # 小 batch 适合小数据量
        alpha=5.0,                   # 平衡 RL 和 BC 的核心参数
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(), # 归一化图像
    )

    # 创建算法实例
    td3bc = config.create(device="cuda" if torch.cuda.is_available() else "cpu")

    # 3. 结果保存路径
    model_save_dir = os.path.join(cfg.paths.get('result_dir', './result'), "td3bc_models")
    os.makedirs(model_save_dir, exist_ok=True)

    print("🚀 TD3+BC 离线训练启动...")
    # 对于 1600 帧，训练 20000 步左右通常就足够捕获 Q 值特征了
    td3bc.fit(
        dataset,
        n_steps=20000,
        n_steps_per_epoch=2000,
        save_interval=5,
        experiment_name="CarRacing_TD3BC_SmallData"
    )

    # 4. 保存模型
    final_path = os.path.join(model_save_dir, "td3bc_final.pt")
    td3bc.save_model(final_path)
    print(f"✨ 训练完成！模型已保存至: {final_path}")

if __name__ == "__main__":
    train_td3_bc()