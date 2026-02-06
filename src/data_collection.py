import gymnasium as gym
import pygame
import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime

# --- 配置区 ---
SAVE_DIR = "data/frames"
CSV_PATH = "data/actions.csv"
SEQ_LENGTH = 5  # 预留：提醒我们未来需要连续5帧

# 平滑参数：alpha 越大响应越快，越小越平滑（建议 0.1~0.3）
SMOOTH_ALPHA = 0.4  # 当前目标值的权重（15%新值 + 85%旧值）
SAVE_INTERVAL = 2  # 每隔多少帧保存一次图像和数据

os.makedirs(SAVE_DIR, exist_ok=True)

class SmoothController:
    """带插值平滑的键盘控制器"""
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        # 当前平滑后的动作状态（会持续变化）
        self.current_action = np.array([0.0, 0.0, 0.0])
        
    def get_smooth_action(self, raw_action):
        """
        对原始键盘输入进行指数移动平均平滑
        raw_action: 键盘直接输入 [steer, gas, brake]
        return: 平滑后的连续动作（变化更自然）
        """
        # 指数平滑：new = alpha * target + (1-alpha) * old
        self.current_action = (
            self.alpha * np.array(raw_action) + 
            (1 - self.alpha) * self.current_action
        )
        return self.current_action.copy()
    
    def reset(self):
        """重置状态（车辆重置时调用）"""
        self.current_action = np.array([0.0, 0.0, 0.0])

def collect():
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = env.reset()
    
    controller = SmoothController(alpha=SMOOTH_ALPHA)
    data_list = []
    clock = pygame.time.Clock()
    total_frames = 0

    step_counter = 0

    running = True

    print("=" * 50)
    print("【插值平滑版本】数据采集器")
    print("控制方式：")
    print("  ← → : 转向（会持续变化，不是瞬间打满）")
    print("  ↑   : 油门（渐进加速）")
    print("  ↓   : 刹车（渐进刹车）")
    print("  ESC : 保存并退出")
    print(f"平滑系数 alpha = {SMOOTH_ALPHA}（越小越顺滑）")
    print("=" * 50)

    while running:
        # 1. 读取原始键盘输入（目标值）
        raw_action = np.array([0.0, 0.0, 0.0])  # [转向, 油门, 刹车]
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # 设置目标值（这里可以用满幅值，平滑器会处理过渡）
        if keys[pygame.K_LEFT]:  raw_action[0] = -1.0
        if keys[pygame.K_RIGHT]: raw_action[0] = 1.0
        if keys[pygame.K_UP]:    raw_action[1] = 1.0   # 改为满油门，让平滑器控制幅度
        if keys[pygame.K_DOWN]:  raw_action[2] = 0.8
        if keys[pygame.K_ESCAPE]: break

        # 2. 【关键】获取平滑后的动作（变化值）
        smooth_action = controller.get_smooth_action(raw_action)
        
        # 调试：每30帧打印一次，观察平滑效果
        if total_frames % 30 == 0 and total_frames > 0:
            print(f"[Frame {total_frames}] 原始: [{raw_action[0]:+.1f}, {raw_action[1]:.1f}, {raw_action[2]:.1f}] "
                  f"→ 平滑: [{smooth_action[0]:+.3f}, {smooth_action[1]:.3f}, {smooth_action[2]:.3f}]")

        # 3. 执行动作（送入环境的也是平滑后的值，更自然）
        obs, reward, terminated, truncated, info = env.step(smooth_action)

        step_counter +=1
        # 4. 保存图像
        if step_counter % SAVE_INTERVAL != 0:
            img_name = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            img_path = os.path.join(SAVE_DIR, img_name)
            cv2.imwrite(img_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            # 5. 【关键】记录平滑后的值（不是原始键盘输入！）
            data_list.append({
                "frame_id": img_name,
                "steering": round(smooth_action[0], 4),   # 保留4位小数，体现连续性
                "throttle": round(smooth_action[1], 4),
                "brake": round(smooth_action[2], 4),
                # 可选：也记录原始值，用于分析
                # "raw_steering": raw_action[0],
                # "raw_throttle": raw_action[1],
                # "raw_brake": raw_action[2],
            })

            total_frames += 1
            if total_frames % 100 == 0:
                avg_steer = np.mean([d["steering"] for d in data_list[-100:]])
                print(f"✓ 已采集 {total_frames} 帧 | 最近100帧平均转向: {avg_steer:+.3f}")

        # 处理回合结束
        if terminated or truncated:
            obs, info = env.reset()
            controller.reset()  # 重置平滑状态
            print("--- 车辆重置，平滑器状态清零 ---")

        clock.tick(30)

    # 6. 保存数据
    df = pd.DataFrame(data_list)
    
    # 计算一些统计信息
    print("\n" + "=" * 50)
    print("采集统计：")
    print(f"  总帧数: {len(df)}")
    print(f"  转向范围: [{df['steering'].min():+.3f}, {df['steering'].max():+.3f}]")
    print(f"  油门范围: [{df['throttle'].min():.3f}, {df['throttle'].max():.3f}]")
    print(f"  转向标准差: {df['steering'].std():.3f}（越大说明变化越丰富）")
    print("=" * 50)
    
    # 保存到CSV
    if not os.path.isfile(CSV_PATH):
        df.to_csv(CSV_PATH, index=False,header=True)  
    else:
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)

    print(f"数据已保存至: {CSV_PATH}")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    collect()