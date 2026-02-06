import gymnasium as gym
import pygame
import cv2
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 确保能找到根目录下的 configs 和 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import cfg, ActionSmoother

def collect_data():
    # 1. 获取 YAML 配置
    is_test_mode = cfg.collection['test_mode']
    save_every_n = cfg.collection['save_every_n_steps']
    target_fps = 30  # 锁定采集频率为 30Hz
    
    frame_dir = cfg.paths['frame_dir']
    csv_path = cfg.paths['csv_path']
    csv_dir = cfg.paths.get('csv_dir', os.path.dirname(csv_path))
    columns = cfg.data['columns']
    
    # 2. 初始化环境与工具
    if not is_test_mode:
        os.makedirs(frame_dir, exist_ok=True)
        
    env = gym.make(cfg.collection['env_name'], render_mode=cfg.collection['render_mode'])
    obs, info = env.reset()
    smoother = ActionSmoother()
    
    # 引入 Pygame 时钟来控制帧率
    clock = pygame.time.Clock()
    
    data_records = []
    total_steps = 0
    saved_count = 0
    running = True

    print("="*60)
    print(f"🔴 采集模式启动 | 锁定 FPS: {target_fps}")
    if is_test_mode:
        print("🛠️  当前模式：测试模式 (仅预览输出)")
    print("="*60)

    try:
        while running:
            # --- 关键修改：控制循环频率 ---
            clock.tick(target_fps) 
            
            # 3. 处理按键输入
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            target_action = np.array([0.0, 0.0, 0.0])
            if keys[pygame.K_LEFT]:  target_action[0] = -1.0
            if keys[pygame.K_RIGHT]: target_action[0] = 1.0
            if keys[pygame.K_UP]:    target_action[1] = 1.0
            if keys[pygame.K_DOWN]:  target_action[2] = 0.8
            
            # 4. 平滑处理
            smoothed_action = smoother.smooth(target_action)
            
            # 5. 环境步进
            next_obs, reward, terminated, truncated, info = env.step(smoothed_action)
            
            # 6. 采样逻辑
            if total_steps % save_every_n == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_name = f"frame_{timestamp}.jpg"
                
                if is_test_mode:
                    print(f"\r[PREVIEW] {img_name} | S:{smoothed_action[0]:.3f} T:{smoothed_action[1]:.3f}, B:{smoothed_action[2]:.3f}" ,end="")
                else:
                    img_path = os.path.join(frame_dir, img_name)
                    cv2.imwrite(img_path, cv2.cvtColor(next_obs, cv2.COLOR_RGB2BGR))
                    
                    data_records.append([
                        img_name, 
                        smoothed_action[0], 
                        smoothed_action[1], 
                        smoothed_action[2]
                    ])
                    saved_count += 1
                    print(f"\r已存: {saved_count} 帧 | 实时 FPS: {clock.get_fps():.1f}", end="")

            total_steps += 1
            if terminated or truncated:
                env.reset()
                smoother.reset()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

    finally:
        # --- 关键修改：CSV 健壮性写入 ---
        try:
            target_columns = columns
            print(f"\n\n📝 目标列: {target_columns}")
        except (AttributeError, KeyError):
            print("\n❌ 错误: YAML 配置文件中缺失 'data: columns' 定义！")
            target_columns = None

        if not is_test_mode and len(data_records) > 0 and target_columns:
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            
            # 使用 YAML 定义的表头创建 DataFrame
            df = pd.DataFrame(data_records, columns=target_columns)
            
            # 检查文件是否存在
            file_exists = os.path.isfile(csv_path)
            
            # 写入 CSV
            # 只有在文件不存在时才写入 header
            df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
            
            print(f"\n\n✅ 数据追加成功！")
            print(f"📊 字段对齐: {target_columns}")
            print(f"📂 存储路径: {csv_path}")
        
        elif is_test_mode:
            print("\n\n🏁 测试模式结束，未写入任何文件。")
            
        env.close()
        pygame.quit()

if __name__ == "__main__":
    collect_data()
