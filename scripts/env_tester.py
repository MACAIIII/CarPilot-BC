import gymnasium as gym
import pygame
import time
import numpy as np
import sys
import os

# 确保脚本能找到项目根目录下的 src 和 configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import cfg, ActionSmoother

def run_env_test():
    """
    环境综合测试：验证环境启动、FPS 性能及平滑控制逻辑。
    """
    print("="*50)
    print("🚀 开始环境综合测试 (Environment & Control Test)")
    print("="*50)

    # 1. 从 YAML 加载配置
    env_name = cfg.collection['env_name']
    render_mode = cfg.collection['render_mode']
    alpha = cfg.collection['smooth_alpha']

    try:
        # 2. 初始化环境
        env = gym.make(env_name, render_mode=render_mode)
        env.reset()
        
        # 3. 初始化平滑器 (自动读取 YAML 中的 alpha)
        smoother = ActionSmoother()
        
        print(f"✅ 环境加载成功: {env_name}")
        print(f"✅ 平滑器初始化成功 (Alpha: {alpha})")
        print("\n[控制说明]:")
        print(" - 使用键盘方向键模拟输入（程序将自动平滑化）")
        print(" - 观察终端输出的动作值变化")
        print(" - 按 'ESC' 键退出测试")
        print("-" * 30)

        # 4. 测试循环
        steps = 0
        start_time = time.time()
        running = True
        
        while running:
            # 获取键盘状态 (模拟 collect_data 中的逻辑)
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            # 基础动作映射 [steering, throttle, brake]
            target_action = np.array([0.0, 0.0, 0.0])
            if keys[pygame.K_LEFT]:  target_action[0] = -1.0
            if keys[pygame.K_RIGHT]: target_action[0] = 1.0
            if keys[pygame.K_UP]:    target_action[1] = 0.8
            if keys[pygame.K_DOWN]:  target_action[2] = 0.8  # 刹车
            
            # 应用平滑逻辑
            smoothed_action = smoother.smooth(target_action)
            
            # 执行环境步进
            obs, reward, terminated, truncated, info = env.step(smoothed_action)
            env.render()
            
            # 实时打印动作值变化 (使用 \r 实现覆盖输出)
            print(f"\rStep: {steps:03d} | 原始输入: {target_action} | 平滑输出: {smoothed_action.round(3)}", end="")
            
            steps += 1
            if terminated or truncated:
                env.reset()
                smoother.reset()

            # 退出机制
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

        # 5. 性能结算
        total_time = time.time() - start_time
        fps = steps / total_time
        print(f"\n\n" + "="*50)
        print(f"📊 测试总结:")
        print(f" - 总步数: {steps}")
        print(f" - 平均帧率 (FPS): {fps:.2f}")
        
        if fps < 40:
            print("⚠️ 警告: 帧率偏低，采集数据可能存在不连续性，请检查硬件占用。")
        else:
            print("🚀 性能达标：环境稳定且响应迅速。")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    finally:
        if 'env' in locals():
            env.close()
            pygame.quit()

if __name__ == "__main__":
    run_env_test()
