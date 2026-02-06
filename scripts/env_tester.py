import gymnasium as gym
import pygame
import numpy as np

# 平滑系数：越小越顺滑（0.1~0.3 之间调整）
SMOOTH_ALPHA = 0.5

class SmoothController:
    """带插值平滑的键盘控制器"""
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        self.current_action = np.array([0.0, 0.0, 0.0])
        
    def get_action(self, raw_action):
        """指数平滑：让动作变化更自然"""
        self.current_action = (
            self.alpha * np.array(raw_action) + 
            (1 - self.alpha) * self.current_action
        )
        return self.current_action.copy()
    
    def reset(self):
        self.current_action = np.array([0.0, 0.0, 0.0])

# 创建环境
env = gym.make("CarRacing-v3", render_mode="human")
obs, info = env.reset()

# 初始化平滑控制器
controller = SmoothController(alpha=SMOOTH_ALPHA)

print("环境初始化成功！")
print(f"【平滑模式开启】alpha = {SMOOTH_ALPHA}（越小越顺滑）")
print("控制: ← → 转向 | ↑ 油门 | ↓ 刹车")
print("特点：松开按键后方向/油门不会瞬间归零，而是缓慢回正")
print("关闭窗口退出")

clock = pygame.time.Clock()
running = True
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. 读取原始键盘输入（目标值）
    raw_action = np.array([0.0, 0.0, 0.0])
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT]:  raw_action[0] = -1.0
    if keys[pygame.K_RIGHT]: raw_action[0] = 1.0
    if keys[pygame.K_UP]:    raw_action[1] = 0.8   # 满油门，让平滑器控制
    if keys[pygame.K_DOWN]:  raw_action[2] = 0.8

    # 2. 【关键】获取平滑后的动作
    action = controller.get_action(raw_action)

    # 调试输出：每秒打印一次当前状态
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"\r转向: {action[0]:+.3f} | 油门: {action[1]:.3f} | 刹车: {action[2]:.3f}  ", 
              end="", flush=True)

    # 3. 执行平滑后的动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 4. 重置时也重置平滑器
    if terminated or truncated:
        obs, info = env.reset()
        controller.reset()
        print("\n--- 车辆重置 ---")

    clock.tick(30)

env.close()
pygame.quit()