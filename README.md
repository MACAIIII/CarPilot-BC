# CarPilot-BC

基于行为克隆（Behavioral Cloning）的 CarRacing-v3 自动驾驶实践 demo。

**项目说明**：这是一个基于 PyTorch 的入门级深度学习项目，旨在通过简单的行为克隆（模仿学习）方法，让模型学习人类在 Gymnasium 赛车环境中的控制决策。本项目主要用于跑通“数据采集-模型训练-闭环部署”的完整流程，是视觉控制任务的一个基础练习，同时也是为了链接RL和VLA完成的基础训练。

---

![Driving Demo](assets/dirve.gif)

## 技术实现
* **控制平滑化**：针对键盘离散输入的局限性，引入指数移动平均（EMA）进行逻辑插值，产出更适合回归模型拟合的连续动作标签。
* **特征提取**：使用简单的卷积神经网络（ResNet18）提取图像特征，并使用（TransFormers）处理时序特征，尝试捕捉车辆运动惯性。
* **数据流管线**：实现了一套轻量化的本地数据管理方案，包括实时采样存图、CSV 标注对齐及离群值清洗。

## 训练监控
![Epoch Loss](assets/Loss_train_epoch.png)
![Step Loss](assets/Loss_train_step.png)

## 文件说明
```text
CarPilot-BC/
├── data/               # 本地数据集存储
├── result/             # 训练权重及可视化评估结果
├── collect_data.py     # 带有平滑逻辑的人类驾驶数据采集脚本
├── clean_data.py       # 删除选中图片和csv区域脚本
├── check_data.py       # 数据一致性验证与抽样检查
├── model_arch.py       # 模型定义（CNN + 基础时序模块）
├── train.py            # 训练逻辑与 Loss 记录
├── drive.py            # 加载模型并在环境中进行闭环推理
└── dataset.py          # PyTorch Dataset 定义与数据预处理
