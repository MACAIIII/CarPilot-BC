import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 接入工程体系
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import cfg

def compare_distributions():
    # 1. 获取路径
    csv_dir = cfg.paths['csv_dir']
    raw_path = os.path.join(csv_dir, "actions.csv")
    cleaned_path = os.path.join(csv_dir, "actions_cleaned.csv")
    assets_dir = cfg.paths['assets_dir']
    # 检查文件是否存在
    if not os.path.exists(raw_path) or not os.path.exists(cleaned_path):
        print("❌ 缺失 CSV 文件，请确保 actions.csv 和 actions_cleaned.csv 都在数据目录下。")
        return

    # 2. 读取数据
    df_raw = pd.read_csv(raw_path)
    df_cleaned = pd.read_csv(cleaned_path)
    
    print(f"📊 原始数据量: {len(df_raw)} 帧")
    print(f"⚖️ 清洗后数据量: {len(df_cleaned)} 帧")

    # 3. 绘图对比
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # 原始数据直方图
    sns.histplot(df_raw['steering'], bins=50, kde=True, color='gray', ax=ax1)
    ax1.set_title(f"Original Distribution\n(Total: {len(df_raw)})")
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 清洗后数据直方图
    sns.histplot(df_cleaned['steering'], bins=50, kde=True, color='dodgerblue', ax=ax2)
    ax2.set_title(f"Cleaned Distribution\n(Total: {len(df_cleaned)})")
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # 保存对比图以便远程查看
    compare_img = os.path.join(assets_dir, "distribution_compare.png")
    plt.savefig(compare_img)
    print(f"📈 对比图已保存至: {compare_img}")
    
    # 展示
    plt.show()

if __name__ == "__main__":
    compare_distributions()
