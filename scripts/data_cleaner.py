import pandas as pd
import os
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 接入工程体系
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import cfg

def clean_data(args):
    # 从配置文件读取路径
    csv_path = cfg.paths['csv_path']
    img_dir = cfg.paths['frame_dir']
    target_col = cfg.data['columns'][0] # 通常是 'img_path'

    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df[target_col] = df[target_col].astype(str)
    initial_count = len(df)

    # ==========================================
    # 1. 区间删除逻辑 (基于文件名)
    # ==========================================
    if args.start and args.end:
        start_f = args.start if args.start.endswith(".jpg") else args.start + ".jpg"
        end_f = args.end if args.end.endswith(".jpg") else args.end + ".jpg"
        
        # 你的核心逻辑：利用时间戳文件名的可比较性
        mask = (df[target_col] >= start_f) & (df[target_col] <= end_f)
        to_delete = df[mask]
        
        if len(to_delete) > 0:
            print(f"⚠️ 正在处理区间删除: {start_f} -> {end_f} ({len(to_delete)} 帧)")
            
            # 物理删除
            del_count = 0
            for img_name in to_delete[target_col]:
                img_p = os.path.join(img_dir, img_name)
                if os.path.exists(img_p):
                    os.remove(img_p)
                    del_count += 1
            
            # 更新 DataFrame
            df = df[~mask]
            print(f"✅ 物理删除 {del_count} 张，CSV 移除 {len(to_delete)} 条")
        else:
            print(f"❓ 未找到该区间内的数据，请检查文件名。")

    # ==========================================
    # 2. 磁盘同步 (手动删除补充)
    # ==========================================
    # 防止你直接在文件夹里删了图，但没用脚本删，这里做一个兜底同步
    print("🔍 正在同步磁盘剩余文件...")
    df['exists'] = df[target_col].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
    df = df[df['exists']].drop(columns=['exists'])
    sync_deleted = initial_count - len(df) - (len(to_delete) if 'to_delete' in locals() else 0)
    if sync_deleted > 0:
        print(f"✨ 自动同步：移除了 {sync_deleted} 条在磁盘上不存在的记录")

    # ==========================================
    # 3. 模式选择：自动降采样 (可选)
    # ==========================================
    if args.mode == 'auto':
        print(f"🤖 开启自动模式：正在对直道数据进行 {args.keep_ratio} 比例降采样...")
        # 假设 steering 在配置列的第二位，或者直接按名取
        steer_col = cfg.data['columns'][1] 
        straight_mask = df[steer_col].abs() < args.threshold
        
        df_straight = df[straight_mask]
        df_curvy = df[~straight_mask]
        
        df_straight_sampled = df_straight.sample(frac=args.keep_ratio, random_state=42)
        df = pd.concat([df_curvy, df_straight_sampled]).sort_values(by=target_col)
        print(f"⚖️ 降采样完成，最终剩余: {len(df)} 帧")

    # ==========================================
    # 4. 保存与预览
    # ==========================================
    cleaned_csv = csv_path.replace(".csv", "_cleaned.csv")
    df.to_csv(cleaned_csv, index=False)
    print(f"\n🎉 处理完毕！清洗后的数据存为: {cleaned_csv}")
    
    # 快速看一眼分布
    plt.figure(figsize=(8, 4))
    sns.histplot(df[cfg.data['columns'][1]], bins=50, kde=True)
    plt.title("Filtered Steering Distribution")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高级数据清洗工具")
    # 区间参数（可选）
    parser.add_argument("--start", help="起始文件名 (带或不带.jpg)")
    parser.add_argument("--end", help="结束文件名 (带或不带.jpg)")
    
    # 模式选择
    parser.add_argument("--mode", choices=['manual', 'auto'], default='manual', 
                        help="manual: 只执行区间或同步; auto: 执行降采样")
    
    # 自动模式超参
    parser.add_argument("--keep_ratio", type=float, default=0.4, help="直道保留比例")
    parser.add_argument("--threshold", type=float, default=0.05, help="直道判定阈值")

    args = parser.parse_args()
    clean_data(args)
