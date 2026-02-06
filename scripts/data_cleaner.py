import pandas as pd
import os
import argparse

def clean_data_by_filename(csv_path, img_dir, start_file, end_file):
    # 统一格式：确保 start 和 end 都带有 .jpg，因为你的 CSV 里有
    if not start_file.endswith(".jpg"): start_file += ".jpg"
    if not end_file.endswith(".jpg"): end_file += ".jpg"

    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {csv_path}")
        return
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 你的列名是 img_path
    target_col = 'img_path'
    
    if target_col not in df.columns:
        print(f"❌ 错误：在 CSV 中找不到列名 '{target_col}'")
        print(f"实际列名为: {list(df.columns)}")
        return
    
    # 确保字符串比较
    df[target_col] = df[target_col].astype(str)

    # 筛选范围 (在字符串排序中，这种带时间戳的文件名直接比较是有效的)
    mask = (df[target_col] >= start_file) & (df[target_col] <= end_file)
    to_delete = df[mask]
    
    if len(to_delete) == 0:
        print(f"❓ 未找到在 {start_file} 与 {end_file} 之间的数据")
        # 打印一行样例，帮你对比格式
        print(f"CSV 中的样例数据: {df[target_col].iloc[0]}")
        return

    print(f"⚠️ 准备从 CSV 和磁盘中删除 {len(to_delete)} 帧数据...")

    # 执行物理删除
    del_count = 0
    for img_name in to_delete[target_col]:
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            os.remove(img_path)
            del_count += 1
    
    # 保存更新后的 CSV
    df_cleaned = df[~mask]
    df_cleaned.to_csv(csv_path, index=False)
    
    print(f"✅ 清理完成！")
    print(f"物理删除图片: {del_count} 张")
    print(f"CSV 移除记录: {len(to_delete)} 条")
    print(f"剩余总帧数: {len(df_cleaned)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--csv", default="data/actions.csv")
    parser.add_argument("--img", default="data/frames/")
    
    args = parser.parse_args()
    clean_data_by_filename(args.csv, args.img, args.start, args.end)
