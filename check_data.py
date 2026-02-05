import pandas as pd
import cv2
import os

# 读取 CSV
df = pd.read_csv("data/actions.csv")
print(f"当前已采集数据量: {len(df)}")

# 抽查第 100 帧（或者最后几帧）
sample_idx = min(100, len(df)-1)
row = df.iloc[sample_idx]
img_path = os.path.join("data/frames", row['frame_id'])

if os.path.exists(img_path):
    img = cv2.imread(img_path)
    print(f"抽查图片成功！动作数据: 转向={row['steering']}, 油门={row['throttle']}")
    # 弹窗显示 2 秒
    cv2.imshow("Check", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
else:
    print("错误：找不到图片文件，请检查 CSV 中的路径名和 frames 文件夹是否一致！")
