import sys
import json
import os

# 強制重新載入所有相關模組
modules_to_reload = [
    'trajector_processing_unified',
    'trajector_2D_sync',
    'trajector_2D_smoothing',
    'trajector_3D_smoothing',
    'trajectory_2D_output',
    'trajectory_3D_output',
    'trajectory_knn',
    'trajectory_gpt_single_feedback'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        del sys.modules[module_name]

import trajector_processing_unified
from ultralytics import YOLO
import numpy as np

username = "tim90"
trajectory_base = r"C:\Users\user\Documents\AI_Coach_Detection-prd\trajectory"
user_trajectory_path = os.path.join(trajectory_base, f"{username}__trajectory")

# 讀取分段結果
segmentation_file = os.path.join(user_trajectory_path, f"{username}__segmentation_results.json")
with open(segmentation_file, 'r', encoding='utf-8') as f:
    seg_data = json.load(f)

print(f"📊 讀取 {username} 的數據:")
print(f"   球對數量: {len(seg_data['ball_pairs'])} 對")

# 載入 YOLO 模型
print(f"\n🤖 載入 AI 模型...")
yolo_pose_model = YOLO("model/yolov8n-pose.pt")
yolo_tennis_ball_model = YOLO("model/tennisball_OD_v1.pt")

# 投影矩陣設定
P1 = np.array([
    [  877.037008,     0.000000,   956.954783,     0.000000],
    [    0.000000,   879.565925,   564.021385,     0.000000],
    [    0.000000,     0.000000,     1.000000,     0.000000],
])

P2 = np.array([
    [  408.666240,    -7.066100,  1265.246736, -264697.889698],
    [ -232.265915,   870.289013,   512.645370, 42861.701021],
    [   -0.400331,    -0.014736,     0.916252,    76.895470],
])

# KNN 資料集路徑
knn_dataset_path = 'knn_dataset_new.json'

print(f"✅ 模型和校正矩陣已載入\n")

# 處理每一對球
for ball_idx, ball_pair in enumerate(seg_data['ball_pairs'], 1):
    print(f"{'='*60}")
    print(f"🏐 處理第 {ball_idx} 球")
    print(f"{'='*60}")
    
    # 建立軌跡資料夾
    trajectory_folder = os.path.join(user_trajectory_path, f"trajectory_{ball_idx}")
    os.makedirs(trajectory_folder, exist_ok=True)
    
    # 取得分段影片路徑
    if ball_pair.get('side_data'):
        video_side = ball_pair['side_data'].get('segment')
    else:
        print(f"⚠️ 第 {ball_idx} 球沒有側面影片，跳過")
        continue
        
    if ball_pair.get('deg45_data'):
        video_45 = ball_pair['deg45_data'].get('segment')
    else:
        print(f"⚠️ 第 {ball_idx} 球沒有45度影片，跳過")
        continue
    
    print(f"   側面影片: {video_side}")
    print(f"   45度影片: {video_45}")
    
    try:
        # 執行完整的軌跡處理流程
        timing_results = {"description": f"Reprocessing ball {ball_idx}"}
        
        result = trajector_processing_unified.process_single_video_set(
            P1, P2,
            yolo_pose_model,
            yolo_tennis_ball_model,
            video_side,
            video_45,
            knn_dataset_path,
            username,
            trajectory_folder,
            timing_results
        )
        
        if result:
            print(f"✅ 第 {ball_idx} 球處理完成\n")
        else:
            print(f"⚠️ 第 {ball_idx} 球處理失敗\n")
            
    except Exception as e:
        print(f"❌ 第 {ball_idx} 球處理時發生錯誤: {str(e)}\n")
        import traceback
        traceback.print_exc()

print(f"{'='*60}")
print(f"🎉 {username} 所有球的軌跡處理完成！")
print(f"{'='*60}")
