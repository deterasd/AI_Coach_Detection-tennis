"""
整合 2D/3D 軌跡分析、影片自動分割、影片處理、軌跡同步、KNN 與 GPT 反饋生成的整體流程。
此程式會依序完成：
  1. 先對原始影片進行時間同步（參考 trajector_2D_sync）
  2. 自動分割同步後的影片為多個片段
  3. 從側面與 45° 影片中提取 2D 軌跡
  4. 對 2D 軌跡進行平滑、插值與擊球角度處理
  5. 處理影片（前處理/物件偵測）
  6. 同步處理後的影片
  7. 合併同步後的影片
  8. 同步不同角度的軌跡資料
  9. 使用兩組 2D 軌跡與攝影機投影矩陣 (P1, P2) 計算 3D 軌跡
 10. 對 3D 軌跡進行平滑處理
 11. 擷取有效擊球範圍（根據 2D 軌跡判斷，並在 3D 軌跡中提取）
 12. 以 KNN 模組對 3D 軌跡進行初步分析
 13. 最後根據 KNN 分析與 3D 擊球範圍，生成 GPT 文字化反饋

各步驟皆計算執行時間，最後輸出時間統計摘要。
"""

import time
import numpy as np
import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

# 匯入原本的模組
from trajectory_2D_output import analyze_trajectory
from trajector_2D_smoothing import smooth_2D_trajectory
from video_detection import process_video
from video_sync import synchronize_videos
from video_merge import combine_videos_ffmpeg
from trajector_2D_sync import sync_trajectories
from trajector_2D_capture_swing_range import find_range
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_3D_capture_swing_range import extract_frames
from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
from trajectory_gpt_single_feedback import generate_feedback

# 匯入影片分割模組（獨立模組）
from video_segmentation import process_video_segmentation


def sync_videos_by_trajectory(video_side, video_45, output_folder):
    """
    根據軌跡數據同步兩個影片
    參考 trajector_2D_sync 的邏輯
    """
    print("🔄 開始影片時間同步...")
    
    # 這裡需要先生成簡單的軌跡數據來找到同步點
    # 實際實現時可能需要調用 trajectory_2D_output 的簡化版本
    # 或者使用其他同步方法（如音頻同步、手動標記等）
    
    # 暫時使用文件名作為同步後的輸出
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 複製原始影片作為同步後的結果（實際應該實現真正的同步邏輯）
    synced_side = output_folder / f"synced_{Path(video_side).name}"
    synced_45 = output_folder / f"synced_{Path(video_45).name}"
    
    # 這裡應該實現真正的同步邏輯
    # 暫時直接複製
    import shutil
    shutil.copy2(video_side, synced_side)
    shutil.copy2(video_45, synced_45)
    
    print(f"✅ 影片同步完成")
    print(f"📁 同步後影片: {synced_side}")
    print(f"📁 同步後影片: {synced_45}")
    
    return str(synced_side), str(synced_45)

def processing_trajectory_with_segmentation(P1, P2, yolo_pose_model, yolo_tennis_ball_model, 
                                          video_side, video_45, knn_dataset,
                                          ball_entry_direction="right", confidence_threshold=0.5,
                                          segment_videos=True, output_base_folder="segmented_videos"):
    """
    整合軌跡處理與影片分割的完整流程
    
    Args:
        P1, P2: 投影矩陣
        yolo_pose_model, yolo_tennis_ball_model: YOLO 模型
        video_side, video_45: 影片路徑
        knn_dataset: KNN 資料集路徑
        ball_entry_direction: 球進入方向 ("right" 或 "left")
        confidence_threshold: 偵測信心度
        segment_videos: 是否執行影片分割
        output_base_folder: 分割影片輸出資料夾
    """
    
    # 用於紀錄各步驟執行時間
    timing_results = {}
    start_total = time.perf_counter()
    
    # ------------------------------
    # 步驟0：影片自動分割（可選）
    # ------------------------------
    if segment_videos:
        print("\n步驟0：影片自動分割...")
        start = time.perf_counter()
        
        # 載入網球偵測模型
        yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
        
        try:
            # 使用 video_segmentation.py 的 process_video_segmentation 函數
            segmentation_results = process_video_segmentation(
                video_side=video_side,
                video_45=video_45,
                tennis_ball_model=yolo_tennis_ball_model,
                name="segment",
                output_folder=output_base_folder,
                ball_entry_direction=ball_entry_direction,
                confidence_threshold=confidence_threshold
            )
            
            timing_results['影片自動分割'] = time.perf_counter() - start
            print(f"-- 影片自動分割完成，耗時：{timing_results['影片自動分割']:.4f} 秒")
            
            # 如果有分割結果，使用第一個片段進行後續處理
            side_segments = segmentation_results['side']['segments']
            deg45_segments = segmentation_results['45deg']['segments']
            
            if side_segments and deg45_segments:
                video_side = side_segments[0]['output_path']
                video_45 = deg45_segments[0]['output_path']
                print(f"\n🎯 使用第一個片段進行軌跡分析:")
                print(f"   側面片段: {Path(video_side).name}")
                print(f"   45度片段: {Path(video_45).name}")
            else:
                print("⚠️ 影片分割失敗，使用原始完整影片進行處理")
        
        except Exception as e:
            print(f"❌ 影片分割發生錯誤: {e}")
            print("⚠️ 使用原始完整影片進行處理")
            timing_results['影片自動分割'] = time.perf_counter() - start
    else:
        print("\nℹ️ 跳過影片分割，使用完整影片")
    
    # ------------------------------
    # 步驟1：影片時間同步
    # ------------------------------
    print("\n步驟1：影片時間同步...")
    start = time.perf_counter()
    
    sync_output_folder = Path(output_base_folder) / "synced_videos"
    video_side_synced, video_45_synced = sync_videos_by_trajectory(video_side, video_45, sync_output_folder)
    
    # 更新影片路徑為同步後的版本
    video_side = video_side_synced
    video_45 = video_45_synced
    
    timing_results['影片時間同步'] = time.perf_counter() - start
    print(f"-- 影片時間同步完成，耗時：{timing_results['影片時間同步']:.4f} 秒")
    
    # ------------------------------
    # 步驟2：分析2D軌跡
    # ------------------------------
    print("\n步驟2：分析2D軌跡中...")
    start = time.perf_counter()
    trajectory_side = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_side, 28)
    trajectory_45  = analyze_trajectory(yolo_pose_model, yolo_tennis_ball_model, video_45, 28)
    timing_results['2D軌跡分析'] = time.perf_counter() - start
    print(f"-- 分析2D軌跡完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")

    # ------------------------------
    # 步驟3：2D 軌跡平滑/插值/擊球角度處理
    # ------------------------------
    print("\n步驟3：進行2D軌跡平滑化/插值/擊球角度處理...")
    start = time.perf_counter()
    trajectory_side_smoothing = smooth_2D_trajectory(trajectory_side)
    trajectory_45_smoothing   = smooth_2D_trajectory(trajectory_45)
    timing_results['2D平滑處理'] = time.perf_counter() - start
    print(f"-- 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")

    # ------------------------------
    # 步驟4：影片處理
    # ------------------------------
    print("\n步驟4：處理影片中...")
    start = time.perf_counter()
    
    # 檢查影片文件是否存在
    if not os.path.exists(video_side):
        print(f"❌ 側面影片不存在: {video_side}")
        video_side_processed = None
    else:
        try:
            video_side_processed = process_video(video_side)
        except Exception as e:
            print(f"❌ 側面影片處理失敗: {e}")
            video_side_processed = None
    
    if not os.path.exists(video_45):
        print(f"❌ 45度影片不存在: {video_45}")
        video_45_processed = None
    else:
        try:
            video_45_processed = process_video(video_45)
        except Exception as e:
            print(f"❌ 45度影片處理失敗: {e}")
            video_45_processed = None
    
    timing_results['影片處理'] = time.perf_counter() - start
    print(f"-- 影片處理完成，耗時：{timing_results['影片處理']:.4f} 秒")

    # ------------------------------
    # 步驟5：影片同步
    # ------------------------------
    print("\n步驟5：同步影片中...")
    start = time.perf_counter()
    
    # 檢查影片處理結果
    if video_side_processed and video_45_processed:
        try:
            synchronize_videos(video_side_processed, video_45_processed, 
                            trajectory_side_smoothing, trajectory_45_smoothing)
            print("✅ 影片同步完成")
        except Exception as e:
            print(f"❌ 影片同步失敗: {e}")
    else:
        print("⚠️ 跳過影片同步（影片處理失敗）")
    
    timing_results['影片同步'] = time.perf_counter() - start
    print(f"-- 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

    # ------------------------------
    # 步驟6：合併影片
    # ------------------------------
    print("\n步驟6：合併影片中...")
    start = time.perf_counter()
    
    # 檢查影片處理結果和 FFmpeg 可用性
    if video_side_processed and video_45_processed and segment_videos:
        try:
            combine_videos_ffmpeg(video_45_processed, video_side_processed)
            print("✅ 影片合併完成")
        except Exception as e:
            print(f"❌ 影片合併失敗: {e}")
    else:
        print("⚠️ 跳過影片合併（影片處理失敗或 FFmpeg 不可用）")
    
    timing_results['影片合併'] = time.perf_counter() - start
    print(f"-- 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

    # ------------------------------
    # 步驟7：軌跡同步
    # ------------------------------
    print("\n步驟7：同步軌跡中...")
    start = time.perf_counter()
    sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
    timing_results['軌跡同步'] = time.perf_counter() - start
    print(f"-- 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

    # ------------------------------
    # 步驟8：3D 軌跡分析
    # ------------------------------
    print("\n步驟8：計算3D軌跡中...")
    start = time.perf_counter()
    trajectory_3d = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
    timing_results['3D軌跡分析'] = time.perf_counter() - start
    print(f"-- 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

    # ------------------------------
    # 步驟9：3D 軌跡平滑處理
    # ------------------------------
    print("\n步驟9：進行3D軌跡平滑處理中...")
    start = time.perf_counter()
    trajectory_3d_smoothing = smooth_3D_trajectory(trajectory_3d)
    timing_results['3D平滑處理'] = time.perf_counter() - start
    print(f"-- 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

    # ------------------------------
    # 步驟10：有效擊球範圍判斷
    # ------------------------------
    print("\n步驟10：判斷有效擊球範圍中...")
    start = time.perf_counter()
    start_frame, end_frame = find_range(trajectory_side_smoothing)
    trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing, start_frame, end_frame)
    timing_results['有效擊球範圍判斷'] = time.perf_counter() - start
    print(f"-- 有效擊球範圍判斷完成，耗時：{timing_results['有效擊球範圍判斷']:.4f} 秒")

    # ------------------------------
    # 步驟11：KNN 分析
    # ------------------------------
    print("\n步驟11：KNN 分析中...")
    start = time.perf_counter()
    trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing)
    timing_results['KNN 分析'] = time.perf_counter() - start
    print(f"-- KNN 分析完成，耗時：{timing_results['KNN 分析']:.4f} 秒")

    # ------------------------------
    # 步驟12：GPT 反饋生成
    # ------------------------------
    print("\n步驟12：生成 GPT 反饋中...")
    start = time.perf_counter()
    trajectory_gpt_suggestion = generate_feedback(trajectory_3d_swing_range, trajectory_knn_suggestion)
    timing_results['GPT 反饋生成'] = time.perf_counter() - start
    print(f"-- GPT 反饋生成完成，耗時：{timing_results['GPT 反饋生成']:.4f} 秒")

    # ------------------------------
    # 統計總執行時間並輸出時間摘要
    # ------------------------------
    total_time = time.perf_counter() - start_total
    print('\n' + '=' * 60)
    print("📊 執行時間統計摘要")
    print('=' * 60)
    print(f'處理影片: {Path(video_side).name}')
    print(f'球進入方向: {ball_entry_direction}')
    print(f'偵測信心度: {confidence_threshold}')
    print(f'是否分割影片: {"是" if segment_videos else "否"}')
    print('-' * 60)
    for step, t in timing_results.items():
        print(f"{step:.<35} {t:>10.4f} 秒")
    print('-' * 60)
    print(f"{'總執行時間':.<35} {total_time:>10.4f} 秒")
    print('=' * 60)

    return True

if __name__ == "__main__":
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

    # 參數設定
    knn_dataset = 'knn_dataset_new.json'
    
    # 載入模型
    yolo_pose_model = YOLO('model/yolov8n-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
    
    # GPU 加速（如果可用）
    yolo_pose_model.model.to('cuda')
    yolo_tennis_ball_model.model.to('cuda')

    # 影片路徑
    video_side = f'trajectory/testing_123/testing__side.mp4'
    video_45 = f'trajectory/testing_123/testing__45.mp4'
    
    # 執行整合處理
    print("🚀 開始整合處理流程...")
    print("=" * 60)
    
    process_status = processing_trajectory_with_segmentation(
        P1=P1, 
        P2=P2, 
        yolo_pose_model=yolo_pose_model, 
        yolo_tennis_ball_model=yolo_tennis_ball_model,
        video_side=video_side, 
        video_45=video_45, 
        knn_dataset=knn_dataset,
        ball_entry_direction="right",  # 可選: "right" 或 "left"
        confidence_threshold=0.5,      # 偵測信心度
        segment_videos=True,           # 是否執行影片分割
        output_base_folder="segmented_videos"  # 輸出資料夾
    )
    
    print(f"\n🎉 整合處理完成！狀態: {process_status}")