import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# 確保可以匯入管線化腳本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trajector_processing_unified_pipeline import process_video_pipeline

def get_user_info():
    """獲取使用者資訊"""
    print("\n👤 請輸入使用者資訊:")
    print("="*40)
    name = input("請輸入姓名 (預設: test_user): ").strip() or "test_user"
    height = input("請輸入身高 (cm, 預設: 175): ").strip() or "175"
    return name, int(height)

def find_videos(input_folder="input_videos"):
    """自動尋找影片檔案"""
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ 找不到資料夾: {input_path.absolute()}")
        return None, None

    all_videos = list(input_path.glob("*.MP4")) + list(input_path.glob("*.mp4"))
    side_video = None
    deg45_video = None

    for video in all_videos:
        name = video.name.lower()
        if "side" in name or "側面" in name:
            side_video = str(video.absolute())
        elif "45" in name or "角度" in name:
            deg45_video = str(video.absolute())

    if not side_video or not deg45_video:
        print("⚠️ 無法自動識別影片，請手動指定或確保檔名包含 'side' 與 '45'")
        # 如果只有兩個影片，就直接分配
        if len(all_videos) >= 2:
            side_video = str(all_videos[0].absolute())
            deg45_video = str(all_videos[1].absolute())
            print(f"   自動分配: 側面={all_videos[0].name}, 45度={all_videos[1].name}")
    
    return side_video, deg45_video

def verify_video(video_path):
    """驗證影片是否可讀取"""
    if not video_path: return False
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if frames <= 0 or fps <= 0:
        print(f"❌ 影片讀取失敗: {video_path} (Frames: {frames}, FPS: {fps})")
        return False
    print(f"✅ 影片驗證成功: {Path(video_path).name} ({frames} 幀, {fps:.2f} FPS)")
    return True

def main():
    print("🎾 AI 網球教練 - 管線化流程測試工具 (Pipeline Mode)")
    print("=" * 60)
    
    # 1. 獲取資訊
    name, height = get_user_info()
    
    # 2. 尋找影片
    side_v, deg45_v = find_videos()
    
    if not side_v or not deg45_v:
        print("❌ 找不到必要的影片檔案，請檢查 input_videos 資料夾。")
        return

    # 3. 驗證影片
    if not verify_video(side_v) or not verify_video(deg45_v):
        print("❌ 影片檔案有問題，請檢查路徑或格式。")
        return

    # 4. 設定投影矩陣 (使用預設值)
    P1 = np.array([
         [  916.626242,     0.000000,   960.250417,     0.000000],
         [    0.000000,   921.951283,   523.154606,     0.000000],
         [    0.000000,     0.000000,     1.000000,     0.000000],
    ])

    P2 = np.array([
        [  782.909772,   -18.152980,  1066.677600, -255341.954492],
        [  -25.104948,   925.678666,   514.730223, 46851.486878],
        [   -0.122625,     0.020539,     0.992241,    90.876653],
    ])

    knn_dataset_path = "knn_dataset_new.json"

    # 5. 啟動管線化流程
    print(f"\n🚀 啟動管線化流程...")
    print(f"   使用者: {name}")
    print(f"   側面影片: {Path(side_v).name}")
    print(f"   45度影片: {Path(deg45_v).name}")
    print("-" * 40)
    
    start_time = time.time()
    success = process_video_pipeline(side_v, deg45_v, name, P1, P2, knn_dataset_path)
    
    if success:
        print(f"\n✨ 測試完成！總耗時: {time.time() - start_time:.2f} 秒")
    else:
        print("\n❌ 處理流程失敗。")

if __name__ == "__main__":
    main()
