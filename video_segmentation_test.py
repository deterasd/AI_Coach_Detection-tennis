import time
import cv2
from pathlib import Path
from ultralytics import YOLO
from video_segmentation import detect_ball_entries_optimized

def test_segmentation_speed(video_path, model_path):
    """測試影片分割速度與準確度"""
    print(f"🚀 開始測試影片分割優化: {video_path}")
    
    # 載入模型
    print("📦 載入模型中...")
    try:
        model = YOLO(model_path)
        # 預熱模型
        model(cv2.imread(video_path), verbose=False)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 執行偵測
    start_time = time.time()
    
    try:
        entries, exits = detect_ball_entries_optimized(
            video_path, 
            model, 
            confidence_threshold=0.5,
            ball_entry_direction="right",
            enable_exit_detection=True,
            exit_timeout=1.5
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*40)
        print(f"📊 測試結果:")
        print(f"   耗時: {duration:.4f} 秒")
        print(f"   偵測到的球數: {len(entries)}")
        print(f"   進入時間點: {[f'{t:.2f}s' for t in entries]}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"❌ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 設定測試參數
    video_file = "input_videos/tennis_side.MP4"  # 請確保此檔案存在
    model_file = "model/tennisball_OD_v1.pt"
    
    if not Path(video_file).exists():
        # 嘗試自動尋找
        videos = list(Path("input_videos").glob("*.MP4"))
        if videos:
            video_file = str(videos[0])
            print(f"⚠️ 預設影片不存在，自動使用: {video_file}")
        else:
            print("❌ 找不到測試影片，請檢查 input_videos 資料夾")
            exit(1)
            
    test_segmentation_speed(video_file, model_file)
