import cv2
import json
import numpy as np
import time
import tqdm
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_hit_frame(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for frame_info in data:
        if frame_info.get("tennis_ball_hit", False):
            return frame_info["frame"]
    return None

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    cap.release()
    return total_frames, fps, duration
"""
def process_video(input_path, output_path, start_frame, frames_to_process, dimensions):
    cap = cv2.VideoCapture(input_path)
    width, height = dimensions
    
    # 🟢 無論是否支援 CUDA，都先定義 fourcc，避免變數不存在
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 若可用 CUDA，則可在未來加速解碼（可選）
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("[INFO] CUDA device detected, using GPU-accelerated video processing.")
    else:
        print("[INFO] No CUDA device detected, using CPU mode.")

    # 初始化輸出
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    
    # 設置讀取緩衝區大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
    
    # 跳到起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    batch_size = 32
    frames = []
    
    for i in range(0, frames_to_process, batch_size):
        batch_frames = min(batch_size, frames_to_process - i)
        for _ in range(batch_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        for frame in frames:
            out.write(frame)
        frames = []

    cap.release()
    out.release()

"""
def process_video(input_path, output_path, start_frame, frames_to_process, dimensions):
    cap = cv2.VideoCapture(input_path)
    width, height = dimensions
    
    # 使用 H.264 編碼器提升效能
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    
    # 設置讀取緩衝區大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
    
    # 直接跳到起始幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 批次讀取和寫入
    batch_size = 32
    frames = []
    
    for i in range(0, frames_to_process, batch_size):
        batch_frames = min(batch_size, frames_to_process - i)
        for _ in range(batch_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        # 批次寫入
        for frame in frames:
            out.write(frame)
        frames = []
    
    cap.release()
    out.release()

def synchronize_videos(input_path_1, input_path_2, json_path_1, json_path_2):
    trim_length=60

    # 獲取影片資訊
    frames1, fps1, duration1 = get_video_info(input_path_1)
    frames2, fps2, duration2 = get_video_info(input_path_2)
    
    print("\n原始影片資訊:")
    print(f"影片 1: {frames1} 幀, {duration1:.2f} 秒")
    print(f"影片 2: {frames2} 幀, {duration2:.2f} 秒")

    # 獲取擊球幀
    hit_frame_1 = get_hit_frame(json_path_1)
    hit_frame_2 = get_hit_frame(json_path_2)
    
    print(f"\n擊球幀位置:")
    print(f"影片 1: 第 {hit_frame_1} 幀")
    print(f"影片 2: 第 {hit_frame_2} 幀")

    # 計算剪輯範圍
    max_frames_after = min(frames1 - hit_frame_1, frames2 - hit_frame_2)
    max_frames_before = min(hit_frame_1, hit_frame_2)
    
    frames_before = min(trim_length // 2, max_frames_before)
    frames_after = min(trim_length - frames_before, max_frames_after)
    
    start_frame_1 = hit_frame_1 - frames_before
    start_frame_2 = hit_frame_2 - frames_before
    frames_to_process = frames_before + frames_after

    print(f"\n剪輯資訊:")
    print(f"影片 1: 從第 {start_frame_1} 幀到第 {start_frame_1 + frames_to_process} 幀")
    print(f"影片 2: 從第 {start_frame_2} 幀到第 {start_frame_2 + frames_to_process} 幀")

    # 獲取影片尺寸
    cap1 = cv2.VideoCapture(input_path_1)
    cap2 = cv2.VideoCapture(input_path_2)
    dimensions1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    dimensions2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap1.release()
    cap2.release()

    output_path_1 = input_path_1
    output_path_2 = input_path_2

    # 使用線程池並行處理兩個影片
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_video, input_path_1, output_path_1, 
                          start_frame_1, frames_to_process, dimensions1),
            executor.submit(process_video, input_path_2, output_path_2, 
                          start_frame_2, frames_to_process, dimensions2)
        ]
        
        # 等待所有處理完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"處理過程中發生錯誤: {str(e)}")

    final_duration = frames_to_process / fps1
    print(f"\n最終影片資訊:")
    print(f"兩個影片都是 {frames_to_process} 幀, {final_duration:.2f} 秒")
    print("\n同步完成!")

if __name__ == "__main__":
    start_time = time.time()
    
    input_video_1 = "pro_1_1_45_temp.mp4"
    input_video_2 = "pro_1_1_side_temp.mp4"
    json_path_1 = "pro_1_1_45_temp(2D_trajectory_smoothed).json"
    json_path_2 = "pro_1_1_side_temp(2D_trajectory_smoothed).json"

    print("開始執行影片同步...")
    synchronize_videos(input_video_1, input_video_2, json_path_1, json_path_2)
    
    print(f"執行時間: {time.time() - start_time:.4f}秒")