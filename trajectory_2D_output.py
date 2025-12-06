import numpy as np
import cv2
import json
import time
import torch
import gc
import math
from torch.cuda.amp import autocast
from ultralytics import YOLO
import threading
import queue

class NanToNullEncoder(json.JSONEncoder):
    """自定義 JSON encoder，將 NaN 轉換為 null"""
    def encode(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 'null'
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        for chunk in super().iterencode(obj, _one_shot):
            yield chunk.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')

def frame_reader(video_path, frame_queue, stop_event):
    """持續讀取影片 frame 並放入 queue"""
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((frame_number, frame))
        frame_number += 1
    cap.release()
    stop_event.set()  # 訊號：影片已讀取完畢
"""
def process_single_frame(body_result, ball_result,paddle_result, keypoint_names, frame_number):
    #處理單一 frame 的結果，回傳 frame_data 字典
    frame_data = {
        "frame": frame_number,
        "tennis_ball": {"x": None, "y": None},
        "paddle": {"x": None, "y": None}   # 👈 新增球拍欄位
    }
    for keypoint in keypoint_names:
        frame_data[keypoint] = {"x": None, "y": None}

    # 處理身體關鍵點
    if body_result.keypoints is not None:
        keypoints = body_result.keypoints.xy[0].cpu().numpy()
        if len(keypoints) == len(keypoint_names):
            for idx, keypoint in enumerate(keypoint_names):
                x, y = keypoints[idx][:2]
                coords = {
                    "x": int(x) if x != 0.0 else None,
                    "y": int(y) if y != 0.0 else None
                }
                frame_data[keypoint].update(coords)
    # 處理網球位置
    for box in ball_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if float(box.conf[0]) > 0.8:
            frame_data["tennis_ball"].update({
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2
            })
            break
    for box in paddle_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if float(box.conf[0]) > 0.3:  # 門檻值可調
            frame_data["paddle"].update({
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2
            })
            break
    return frame_data
"""
def process_single_frame(body_result, ball_result, paddle_result, keypoint_names, frame_number):
    """處理單一 frame 的結果，回傳 frame_data 字典"""
    frame_data = {
        "frame": frame_number,
        "tennis_ball": {"x": None, "y": None},
        "paddle": {
            "top": {"x": None, "y": None},
            "right": {"x": None, "y": None},
            "bottom": {"x": None, "y": None},
            "left": {"x": None, "y": None},
            "center": {"x": None, "y": None}
        }
    }
    for keypoint in keypoint_names:
        frame_data[keypoint] = {"x": None, "y": None}

    # --- 身體關鍵點 ---
    if body_result.keypoints is not None:
        keypoints = body_result.keypoints.xy[0].cpu().numpy()
        if len(keypoints) == len(keypoint_names):
            for idx, keypoint in enumerate(keypoint_names):
                x, y = keypoints[idx][:2]
                coords = {
                    "x": int(x) if x != 0.0 else None,
                    "y": int(y) if y != 0.0 else None
                }
                frame_data[keypoint].update(coords)

    # --- 網球位置 (物件偵測) ---
    for box in ball_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if float(box.conf[0]) > 0.8:
            frame_data["tennis_ball"].update({
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2
            })
            break

    # --- 球拍位置 (keypoint 偵測，四點) ---
        # --- 球拍位置 (keypoint 偵測，四點) ---
    if paddle_result.keypoints is not None and len(paddle_result.keypoints.xy) > 0:
        kpts = paddle_result.keypoints.xy[0].cpu().numpy()
        if kpts.shape[0] >= 4:  # 確保至少有四個點
            top, right, bottom, left = kpts[:4]
            frame_data["paddle"]["top"] = {"x": int(top[0]), "y": int(top[1])}
            frame_data["paddle"]["right"] = {"x": int(right[0]), "y": int(right[1])}
            frame_data["paddle"]["bottom"] = {"x": int(bottom[0]), "y": int(bottom[1])}
            frame_data["paddle"]["left"] = {"x": int(left[0]), "y": int(left[1])}
            # 計算球拍中心
            cx = int((top[0] + right[0] + bottom[0] + left[0]) / 4)
            cy = int((top[1] + right[1] + bottom[1] + left[1]) / 4)
            frame_data["paddle"]["center"] = {"x": cx, "y": cy}
    else:
        # Debug：確認沒偵測到 paddle 的情況
        print(f"[DEBUG] frame {frame_number}: no paddle detected")

    return frame_data
def process_video_batch(pose_model, ball_model,paddle_model, video_path, batch_size=16):
    """使用異步讀取與批次推論加速影片處理"""
    frame_queue = queue.Queue(maxsize=2 * batch_size)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(video_path, frame_queue, stop_event))
    reader_thread.start()

    frame_json = []
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"  
    ]
    
    batch_frames = []
    batch_indices = []

    # 持續從 queue 讀取 frame 並累積成批次
    while not (stop_event.is_set() and frame_queue.empty()):
        try:
            frame_index, frame = frame_queue.get(timeout=0.1)
            batch_frames.append(frame)
            batch_indices.append(frame_index)
            if len(batch_frames) == batch_size:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    body_results = pose_model(batch_frames, verbose=False)
                    ball_results = ball_model(batch_frames, verbose=False)
                    paddle_results = paddle_model(batch_frames, verbose=False, conf=0.1)
                for idx, (body_result, ball_result,paddle_result) in enumerate(zip(body_results, ball_results,paddle_results)):
                    frame_data = process_single_frame(body_result, ball_result,paddle_result, keypoint_names, batch_indices[idx])
                    frame_json.append(frame_data)
                # 清除批次資料
                del batch_frames, batch_indices, body_results, ball_results
                batch_frames = []
                batch_indices = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except queue.Empty:
            continue

    # 處理剩餘的 frame
    if batch_frames:
        with torch.no_grad(), torch.amp.autocast('cuda'):
            body_results = pose_model(batch_frames, verbose=False)
            ball_results = ball_model(batch_frames, verbose=False)
            paddle_results = paddle_model(batch_frames, verbose=False, conf=0.1)
        for idx, (body_result, ball_result, paddle_result) in enumerate(zip(body_results, ball_results,paddle_results)):
            frame_data = process_single_frame(body_result, ball_result,paddle_result, keypoint_names, batch_indices[idx])
            frame_json.append(frame_data)
        del batch_frames, batch_indices, body_results, ball_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 若最後一幀關鍵點缺失，使用前一幀補上
    if frame_json and len(frame_json) > 1:
        last_frame = frame_json[-1]
        prev_frame = frame_json[-2]
        for keypoint in keypoint_names:
            if last_frame[keypoint]["x"] is None:
                last_frame[keypoint] = prev_frame[keypoint]

    reader_thread.join()
    return frame_json

def analyze_trajectory(pose_model, ball_model,paddle_model, video_path, batch_size):
    trajectory = process_video_batch(pose_model, ball_model,paddle_model, video_path, batch_size=batch_size)
    output_path = video_path.replace('.mp4', '(2D_trajectory).json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    return output_path

if __name__ == "__main__":
    total_start_time = time.time()
    
    model_load_start = time.time()
    pose_model = YOLO('model/yolov8n-pose.pt')
    ball_model = YOLO('model/tennisball_OD_v1.pt')
    paddle_model = YOLO('model/best-paddlekeypoint.pt')  # 新增：球拍模型
    # 將模型移至 GPU（若有 CUDA）
    if torch.cuda.is_available():
        pose_model.model.to('cuda')
        ball_model.model.to('cuda')
        paddle_model.model.to('cuda')
        print("Models moved to CUDA.")
    else:
        print("CUDA not available, using CPU.")
    
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.8f}s")
    
    video_path = '測試2__1_45_compressed.mp4'
    
    analysis_start = time.time()
    output_path = analyze_trajectory(pose_model, ball_model,paddle_model, video_path, batch_size=4)
    analysis_time = time.time() - analysis_start
    print(f"Trajectory analysis time: {analysis_time:.8f}s")
    
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")
