#使用來測試的程式，可以直接放影片進input_video

"""
統一輸出管理的軌跡處理流程 - 專門用於 simple_test
支援多球分析，每顆球會有獨立的資料夾和完整的分析結果
整合影片自動分割功能
"""

import time
import numpy as np
import os
import json
import shutil
import torch
import gc
import psutil
import cv2
import subprocess
import math
import threading
import queue
from pathlib import Path
from ultralytics import YOLO

def load_segmentation_config():
    """載入分割設定檔，如果不存在則回傳預設值"""
    config_path = Path("segmentation_config.json")
    default_config = {
        "segmentation": {
            "next_ball_offset": 1.2,
            "exit_buffer_time": 0.2,
            "preview_start_time": -0.2
        }
    }
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 讀取設定檔失敗: {e}，使用預設值")
    return default_config

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

def analyze_trajectory_with_output_folder(pose_model, ball_model, video_path, batch_size, output_folder, paddle_model=None):
    """
    分析軌跡並將結果保存到指定資料夾
    """
    from trajectory_2D_output import process_video_batch
    import json
    
    # 如果沒有提供 paddle_model，載入預設模型
    if paddle_model is None:
        try:
            paddle_model = YOLO('model/tennispaddle.pt')  # 使用正確的球拍關鍵點模型
            print("📦 已自動載入球拍模型: model/tennispaddle.pt")
        except Exception as e:
            print(f"⚠️ 球拍模型載入失敗: {e}")
            # 如果載入失敗，使用 ball_model 作為替代（雖然不會有效果）
            paddle_model = ball_model
    
    trajectory = process_video_batch(pose_model, ball_model, paddle_model, video_path, batch_size=batch_size)
    
    # 生成輸出檔案名稱（基於原始檔名）
    video_name = Path(video_path).stem
    output_path = Path(output_folder) / f"{video_name}(2D_trajectory).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def smooth_2D_trajectory_with_output_folder(trajectory_path, output_folder):
    """
    平滑處理2D軌跡並將結果保存到指定資料夾
    """
    from trajector_2D_smoothing import smooth_2D_trajectory
    import json
    
    # 執行平滑處理
    smoothed_trajectory_path = smooth_2D_trajectory(trajectory_path)
    
    # 移動結果到指定資料夾
    source_path = Path(smoothed_trajectory_path)
    target_path = Path(output_folder) / source_path.name
    
    if source_path.exists() and source_path != target_path:
        import shutil
        shutil.move(str(source_path), str(target_path))
        return str(target_path)
    
    return smoothed_trajectory_path

def process_video_with_output_folder(video_path, output_folder, ball_model=None, pose_model=None, paddle_model=None, json_path=None):
    """
    處理影片並將結果保存到指定資料夾
    """
    from video_detection import process_video
    import shutil
    
    # 執行影片處理 (process_video 會自動使用正確的 paddle_model_path)
    processed_video_path = process_video(
        video_path,
        ball_model=ball_model,
        pose_model=pose_model,
        paddle_model=paddle_model,
        json_path=json_path
    )
    
    # 移動結果到指定資料夾
    if processed_video_path and Path(processed_video_path).exists():
        source_path = Path(processed_video_path)
        target_path = Path(output_folder) / source_path.name
        
        if source_path != target_path:
            shutil.move(str(source_path), str(target_path))
            return str(target_path)
    
    return processed_video_path

def save_3d_trajectory_with_output_folder(trajectory_3d, output_folder, name):
    """
    保存3D軌跡到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_3d, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def save_3d_smoothed_trajectory_with_output_folder(trajectory_3d_smoothing, output_folder, name):
    """
    保存3D平滑軌跡到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory_smoothed).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_3d_smoothing, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def save_3d_swing_range_with_output_folder(trajectory_3d_swing_range, output_folder, name):
    """
    保存3D擊球範圍軌跡到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory_smoothed)_only_swing.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_3d_swing_range, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def save_knn_feedback_with_output_folder(knn_result, output_folder, name):
    """
    保存KNN反饋到指定資料夾
    """
    output_path = Path(output_folder) / f"{name}_segment_knn_feedback.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 如果是列表，取第一個元素或合併成字串
        if isinstance(knn_result, list):
            if len(knn_result) > 0:
                f.write(knn_result[0])
            else:
                f.write("無KNN分析結果")
        else:
            f.write(knn_result)
    
    return str(output_path)

def save_gpt_feedback_with_output_folder(gpt_result, output_folder, name):
    """
    保存GPT反饋到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment_gpt_feedback.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gpt_result, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
    
    return str(output_path)

def clear_all_memory():
    """清理所有記憶體（GPU + RAM）"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def check_system_memory():
    """檢查系統記憶體使用情況"""
    memory = psutil.virtual_memory()
    return memory.available > 2 * 1024**3  # 至少需要 2GB 可用

def check_gpu_memory():
    """檢查 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        return (total_memory - cached_memory) > 1.0  # 至少需要 1GB 可用
    return False

def detect_ball_in_frame(frame, model):
    """偵測畫面中的網球"""
    results = model(frame, verbose=False)
    
    if not results[0].boxes:
        return None, 0
    
    best_box = max(results[0].boxes, key=lambda box: float(box.conf[0]))
    confidence = float(best_box.conf[0])
    
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
    position = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    return position, confidence

def update_ball_tracking(active_balls, position, current_time, fps):
    """更新球追蹤資訊"""
    if not position:
        return
    
    # 動態調整追蹤距離
    max_tracking_distance = max(200, fps * 8)
    
    # 找到最近的球進行位置更新
    min_distance = float('inf')
    closest_ball_id = None
    
    for ball_id, ball_info in active_balls.items():
        if ball_info['positions']:
            last_pos = ball_info['positions'][-1]
            distance = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
            if distance < min_distance and distance <= max_tracking_distance:
                min_distance = distance
                closest_ball_id = ball_id
    
    # 更新最近球的位置
    if closest_ball_id is not None:
        active_balls[closest_ball_id]['positions'].append(position)
        active_balls[closest_ball_id]['last_seen'] = current_time

def check_ball_exits(active_balls, edges, current_time, exit_timeout):
    """檢查球是否出場"""
    exited_balls = []
    balls_to_remove = []
    
    for ball_id, ball_info in active_balls.items():
        time_since_last_seen = current_time - ball_info['last_seen']
        
        # 檢查是否超過出場等待時間
        if time_since_last_seen >= exit_timeout:
            # 分析球的移動軌跡判斷是否真的出場
            if len(ball_info['positions']) >= 2:
                is_exit, reason = is_ball_exit_right_edge(ball_info['positions'], edges)
                if is_exit:
                    exit_time = ball_info['last_seen']
                    exited_balls.append((ball_id, exit_time, reason))
                    balls_to_remove.append(ball_id)
                else:
                    # 如果不是真的出場，重新開始追蹤
                    balls_to_remove.append(ball_id)
            else:
                balls_to_remove.append(ball_id)
    
    # 移除已出場的球
    for ball_id in balls_to_remove:
        del active_balls[ball_id]
    
    return exited_balls

def is_ball_exit_right_edge(positions, edges):
    """檢查是否為右邊出場"""
    if len(positions) < 2:
        return False, "軌跡太短"
    
    # 分析最近的軌跡
    recent_positions = positions[-min(8, len(positions)):]
    
    # 檢查最終位置是否在右邊範圍
    end_pos = recent_positions[-1]
    right_boundary = edges['right'] - 100
    
    is_at_right_edge = end_pos[0] > right_boundary
    
    if not is_at_right_edge:
        return False, "不在右邊界"
    
    # 分析移動趨勢
    movement_analysis = analyze_movement_trend(recent_positions, edges)
    
    # 多種出場情況判斷
    exit_reasons = []
    
    if movement_analysis['moving_right']:
        exit_reasons.append("向右移動")
    
    if movement_analysis['from_center']:
        exit_reasons.append("從中央區域出場")
    
    if movement_analysis['consistently_right']:
        exit_reasons.append("持續在右邊緣")
    
    if movement_analysis['moving_outward']:
        exit_reasons.append("向邊緣移動")
    
    # 右邊界移動檢查
    if len(recent_positions) >= 2:
        x_trend = recent_positions[-1][0] - recent_positions[0][0]
        if x_trend > 10:
            exit_reasons.append(f"右邊界移動 (ΔX: {x_trend:.0f})")
    
    # 判斷出場
    is_exit = len(exit_reasons) > 0
    reason = "; ".join(exit_reasons) if exit_reasons else "無明確出場跡象"
    
    return is_exit, reason

def analyze_movement_trend(positions, edges):
    """分析球的移動趨勢"""
    if len(positions) < 2:
        return {'moving_right': False, 'from_center': False, 'consistently_right': False, 'moving_outward': False}
    
    # 計算畫面區域
    width = edges['right'] - edges['left']
    center_x_min = edges['left'] + width * 0.25
    center_x_max = edges['right'] - width * 0.25
    right_zone = edges['right'] - width * 0.3
    
    # X方向移動趨勢
    x_start = positions[0][0]
    x_end = positions[-1][0]
    x_trend = x_end - x_start
    
    # 檢查是否從中央開始
    from_center = center_x_min <= x_start <= center_x_max
    
    # 檢查是否向右移動
    moving_right = x_trend > 10
    
    # 檢查是否持續在右邊
    consistently_right = all(pos[0] > right_zone for pos in positions[-min(3, len(positions)):])
    
    # 檢查是否向外移動
    moving_outward = moving_right or consistently_right or x_trend > 8
    
    return {
        'moving_right': moving_right,
        'from_center': from_center,
        'consistently_right': consistently_right,
        'moving_outward': moving_outward
    }

def detect_ball_entries_optimized(video_path, model, confidence_threshold=0.5, 
                                detection_area="right_upper_two_thirds", 
                                enable_exit_detection=True, exit_timeout=1.5,
                                ball_entry_direction="right"):
    """
    優化的球進入偵測，支援多球追蹤和動態分割模式
    採用 video_segment_tester_optimized 的進階算法
    """
    print(f"🔍 開始偵測球進入時間點: {Path(video_path).name}")
    print(f"   球進入方向: {'右邊' if ball_entry_direction == 'right' else '左邊'}")
    print(f"   偵測範圍: {detection_area}")
    print(f"   信心度閾值: {confidence_threshold}")
    print(f"   球出場偵測: {'啟用' if enable_exit_detection else '停用'}")
    if enable_exit_detection:
        print(f"   出場等待時間: {exit_timeout} 秒")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   影片資訊: {total_frames} 幀, {fps:.2f} FPS")
    print(f"   🎯 球追蹤距離: {max(200, fps * 8):.0f}像素 (根據{fps:.1f}FPS調整)")
    
    # 邊緣檢測參數
    edge_ratio = 0.15
    edges = {
        'left': frame_width * edge_ratio,
        'right': frame_width * (1 - edge_ratio),
        'top': frame_height * edge_ratio,
        'bottom': frame_height * (1 - edge_ratio)
    }
    
    # 偵測範圍設定 - 改進版本
    if ball_entry_direction == "right":
        print(f"   偵測範圍: 右邊緣上2/3區域 + 上邊緣右側2/3區域")
    else:
        print(f"   偵測範圍: 左邊緣上2/3區域 + 上邊緣左側2/3區域")
    
    # 初始化變數（使用 video_segment_tester_optimized 的算法）
    ball_entry_times = []
    ball_exit_times = []
    active_balls = {}  # 活躍球追蹤
    next_ball_id = 0
    
    # === 優化參數 ===
    SKIP_FRAMES_AFTER_FOUND = 60  # 找到球後跳過 60 幀 (約 1 秒)
    frames_to_skip = 0
    
    # --- 批次處理優化 (多執行緒讀取) ---
    batch_size = 32
    frames_batch = []
    
    # 定義多執行緒讀取類別
    class ThreadedVideoCapture:
        def __init__(self, path, queue_size=256): # 增加緩衝區大小
            self.cap = cv2.VideoCapture(path)
            self.q = queue.Queue(maxsize=queue_size)
            self.stopped = False
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            
        def start(self):
            self.thread.start()
            return self
            
        def update(self):
            while not self.stopped:
                if not self.q.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stopped = True
                        return
                    self.q.put(frame)
                else:
                    time.sleep(0.001) # 減少睡眠時間
                    
        def read(self):
            return self.q.get() if not self.q.empty() else None
            
        def running(self):
            return not self.stopped or not self.q.empty()
            
        def stop(self):
            self.stopped = True
            # 只有在非當前執行緒時才進行 join，避免 "cannot join current thread" 錯誤
            if self.thread.is_alive() and threading.current_thread() != self.thread:
                self.thread.join(timeout=1.0)
            if self.cap.isOpened():
                self.cap.release()

    # 啟動多執行緒讀取
    # 注意：原本的 cap 已經被用來讀取屬性，這裡重新開啟一個用於讀取幀
    cap.release() 
    video_stream = ThreadedVideoCapture(video_path).start()
    
    # 等待緩衝區填充一點
    time.sleep(0.5) # 減少等待時間
    
    frame_count = 0
    while video_stream.running() and frame_count < total_frames:
        if video_stream.q.empty():
            time.sleep(0.001) # 減少睡眠時間
            continue
            
        frame = video_stream.read()
        if frame is None:
            break
            
        # === 激進跳幀邏輯 (累積階段) ===
        if frames_to_skip > 0:
            frames_to_skip -= 1
            frame_count += 1
            continue
            
        frames_batch.append(frame)
        frame_count += 1
        
        # 當批次滿了或是最後一幀時進行推論
        if len(frames_batch) == batch_size or frame_count == total_frames:
            
            # --- 跳幀掃描優化 (Sparse Scanning) ---
            # 如果目前沒有追蹤任何球，我們可以先進行「稀疏掃描」
            # 只推論批次中的部分幀 (例如每 4 幀測一次)
            # 如果發現球，再對整個批次進行完整推論
            
            need_full_inference = True
            
            if not active_balls and len(frames_batch) >= 4:
                skip_step = 4
                sparse_indices = list(range(0, len(frames_batch), skip_step))
                sparse_frames = [frames_batch[i] for i in sparse_indices]
                
                # 稀疏推論
                sparse_results = model(sparse_frames, verbose=False, half=True)
                
                # 檢查是否有任何球被偵測到
                ball_found_in_sparse = False
                for res in sparse_results:
                    if res.boxes:
                        for box in res.boxes:
                            if float(box.conf[0]) >= confidence_threshold:
                                ball_found_in_sparse = True
                                break
                    if ball_found_in_sparse:
                        break
                
                if not ball_found_in_sparse:
                    # 如果稀疏掃描沒看到球，就跳過這個批次的完整推論
                    # 但我們還是要更新進度條
                    need_full_inference = False
                    
                    # 模擬進度更新
                    last_frame_idx = frame_count - 1
                    if last_frame_idx % int(fps * 10) < batch_size: # 簡單的檢查方式
                         progress = (last_frame_idx / total_frames) * 100
                         print(f"   進度: {progress:.1f}% (跳過空白片段)")

            # --- 完整推論 (Dense Inference) ---
            if need_full_inference:
                # 批次推論
                results = model(frames_batch, verbose=False, half=True) # 啟用 FP16 加速
                
                # 處理批次中的每一幀結果
                for i, result in enumerate(results):
                    # === 激進跳幀邏輯 (批次階段) ===
                    if frames_to_skip > 0:
                        frames_to_skip -= 1
                        continue
                        
                    # 計算當前處理的幀編號和時間
                    current_frame_idx = frame_count - len(frames_batch) + i
                    current_time = current_frame_idx / fps
                    
                    # 解析結果 (取代 detect_ball_in_frame)
                    position = None
                    confidence = 0.0
                    
                    if result.boxes:
                        # 找出信心度最高的框
                        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))
                        if float(best_box.conf[0]) >= confidence_threshold:
                            confidence = float(best_box.conf[0])
                            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                            position = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    ball_detected = position is not None
                    
                    # 檢查是否在邊緣區域 - 使用改進的偵測邏輯
                    in_edge = False
                    if ball_detected:
                        x, y = position
                        detection_mode = "right_only" if ball_entry_direction == "right" else "left_only"
                        in_edge = _is_ball_entry_edge(x, y, edges, detection_mode, frame_width, frame_height)
                    
                    # 更新活躍球追蹤
                    if ball_detected:
                        if in_edge and not active_balls:
                            # 沒有活躍球，這是新球進入
                            active_balls[next_ball_id] = {
                                'entry_time': current_time,
                                'positions': [position],
                                'last_seen': current_time
                            }
                            ball_entry_times.append(current_time)
                            print(f"   ⚾ 球進入時間: {current_time:.2f} 秒 (幀 {current_frame_idx}) - 球#{next_ball_id}")
                            next_ball_id += 1
                            
                            # === 激進跳幀邏輯 ===
                            frames_to_skip = SKIP_FRAMES_AFTER_FOUND
                            print(f"   🚀 發現新球！跳過接下來 {frames_to_skip} 幀...")
                            
                        elif active_balls:
                            # 已有活躍球，持續追蹤其位置
                            update_ball_tracking(active_balls, position, current_time, fps)
                    
                    # 檢查球出場
                    if enable_exit_detection:
                        exited_balls = check_ball_exits(active_balls, edges, current_time, exit_timeout)
                        for ball_id, exit_time, reason in exited_balls:
                            ball_exit_times.append(exit_time)
                            print(f"   🎯 球出場時間: {exit_time:.2f} 秒 - 球#{ball_id}: {reason}")
                    
                    # 顯示進度
                    if current_frame_idx % int(fps * 10) == 0:  # 每10秒顯示一次
                        progress = (current_frame_idx / total_frames) * 100
                        print(f"   進度: {progress:.1f}%")
            
            # 清空批次
            frames_batch = []
            
    video_stream.stop()
    
    # 處理最後一個球（影片結束時仍在畫面中的球）
    for ball_id, ball_info in active_balls.items():
        final_exit_time = (total_frames - 1) / fps
        ball_exit_times.append(final_exit_time)
        print(f"   🎯 最後片段延伸到影片結束: {final_exit_time:.2f} 秒")
    
    cap.release()
    
    print(f"✅ 偵測完成: 找到 {len(ball_entry_times)} 個球進入時間點")
    print(f"   總出場點: {len(ball_exit_times)}")
    
    return ball_entry_times, ball_exit_times
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   影片資訊: {total_frames} 幀, {fps:.2f} FPS, {frame_width}x{frame_height}")
    print(f"   🎯 球追蹤距離: {max(200, fps * 8):.0f}像素 (根據{fps:.1f}FPS調整)")
    
    # 邊緣檢測參數 - 使用與 video_segment_tester_optimized 相同的邏輯
    edge_ratio = 0.15
    edges = {
        'left': frame_width * edge_ratio,
        'right': frame_width * (1 - edge_ratio),
        'top': frame_height * edge_ratio,
        'bottom': frame_height * (1 - edge_ratio)
    }
    
    # 根據球進入方向調整偵測模式 - 改進版本
    if ball_entry_direction == "right":
        detection_mode = "right_only"  # 右邊緣上2/3 + 上方2/3右半邊
    else:
        detection_mode = "left_only"   # 左邊緣上2/3 + 上方2/3左半邊
    
    print(f"   偵測模式: {detection_mode}")
    print(f"   偵測邊界: 左{edges['left']:.0f}, 右{edges['right']:.0f}, 上{edges['top']:.0f}, 下{edges['bottom']:.0f}")
    if ball_entry_direction == "right":
        print(f"   偵測範圍: 右邊緣上2/3區域 + 上邊緣右側2/3區域")
    else:
        print(f"   偵測範圍: 左邊緣上2/3區域 + 上邊緣左側2/3區域")
    
    # 初始化追蹤變數
    ball_entry_times = []
    ball_exit_times = []
    active_balls = {}       # 活躍球追蹤 {ball_id: {'entry_time': float, 'positions': [], 'last_seen': float}}
    next_ball_id = 0        # 下一個球的ID
    min_interval = 2.0      # 最小間隔時間
    last_entry_time = -min_interval
    tracking_distance = max(200, fps * 8)  # 球追蹤距離
    
    detection_count = 0
    
    try:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            # YOLO 偵測
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # 檢查偵測結果並獲取球的位置
            detected_balls = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        confidence = box.conf[0].cpu().numpy()
                        
                        # 檢查是否在邊緣區域（球進入點）- 使用改進的偵測邏輯
                        is_entry = _is_ball_entry_edge(center_x, center_y, edges, detection_mode, frame_width, frame_height)
                        
                        if is_entry:
                            detected_balls.append({
                                'position': (center_x, center_y),
                                'confidence': confidence,
                                'time': current_time
                            })
                            detection_count += 1
            
            # 更新活躍球追蹤
            next_ball_id = _update_active_balls(active_balls, detected_balls, current_time, tracking_distance, next_ball_id)
            
            # 檢查球進入
            for ball_id, ball_data in active_balls.items():
                if ball_data.get('entry_recorded', False):
                    continue
                    
                # 檢查是否滿足進入條件
                if (current_time - last_entry_time >= min_interval and 
                    len(ball_data['positions']) >= 3):  # 至少被偵測到3次才認定為有效進入
                    
                    entry_time = ball_data['entry_time']
                    ball_entry_times.append(entry_time)
                    last_entry_time = entry_time
                    ball_data['entry_recorded'] = True
                    
                    print(f"   ⚾ 球進入時間: {entry_time:.2f} 秒 (球#{ball_id})")
            
            # 檢查球出場（如果啟用）
            if enable_exit_detection:
                balls_to_exit = []
                for ball_id, ball_data in active_balls.items():
                    time_since_last_seen = current_time - ball_data['last_seen']
                    if time_since_last_seen >= exit_timeout:
                        # 檢查球是否真的離開了畫面
                        if _is_ball_exited(ball_data['positions'], edges):
                            exit_time = ball_data['last_seen']
                            ball_exit_times.append(exit_time)
                            balls_to_exit.append(ball_id)
                            print(f"   🎯 球出場時間: {exit_time:.2f} 秒 (球#{ball_id})")
                
                # 移除已出場的球
                for ball_id in balls_to_exit:
                    del active_balls[ball_id]
            
            # 顯示進度
            if frame_idx % (fps * 10) == 0:  # 每10秒顯示一次
                progress = (frame_idx / total_frames) * 100
                print(f"   進度: {progress:.1f}% (偵測次數: {detection_count})")
    
    finally:
        cap.release()
    
    # 處理最後仍在追蹤的球
    if enable_exit_detection:
        final_time = (total_frames - 1) / fps
        for ball_id, ball_data in active_balls.items():
            if ball_data.get('entry_recorded', False):
                ball_exit_times.append(final_time)
                print(f"   🎯 最後球延伸到影片結束: {final_time:.2f} 秒 (球#{ball_id})")
    
    print(f"✅ 偵測完成: 找到 {len(ball_entry_times)} 個球進入時間點")
    print(f"   總偵測次數: {detection_count}")
    
    return ball_entry_times, ball_exit_times


def _is_ball_entry_edge(x, y, edges, detection_mode, frame_width, frame_height):
    """
    檢查球是否在進入邊緣區域 - 改進版本
    新增上邊緣偵測，避免從右上方飛來的球被遺漏
    
    偵測區域設計：
    - 右邊進入: 右邊緣上2/3 + 上邊緣右半邊
    - 左邊進入: 左邊緣上2/3 + 上邊緣左半邊
    """
    
    two_thirds_height = frame_height * (2/3)
    right_top_band = frame_width * (2/3)
    left_top_band = frame_width * (1/3)
    if detection_mode == "right_only":
        # 右側邊界：限制在上方 2/3 區域，避免下方誤判
        right_edge_y_threshold = two_thirds_height
        right_edge_in_zone = (x > edges['right'] and y < right_edge_y_threshold)
        
        # 上方區域：取上緣的右側 2/3 區域
        top_edge_in_zone = (y < two_thirds_height and x > right_top_band)
        
        return right_edge_in_zone or top_edge_in_zone
        
    elif detection_mode == "left_only":
        left_edge_y_threshold = two_thirds_height
        left_edge_in_zone = (x < edges['left'] and y < left_edge_y_threshold)
        
        # 上方區域：取上緣的左側 2/3 區域
        top_edge_in_zone = (y < two_thirds_height and x < left_top_band)
        
        return left_edge_in_zone or top_edge_in_zone
        
    elif detection_mode == "top_only":
        return y < two_thirds_height
    elif detection_mode == "right_top":
        return x > edges['right'] or y < edges['top']
    else:  # all_edges
        return (x < edges['left'] or x > edges['right'] or 
                y < edges['top'] or y > edges['bottom'])


def _update_active_balls(active_balls, detected_balls, current_time, tracking_distance, next_ball_id):
    """更新活躍球追蹤"""
    # 為每個偵測到的球找最接近的活躍球或創建新球
    for detection in detected_balls:
        pos = detection['position']
        matched_ball_id = None
        min_distance = float('inf')
        
        # 尋找最接近的活躍球
        for ball_id, ball_data in active_balls.items():
            if ball_data['positions']:
                last_pos = ball_data['positions'][-1]
                distance = ((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)**0.5
                if distance < tracking_distance and distance < min_distance:
                    min_distance = distance
                    matched_ball_id = ball_id
        
        if matched_ball_id is not None:
            # 更新現有球
            active_balls[matched_ball_id]['positions'].append(pos)
            active_balls[matched_ball_id]['last_seen'] = current_time
        else:
            # 創建新球
            active_balls[next_ball_id] = {
                'entry_time': current_time,
                'positions': [pos],
                'last_seen': current_time,
                'entry_recorded': False
            }
            next_ball_id += 1
    
    return next_ball_id


def _is_ball_exited(positions, edges):
    """檢查球是否真的離開了畫面"""
    if len(positions) < 3:
        return False
    
    # 檢查最後幾個位置是否都在邊緣
    recent_positions = positions[-3:]
    for pos in recent_positions:
        x, y = pos
        if not (x < edges['left'] or x > edges['right'] or 
                y < edges['top'] or y > edges['bottom']):
            return False
    return True


def merge_quick_reentry_segments(ball_entries, ball_exits, gap_threshold=0.4, max_combined_duration=3.5):
    """將短時間內再次進入畫面的球片段合併，避免同一球被拆成多段"""

    if not ball_entries or not ball_exits or len(ball_entries) != len(ball_exits):
        return ball_entries, ball_exits, []

    merged_entries = [ball_entries[0]]
    merged_exits = [ball_exits[0]]
    merge_events = []

    for idx in range(1, len(ball_entries)):
        entry_time = ball_entries[idx]
        exit_time = ball_exits[idx]
        gap = entry_time - merged_exits[-1]
        combined_exit = max(merged_exits[-1], exit_time)
        combined_duration = combined_exit - merged_entries[-1]

        if gap <= gap_threshold and combined_duration <= max_combined_duration:
            merge_events.append({
                "from_segment": len(merged_entries),
                "merged_segment": idx + 1,
                "gap": gap,
                "new_exit": combined_exit
            })
            merged_exits[-1] = combined_exit
        else:
            merged_entries.append(entry_time)
            merged_exits.append(exit_time)

    return merged_entries, merged_exits, merge_events


def segment_video_dynamic(video_path, ball_entries, ball_exits, output_folder, 
                         name, angle, preview_start_time=-0.2):
    """
    動態分割影片，根據球進入和出場時間點創建片段
    支援多球分割
    """
    print(f"✂️ 開始動態分割影片: {Path(video_path).name}")
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    segments_created = []
    
    if not ball_entries:
        print("⚠️ 沒有找到球進入時間點，跳過分割")
        return segments_created
    
    # 確保進入點和出場點數量匹配
    # 確保所有球都有對應的出場時間
    original_exits_count = len(ball_exits)
    
    if len(ball_exits) < len(ball_entries):
        # 載入動態設定
        config = load_segmentation_config()
        seg_cfg = config.get("segmentation", {})
        next_ball_offset = seg_cfg.get("next_ball_offset", 1.2)
        
        # 如果出場點不足，使用智能補充邏輯
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        default_segment_duration = 2.0  # 預設片段長度2秒
        
        # 從第一個缺失的出場點開始補充
        missing_exits = len(ball_entries) - len(ball_exits)
        print(f"   ⚠️ 缺少 {missing_exits} 個出場時間，進行智能補充 (offset={next_ball_offset}s)...")
        
        # 重新構建完整的出場時間列表
        complete_exits = []
        
        for i, entry_time in enumerate(ball_entries):
            if i < original_exits_count:
                # 使用原有的出場時間，但檢查是否合理
                original_exit = ball_exits[i]
                duration = original_exit - entry_time
                
                if duration > 4.0:  # 如果原始出場時間導致片段過長
                    # 計算智能出場時間
                    if i + 1 < len(ball_entries):  # 如果有下一顆球
                        next_entry_time = ball_entries[i + 1]
                        smart_exit = next_entry_time - next_ball_offset
                        corrected_exit = max(entry_time + 0.5, min(smart_exit, video_duration))
                        print(f"   🔧 球 {i+1} 原始出場時間過晚 ({original_exit:.2f}s)，使用下一球前 {next_ball_offset}s: {corrected_exit:.2f}s")
                    else:
                        corrected_exit = min(entry_time + default_segment_duration, video_duration)
                        print(f"   🔧 球 {i+1} 原始出場時間過晚 ({original_exit:.2f}s)，修正為: {corrected_exit:.2f}s")
                    complete_exits.append(corrected_exit)
                else:
                    complete_exits.append(original_exit)
            else:
                # 補充缺失的出場時間 - 使用智能邏輯
                if i + 1 < len(ball_entries):  # 如果有下一顆球
                    next_entry_time = ball_entries[i + 1]
                    smart_exit = next_entry_time - next_ball_offset
                    estimated_exit = max(entry_time + 0.5, min(smart_exit, video_duration))
                    complete_exits.append(estimated_exit)
                    print(f"   🎯 補充球 {i+1} 出場時間: {estimated_exit:.2f}s (下一球進入前 {next_ball_offset}s)")
                else:
                    # 最後一顆球，使用預設長度或影片結束時間
                    estimated_exit = min(entry_time + default_segment_duration, video_duration)
                    complete_exits.append(estimated_exit)
                    print(f"   🎯 補充球 {i+1} 出場時間: {estimated_exit:.2f}s (最後一球，使用預設長度)")
        
        # 更新出場時間列表
        ball_exits = complete_exits
    
    # 驗證進入和出場時間配對
    merged_entries, merged_exits, merge_events = merge_quick_reentry_segments(ball_entries, ball_exits)
    if merge_events:
        print(f"   🔁 偵測到短暫離開又回到畫面的球，執行自動合併:")
        for event in merge_events:
            gap_ms = abs(event["gap"]) * 1000
            segment_label = f"{event['from_segment']}→{event['merged_segment']}"
            print(f"      • 片段 {segment_label} 間隔 {gap_ms:.0f}ms，延伸結束時間到 {event['new_exit']:.2f}s")
    ball_entries = merged_entries
    ball_exits = merged_exits

    print(f"   📊 分割配對驗證:")
    for i, (entry_time, exit_time) in enumerate(zip(ball_entries, ball_exits)):
        duration = exit_time - entry_time
        print(f"      球#{i+1}: 進入{entry_time:.2f}s → 出場{exit_time:.2f}s (片段{duration:.2f}s)")
        
        # 檢查是否合理
        if duration > 4.0:  # 如果片段超過4秒，仍有問題
            print(f"      ❌ 球#{i+1} 片段時間仍然異常長 ({duration:.2f}s)")
        elif duration < 0.5:  # 如果片段太短
            print(f"      ⚠️ 球#{i+1} 片段時間太短 ({duration:.2f}s)")
        else:
            print(f"      ✅ 球#{i+1} 片段時間正常")
    
    # 載入動態設定（用於 preview 和 exit buffer）
    config = load_segmentation_config()
    seg_cfg = config.get("segmentation", {})
    cfg_preview = seg_cfg.get("preview_start_time", -0.2)
    cfg_buffer = seg_cfg.get("exit_buffer_time", 0.2)
    
    for i, (entry_time, exit_time) in enumerate(zip(ball_entries, ball_exits)):
        segment_num = i + 1
        
        # 計算片段時間範圍
        start_time = max(0, entry_time + cfg_preview)
        end_time = exit_time + cfg_buffer
        duration = end_time - start_time
        
        if duration < 0.5:  # 片段太短，跳過
            print(f"   ⚠️ 片段 {segment_num} 太短 ({duration:.2f}s)，跳過")
            continue
        
        # 輸出檔案名稱
        output_file = output_folder / f"{name}__{segment_num}_{angle}_segment.mp4"
        
        print(f"   📹 創建片段 {segment_num}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        
        # 使用 FFmpeg 分割
        # 檢查是否有本地 FFmpeg
        ffmpeg_path = 'ffmpeg'
        local_ffmpeg = Path("tools/ffmpeg.exe")
        if local_ffmpeg.exists():
            ffmpeg_path = str(local_ffmpeg)
        
        cmd = [
            ffmpeg_path, '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'h264_nvenc',  # GPU編碼
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                # 驗證生成的檔案
                if output_file.exists() and output_file.stat().st_size > 10240:  # 至少 10KB
                    print(f"   ✅ 片段 {segment_num} 創建成功: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)")
                    segments_created.append({
                        'segment_number': segment_num,
                        'file_path': str(output_file),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'entry_time': entry_time,
                        'exit_time': exit_time
                    })
                else:
                    print(f"   ⚠️ GPU分割生成檔案太小，嘗試CPU模式")
                    # 刪除可能損壞的檔案
                    if output_file.exists():
                        output_file.unlink()
                    # 觸發CPU模式重試
                    result.returncode = 1
            
            if result.returncode != 0:
                # 如果GPU失敗，嘗試CPU模式
                print(f"   ⚠️ GPU分割失敗，嘗試CPU模式")
                cmd_cpu = [
                    ffmpeg_path, '-y',
                    '-i', str(video_path),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    str(output_file)
                ]
                
                result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=60)
                if result_cpu.returncode == 0:
                    # 驗證CPU模式生成的檔案
                    if output_file.exists() and output_file.stat().st_size > 10240:  # 至少 10KB
                        print(f"   ✅ 片段 {segment_num} 創建成功: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)")
                        segments_created.append({
                            'segment_number': segment_num,
                            'file_path': str(output_file),
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'entry_time': entry_time,
                            'exit_time': exit_time
                        })
                    else:
                        print(f"   ❌ CPU分割生成檔案太小或不存在")
                        if output_file.exists():
                            output_file.unlink()  # 刪除損壞檔案
                else:
                    print(f"   ❌ 片段 {segment_num} 創建失敗: {result_cpu.stderr}")
                    # 嘗試第三種方法：使用軟體編碼
                    print(f"   🔧 嘗試軟體編碼模式")
                    cmd_soft = [
                        ffmpeg_path, '-y',
                        '-i', str(video_path),
                        '-ss', str(start_time),
                        '-t', str(duration),
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        str(output_file)
                    ]
                    
                    try:
                        result_soft = subprocess.run(cmd_soft, capture_output=True, text=True, timeout=90)
                        if result_soft.returncode == 0 and output_file.exists() and output_file.stat().st_size > 10240:
                            print(f"   ✅ 軟體編碼成功: {output_file.name} ({output_file.stat().st_size / 1024:.1f} KB)")
                            segments_created.append({
                                'segment_number': segment_num,
                                'file_path': str(output_file),
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': duration,
                                'entry_time': entry_time,
                                'exit_time': exit_time
                            })
                        else:
                            print(f"   ❌ 所有分割方法都失敗")
                            if output_file.exists():
                                output_file.unlink()  # 清理損壞檔案
                    except Exception as e:
                        print(f"   ❌ 軟體編碼錯誤: {e}")
                        if output_file.exists():
                            output_file.unlink()
        except subprocess.TimeoutExpired:
            print(f"   ❌ 片段 {segment_num} 創建超時")
            if output_file.exists():
                output_file.unlink()  # 清理可能損壞的檔案
        except Exception as e:
            print(f"   ❌ 片段 {segment_num} 創建錯誤: {e}")
            if output_file.exists():
                output_file.unlink()  # 清理可能損壞的檔案
    
    print(f"✅ 動態分割完成: 創建了 {len(segments_created)} 個片段")
    return segments_created

def process_video_segmentation(video_side, video_45, yolo_tennis_ball_model, name, output_folder,
                              ball_entry_direction="right", confidence_threshold=0.5):
    """
    處理影片分割的完整流程 - 多球分析版本
    每顆球會產生獨立的分析結果
    """
    print("\n📹 步驟：影片自動分割處理...")
    print("=" * 50)
    
    output_folder = Path(output_folder)
    segments_folder = output_folder / "segments"
    segments_folder.mkdir(parents=True, exist_ok=True)
    
    # 設定分割參數
    if ball_entry_direction == "right":
        detection_area = "right_upper_two_thirds"  # 右邊上方2/3區域
    else:
        detection_area = "left_upper_two_thirds"   # 左邊上方2/3區域
    
    enable_exit_detection = True  # 啟用球出場偵測
    exit_timeout = 1.5  # 出場等待時間1.5秒
    
    print(f"   🎯 分割設定:")
    print(f"      球進入方向: {'右邊' if ball_entry_direction == 'right' else '左邊'}")
    print(f"      偵測區域: {detection_area}")
    print(f"      球出場偵測: 啟用")
    print(f"      出場等待時間: {exit_timeout} 秒")

    segmentation_results = {
        "side_segments": [],
        "deg45_segments": [],
        "ball_pairs": [],  # 新增：對齊的球對
        "parameters": {
            "detection_area": detection_area,
            "enable_exit_detection": enable_exit_detection,
            "exit_timeout": exit_timeout,
            "confidence_threshold": confidence_threshold,
            "ball_entry_direction": ball_entry_direction
        }
    }
    
    side_ball_data = []
    deg45_ball_data = []
    
    # 處理側面影片
    if video_side:
        print(f"\n🎥 處理側面影片: {Path(video_side).name}")
        try:
            ball_entries, ball_exits = detect_ball_entries_optimized(
                video_side, yolo_tennis_ball_model, confidence_threshold,
                detection_area, enable_exit_detection, exit_timeout, ball_entry_direction
            )
            
            side_segments = segment_video_dynamic(
                video_side, ball_entries, ball_exits, segments_folder,
                name, "side", preview_start_time=-0.2
            )
            
            segmentation_results["side_segments"] = side_segments

            if len(ball_entries) == len(side_segments) == len(ball_exits):
                side_ball_data = [
                    (entry, exit, segment)
                    for entry, exit, segment in zip(ball_entries, ball_exits, side_segments)
                    if segment
                ]
            else:
                if side_segments:
                    print(
                        f"   ⚠️ 偵測到側面進入/出場統計與片段數量不一致 (entries={len(ball_entries)}, exits={len(ball_exits)}, segments={len(side_segments)})，改用片段時間資料"
                    )
                side_ball_data = [
                    (segment.get("entry_time"), segment.get("exit_time"), segment)
                    for segment in side_segments
                    if segment
                ]
            
        except Exception as e:
            print(f"❌ 側面影片分割失敗: {e}")
    
    # 處理45度影片
    if video_45:
        print(f"\n🎥 處理45度影片: {Path(video_45).name}")
        try:
            ball_entries, ball_exits = detect_ball_entries_optimized(
                video_45, yolo_tennis_ball_model, confidence_threshold,
                detection_area, enable_exit_detection, exit_timeout, ball_entry_direction
            )
            
            deg45_segments = segment_video_dynamic(
                video_45, ball_entries, ball_exits, segments_folder,
                name, "45", preview_start_time=-0.2
            )
            
            segmentation_results["deg45_segments"] = deg45_segments

            if len(ball_entries) == len(deg45_segments) == len(ball_exits):
                deg45_ball_data = [
                    (entry, exit, segment)
                    for entry, exit, segment in zip(ball_entries, ball_exits, deg45_segments)
                    if segment
                ]
            else:
                if deg45_segments:
                    print(
                        f"   ⚠️ 偵測到45度進入/出場統計與片段數量不一致 (entries={len(ball_entries)}, exits={len(ball_exits)}, segments={len(deg45_segments)})，改用片段時間資料"
                    )
                deg45_ball_data = [
                    (segment.get("entry_time"), segment.get("exit_time"), segment)
                    for segment in deg45_segments
                    if segment
                ]
            
        except Exception as e:
            print(f"❌ 45度影片分割失敗: {e}")
    
    # 球對對齊處理
    print(f"\n🔄 進行球對對齊處理...")
    ball_pairs = align_ball_segments(side_ball_data, deg45_ball_data, name)
    segmentation_results["ball_pairs"] = ball_pairs
    
    # 保存分割結果
    results_file = output_folder / f"{name}__segmentation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(segmentation_results, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
    
    total_segments = len(segmentation_results["side_segments"]) + len(segmentation_results["deg45_segments"])
    total_balls = len(ball_pairs)
    
    print(f"\n✅ 影片分割完成！")
    print(f"   總共創建: {total_segments} 個片段")
    print(f"   側面片段: {len(segmentation_results['side_segments'])} 個")
    print(f"   45度片段: {len(segmentation_results['deg45_segments'])} 個")
    print(f"   對齊球對: {total_balls} 對")
    print(f"   結果保存: {results_file.name}")
    
    return segmentation_results

def create_ball_specific_segments(segmentation_results, output_folder, name):
    """
    將分割片段複製到對應的 trajectory_N 資料夾中
    """
    print("\n📁 將分割片段複製到對應的軌跡資料夾...")
    
    output_folder = Path(output_folder)
    
    for ball_pair in segmentation_results.get("ball_pairs", []):
        ball_number = ball_pair["ball_number"]
        ball_folder = output_folder / f"trajectory_{ball_number}"
        ball_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"   📋 處理第 {ball_number} 顆球的片段...")
        
        # 複製側面片段
        if ball_pair.get("side_data") and ball_pair["side_data"].get("segment"):
            side_segment = ball_pair["side_data"]["segment"]
            if isinstance(side_segment, str):
                source_path = Path(side_segment)
            else:
                source_path = Path(side_segment.get("file_path", ""))
            
            if source_path and source_path.exists():
                target_path = ball_folder / source_path.name
                if source_path != target_path:
                    shutil.copy2(source_path, target_path)
                    # 更新路徑引用
                    if isinstance(side_segment, dict):
                        ball_pair["side_data"]["segment"]["file_path"] = str(target_path)
                    else:
                        ball_pair["side_data"]["segment"] = str(target_path)
                    print(f"      ✅ 側面片段: {source_path.name} → trajectory_{ball_number}/")
        
        # 複製45度片段
        if ball_pair.get("deg45_data") and ball_pair["deg45_data"].get("segment"):
            deg45_segment = ball_pair["deg45_data"]["segment"]
            if isinstance(deg45_segment, str):
                source_path = Path(deg45_segment)
            else:
                source_path = Path(deg45_segment.get("file_path", ""))
            
            if source_path and source_path.exists():
                target_path = ball_folder / source_path.name
                if source_path != target_path:
                    shutil.copy2(source_path, target_path)
                    # 更新路徑引用
                    if isinstance(deg45_segment, dict):
                        ball_pair["deg45_data"]["segment"]["file_path"] = str(target_path)
                    else:
                        ball_pair["deg45_data"]["segment"] = str(target_path)
                    print(f"      ✅ 45度片段: {source_path.name} → trajectory_{ball_number}/")
    
    print("✅ 分割片段複製完成")
    return segmentation_results

def align_ball_segments(side_ball_data, deg45_ball_data, name):
    """
    對齊側面和45度影片的球片段
    基於時間相近性進行配對
    """
    print(f"🔄 開始球對對齊...")
    print(f"   側面球數: {len(side_ball_data)}")
    print(f"   45度球數: {len(deg45_ball_data)}")
    
    # 除錯：顯示每個球的進入時間
    print(f"\n   側面球進入時間:")
    for i, (entry, exit, _) in enumerate(side_ball_data, 1):
        print(f"      球{i}: 進入={entry:.2f}s, 離開={exit:.2f}s")
    
    print(f"\n   45度球進入時間:")
    for i, (entry, exit, _) in enumerate(deg45_ball_data, 1):
        print(f"      球{i}: 進入={entry:.2f}s, 離開={exit:.2f}s")
    
    ball_pairs = []
    time_tolerance = 2.0  # 允許的時間差異（秒）
    
    used_deg45_indices = set()
    
    for side_idx, (side_entry, side_exit, side_segment) in enumerate(side_ball_data):
        best_match_idx = None
        best_time_diff = float('inf')
        
        # 找最接近的45度球
        for deg45_idx, (deg45_entry, deg45_exit, deg45_segment) in enumerate(deg45_ball_data):
            if deg45_idx in used_deg45_indices:
                continue
                
            # 計算時間差異
            time_diff = abs(side_entry - deg45_entry)
            
            if time_diff < best_time_diff and time_diff <= time_tolerance:
                best_time_diff = time_diff
                best_match_idx = deg45_idx
        
        # 創建球對
        ball_number = side_idx + 1
        
        if best_match_idx is not None:
            used_deg45_indices.add(best_match_idx)
            deg45_entry, deg45_exit, deg45_segment = deg45_ball_data[best_match_idx]
            
            ball_pair = {
                "ball_number": ball_number,
                "side_data": {
                    "entry_time": side_entry,
                    "exit_time": side_exit,
                    "segment": side_segment
                },
                "deg45_data": {
                    "entry_time": deg45_entry,
                    "exit_time": deg45_exit,
                    "segment": deg45_segment
                },
                "time_difference": best_time_diff,
                "status": "paired"
            }
            
            print(f"   ⚾ 第{ball_number}球: 側面{side_entry:.2f}s ↔ 45度{deg45_entry:.2f}s (差異{best_time_diff:.2f}s)")
        else:
            ball_pair = {
                "ball_number": ball_number,
                "side_data": {
                    "entry_time": side_entry,
                    "exit_time": side_exit,
                    "segment": side_segment
                },
                "deg45_data": None,
                "time_difference": None,
                "status": "unpaired_side_only"
            }
            
            print(f"   ⚾ 第{ball_number}球: 只有側面{side_entry:.2f}s (無對應45度)")
        
        ball_pairs.append(ball_pair)
    
    # 處理未配對的45度球
    for deg45_idx, (deg45_entry, deg45_exit, deg45_segment) in enumerate(deg45_ball_data):
        if deg45_idx not in used_deg45_indices:
            ball_number = len(ball_pairs) + 1
            
            ball_pair = {
                "ball_number": ball_number,
                "side_data": None,
                "deg45_data": {
                    "entry_time": deg45_entry,
                    "exit_time": deg45_exit,
                    "segment": deg45_segment
                },
                "time_difference": None,
                "status": "unpaired_deg45_only"
            }
            
            ball_pairs.append(ball_pair)
            print(f"   ⚾ 第{ball_number}球: 只有45度{deg45_entry:.2f}s (無對應側面)")
    
    print(f"✅ 球對對齊完成: {len(ball_pairs)} 對球")
    return ball_pairs

def processing_trajectory_unified(P1, P2, yolo_pose_model, yolo_tennis_ball_model, yolo_paddle_model,
                                video_side, video_45, knn_dataset, name,
                                ball_entry_direction="right", confidence_threshold=0.5,
                                output_folder=None, segment_videos=True):
    """
    統一輸出管理的完整軌跡處理流程
    支援多球檢測，每顆球會產生獨立的資料夾
    """
    
    if output_folder is None:
        output_folder = Path("trajectory") / f"{name}__trajectory"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timing_results = {}
    start_total = time.perf_counter()
    
    print(f"🎾 開始 {name} 的完整軌跡分析流程")
    print(f"📁 輸出資料夾: {output_folder}")
    print("=" * 60)
    
    # 檢查系統資源
    print("\n🔍 檢查系統資源...")
    clear_all_memory()
    gpu_ok = check_gpu_memory()
    ram_ok = check_system_memory()
    
    if not ram_ok:
        print("⚠️ 系統記憶體不足，將自動使用 CPU 模式")
    
    try:
        # 步驟0：影片自動分割（如果啟用）
        segmentation_results = None
        if segment_videos:
            print(f"\n📹 步驟0：影片自動分割...")
            start_segment = time.perf_counter()
            
            segmentation_results = process_video_segmentation(
                video_side, video_45, yolo_tennis_ball_model, name, output_folder,
                ball_entry_direction, confidence_threshold
            )
            
            timing_results['影片自動分割'] = time.perf_counter() - start_segment
            print(f"✅ 影片分割完成，耗時：{timing_results['影片自動分割']:.4f} 秒")
            
            clear_all_memory()
        else:
            print("\n⚠️ 影片分割功能已停用（FFmpeg 不可用或手動停用）")
        
        # 使用完整的處理流程 - 多球分析
        if segmentation_results and len(segmentation_results.get("ball_pairs", [])) > 0:
            # 多球處理流程
            success = process_multiple_balls(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model,
                start_total_time=start_total
            )
        else:
            # 單球或未分割處理流程
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model
            )
        
        if success:
            total_time = time.perf_counter() - start_total
            print('\n' + '=' * 60)
            print(f"🎯 {name} 的軌跡分析完成！")
            print('=' * 60)
            print("⏱️ 執行時間統計:")
            print('-' * 60)
            for step, t in timing_results.items():
                print(f"{step:.<30} {t:>10.4f} 秒")
            print('-' * 60)
            print(f"{'總執行時間':.<30} {total_time:>10.4f} 秒")
            print('=' * 60)
            
            # 生成處理摘要
            generate_processing_summary(output_folder, name, timing_results, total_time)
            
        return success
        
    except Exception as e:
        print(f"\n💥 處理過程發生錯誤: {e}")
        
        # 記錄錯誤到日誌
        error_log = output_folder / "logs" / "processing_error.log"
        error_log.parent.mkdir(exist_ok=True)
        
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"錯誤時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用者: {name}\n")
            f.write(f"錯誤訊息: {str(e)}\n")
            f.write(f"輸入影片: {video_side}, {video_45}\n")
            
        return False

def process_multiple_balls(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                          video_side, video_45, knn_dataset, 
                          name, output_folder, timing_results, segmentation_results, paddle_model=None,
                          start_total_time=None):
    """
    處理多球分析 - 為每個球對創建獨立的分析資料夾
    
    Args:
        P1, P2: 校正參數
        yolo_pose_model, yolo_tennis_ball_model: YOLO模型
        video_side, video_45: 影片路徑
        knn_dataset: KNN數據集
        name: 使用者名稱
        output_folder: 主輸出資料夾
        timing_results: 時間記錄
        segmentation_results: 分割結果包含ball_pairs
        start_total_time: 總流程開始時間 (用於計算第一顆球完成時間)
    
    Returns:
        bool: 處理是否成功
    """
    print(f"\n開始多球分析處理 - {name}")
    print(f"偵測到 {len(segmentation_results['ball_pairs'])} 個球對")
    
    # 創建球特定的分割片段
    segmentation_results = create_ball_specific_segments(segmentation_results, output_folder, name)
    
    ball_pairs = segmentation_results["ball_pairs"]
    overall_success = True
    
    for i, ball_pair in enumerate(ball_pairs):
        ball_number = ball_pair["ball_number"]
        print(f"\n處理第 {ball_number} 顆球...")
        
        # 創建該球的專屬資料夾結構
        ball_folder = os.path.join(output_folder, f"trajectory_{ball_number}")
        os.makedirs(ball_folder, exist_ok=True)
        
        # 為該球創建個別的segmentation_results
        ball_segmentation = {
            "side_segments": [ball_pair["side_data"]] if ball_pair["side_data"] else [],
            "deg45_segments": [ball_pair["deg45_data"]] if ball_pair["deg45_data"] else [],
            "ball_pairs": [ball_pair]  # 保持單一球對的結構
        }
        
        # 處理該球對
        try:
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, ball_folder, timing_results, ball_segmentation, paddle_model
            )
            
            if success:
                print(f"✅ 第 {ball_number} 顆球處理完成")
                # 記錄第一顆球完成時間
                if i == 0:
                    if start_total_time is not None:
                        first_ball_time = time.perf_counter() - start_total_time
                        timing_results['第一顆球執行完成'] = first_ball_time
                        print(f"⏱️ 第一顆球執行完成時間: {first_ball_time:.4f} 秒")
                    
                    # === 新增：第一球完成時立即建立標記檔案，讓前端能即時通知 ===
                    try:
                        ready_file_path = os.path.join(ball_folder, "ready.txt")
                        with open(ready_file_path, "w", encoding='utf-8') as f:
                            f.write(f"First ball ready at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"🚩 [第一球完成標記] 已建立：{ready_file_path}")
                    except Exception as e:
                        print(f"⚠️ 無法建立第一球標記檔案: {e}")
            else:
                print(f"⚠️ 第 {ball_number} 顆球處理有部分問題，但已完成可執行的步驟")
                # 不將 overall_success 設為 False，允許繼續處理下一顆球
                
        except Exception as e:
            print(f"❌ 第 {ball_number} 顆球處理發生錯誤: {str(e)}")
            print(f"⚠️ 跳過第 {ball_number} 顆球，繼續處理下一顆...")
            import traceback
            traceback.print_exc()
            # 不將 overall_success 設為 False，允許繼續處理下一顆球
    
    if overall_success:
        print(f"\n🎾 所有球對分析完成！共處理 {len(ball_pairs)} 個球對")
    else:
        print(f"\n⚠️ 部分球對處理失敗")
    
    return overall_success


def process_single_video_set(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                           video_side, video_45, knn_dataset, 
                           name, output_folder, timing_results, segmentation_results=None, paddle_model=None):
    """處理單組影片的完整流程"""
    try:
        # 從output_folder推導球號
        output_folder_path = Path(output_folder)
        folder_name = output_folder_path.name
        if folder_name.startswith("trajectory_"):
            ball_number = folder_name.split("_")[-1]
            segment_name = f"{name}__{ball_number}"
        else:
            segment_name = f"{name}__1"
        
        # 匯入原本的處理模組
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
        from trajectory_gpt_single_feedback import generate_feedback_data_only
        
        # 確定要使用的影片路徑
        actual_video_side = video_side
        actual_video_45 = video_45
        
        # 如果有分割結果，使用分割後的片段
        if segmentation_results and segmentation_results.get("ball_pairs"):
            ball_pair = segmentation_results["ball_pairs"][0]  # 取第一個球對
            
            if ball_pair.get("side_data") and ball_pair["side_data"].get("segment"):
                segment_value = ball_pair["side_data"]["segment"]
                if isinstance(segment_value, dict):
                    segment_path = segment_value.get("file_path")
                else:
                    segment_path = segment_value
                if segment_path:
                    if not os.path.isabs(segment_path):
                        actual_video_side = os.path.abspath(segment_path)
                    else:
                        actual_video_side = segment_path
                    print(f"🎬 使用側面分割片段: {os.path.basename(actual_video_side)}")
                
            if ball_pair.get("deg45_data") and ball_pair["deg45_data"].get("segment"):
                segment_value = ball_pair["deg45_data"]["segment"]
                if isinstance(segment_value, dict):
                    segment_path = segment_value.get("file_path")
                else:
                    segment_path = segment_value
                if segment_path:
                    if not os.path.isabs(segment_path):
                        actual_video_45 = os.path.abspath(segment_path)
                    else:
                        actual_video_45 = segment_path
                    print(f"🎬 使用45度分割片段: {os.path.basename(actual_video_45)}")
        
        # 顯示分割結果摘要
        if segmentation_results:
            print(f"\n📊 影片分割摘要:")
            print(f"   側面片段: {len(segmentation_results['side_segments'])} 個")
            print(f"   45度片段: {len(segmentation_results['deg45_segments'])} 個")
            if segmentation_results.get('parameters'):
                print(f"   偵測範圍: {segmentation_results['parameters']['detection_area']}")
                print(f"   出場等待時間: {segmentation_results['parameters']['exit_timeout']} 秒")
        
        # 步驟1：分析2D軌跡
        print("\n步驟1：分析2D軌跡...")
        start = time.perf_counter()
        
        # 修改為保存到對應資料夾
        trajectory_side = analyze_trajectory_with_output_folder(yolo_pose_model, yolo_tennis_ball_model, actual_video_side, 28, output_folder, paddle_model)
        trajectory_45 = analyze_trajectory_with_output_folder(yolo_pose_model, yolo_tennis_ball_model, actual_video_45, 28, output_folder, paddle_model)
        
        timing_results['2D軌跡分析'] = time.perf_counter() - start
        print(f"✅ 2D軌跡分析完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")
        
        clear_all_memory()

        # 步驟2：2D軌跡平滑處理
        print("\n步驟2：2D軌跡平滑處理...")
        start = time.perf_counter()
        
        # 修改為保存到對應資料夾
        trajectory_side_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_side, output_folder)
        trajectory_45_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_45, output_folder)
        
        timing_results['2D平滑處理'] = time.perf_counter() - start
        print(f"✅ 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")
        
        clear_all_memory()

        # 步驟3：影片處理
        print("\n步驟3：影片物件偵測處理...")
        print("⚠️ 注意：此步驟可能消耗大量記憶體，依序處理以節省資源...")
        start = time.perf_counter()
        
        # 依序處理影片以節省記憶體，並直接保存到對應資料夾
        print("📹 處理側面影片...")
        video_side_processed = process_video_with_output_folder(
            actual_video_side, 
            output_folder,
            ball_model=yolo_tennis_ball_model,
            pose_model=yolo_pose_model,
            paddle_model=paddle_model,
            json_path=trajectory_side
        )
        clear_all_memory()
        
        print("📹 處理45度影片...")
        video_45_processed = process_video_with_output_folder(
            actual_video_45, 
            output_folder,
            ball_model=yolo_tennis_ball_model,
            pose_model=yolo_pose_model,
            paddle_model=paddle_model,
            json_path=trajectory_45
        )
        clear_all_memory()
        
        # 顯示處理結果
        if video_side_processed:
            print(f"📹 側面處理影片已保存: {Path(video_side_processed).name}")
        if video_45_processed:
            print(f"📹 45度處理影片已保存: {Path(video_45_processed).name}")
        
        timing_results['影片處理'] = time.perf_counter() - start
        print(f"✅ 影片處理完成，耗時：{timing_results['影片處理']:.4f} 秒")

        # 步驟4：影片同步
        print("\n步驟4：同步影片...")
        start = time.perf_counter()
        
        synchronize_videos(video_side_processed, video_45_processed, 
                          trajectory_side_smoothing, trajectory_45_smoothing)
        
        timing_results['影片同步'] = time.perf_counter() - start
        print(f"✅ 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

        # 步驟5：合併影片
        print("\n步驟5：合併影片...")
        start = time.perf_counter()
        
        merged_video = combine_videos_ffmpeg(video_45_processed, video_side_processed)
        
        # 移動合併後的影片到對應資料夾
        if merged_video and Path(merged_video).exists():
            final_merged_path = Path(output_folder) / f"{segment_name}_full_video.mp4"
            shutil.move(merged_video, final_merged_path)
            print(f"📹 合併影片已移動到: {final_merged_path.name}")
        
        timing_results['影片合併'] = time.perf_counter() - start
        print(f"✅ 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

        # 步驟6：軌跡同步
        print("\n步驟6：同步軌跡...")
        start = time.perf_counter()
        
        sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
        
        timing_results['軌跡同步'] = time.perf_counter() - start
        print(f"✅ 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

        # 步驟7：3D軌跡分析
        print("\n步驟7：計算3D軌跡...")
        start = time.perf_counter()
        
        trajectory_3d_path = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
        
        # 保存3D軌跡到對應資料夾（從原始位置移動）
        if trajectory_3d_path and Path(trajectory_3d_path).exists():
            source_path = Path(trajectory_3d_path)
            target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory).json"
            
            if source_path != target_path:
                shutil.move(str(source_path), str(target_path))
                trajectory_3d_path = str(target_path)
        
        timing_results['3D軌跡分析'] = time.perf_counter() - start
        print(f"✅ 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

        # 步驟8：3D軌跡平滑處理
        print("\n步驟8：3D軌跡平滑處理...")
        start = time.perf_counter()
        
        # 使用檔案路徑進行平滑處理
        trajectory_3d_smoothing_path = smooth_3D_trajectory(trajectory_3d_path)
        
        # 移動平滑結果到對應資料夾
        if trajectory_3d_smoothing_path and Path(trajectory_3d_smoothing_path).exists():
            source_path = Path(trajectory_3d_smoothing_path)
            target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory_smoothed).json"
            
            if source_path != target_path:
                shutil.move(str(source_path), str(target_path))
                trajectory_3d_smoothing_path = str(target_path)
        
        timing_results['3D平滑處理'] = time.perf_counter() - start
        print(f"✅ 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

        # 步驟9：有效擊球範圍判斷
        print("\n步驟9：判斷有效擊球範圍...")
        start = time.perf_counter()
        
        start_frame, end_frame = find_range(trajectory_side_smoothing)
        
        # 使用檔案路徑進行範圍擷取（extract_frames 期待檔案路徑）
        trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing_path, start_frame, end_frame)
        
        # 移動擊球範圍軌跡到對應資料夾
        if trajectory_3d_swing_range and Path(trajectory_3d_swing_range).exists():
            source_path = Path(trajectory_3d_swing_range)
            target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory_smoothed)_only_swing.json"
            
            if source_path != target_path:
                shutil.move(str(source_path), str(target_path))
                trajectory_3d_swing_range = str(target_path)
        
        timing_results['有效擊球範圍判斷'] = time.perf_counter() - start
        print(f"✅ 有效擊球範圍判斷完成，耗時：{timing_results['有效擊球範圍判斷']:.4f} 秒")

        # 步驟10：KNN分析
        print("\n步驟10：KNN分析...")
        start = time.perf_counter()
        
        # 使用3D平滑軌跡檔案路徑進行KNN分析
        trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing_path)
        
        # 保存KNN反饋到對應資料夾
        knn_feedback_path = save_knn_feedback_with_output_folder(trajectory_knn_suggestion, output_folder, segment_name)
        
        timing_results['KNN 分析'] = time.perf_counter() - start
        print(f"✅ KNN分析完成，耗時：{timing_results['KNN 分析']:.4f} 秒")

        # 步驟11：GPT反饋生成（帶錯誤容錯）
        print("\n步驟11：生成GPT反饋...")
        start = time.perf_counter()
        
        try:
            # GPT分析使用檔案路徑
            # trajectory_3d_swing_range 是 JSON 檔案路徑
            # knn_feedback_path 是 KNN 分析結果的 txt 檔案路徑
            trajectory_gpt_suggestion = generate_feedback_data_only(trajectory_3d_swing_range, knn_feedback_path)
            
            # 檢查是否有錯誤標記
            if isinstance(trajectory_gpt_suggestion, dict) and trajectory_gpt_suggestion.get('error', False):
                error_type = trajectory_gpt_suggestion.get('error_type', 'unknown')
                if error_type == 'quota_exceeded':
                    print("⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代")
                else:
                    print(f"⚠️ GPT API 發生錯誤 ({error_type})，已使用 KNN 分析結果作為替代")
            
            # 保存GPT反饋到對應資料夾（即使有錯誤也保存替代結果）
            gpt_feedback_path = save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
            
            timing_results['GPT 反饋生成'] = time.perf_counter() - start
            print(f"✅ GPT反饋生成完成，耗時：{timing_results['GPT 反饋生成']:.4f} 秒")
            
        except Exception as e:
            print(f"⚠️ GPT反饋生成失敗: {e}")
            print("⚠️ 跳過 GPT 步驟，繼續處理...")
            
            # 創建一個簡單的反饋結果
            trajectory_gpt_suggestion = {
                "problem_frame": "N/A",
                "suggestion": "GPT功能暫時無法使用，請參考KNN分析結果",
                "error": True,
                "error_type": "processing_error"
            }
            
            # 嘗試保存錯誤反饋
            try:
                gpt_feedback_path = save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
            except:
                print("⚠️ 無法保存 GPT 反饋檔案，繼續處理...")
            
            timing_results['GPT 反饋生成'] = time.perf_counter() - start

        # === 確保全部檔案都寫入後，建立完成標記檔案 ===
        try:
            ready_file_path = os.path.join(output_folder, "ready.txt")
            with open(ready_file_path, "w", encoding='utf-8') as f:
                f.write(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🚩 [完成標記] 已建立：{ready_file_path}")
        except Exception as e:
            print(f"⚠️ 無法建立標記檔案: {e}")

        return True
        
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def move_processed_videos(video_side_processed, video_45_processed, name, output_folder):
    """移動並重新命名處理後的影片檔案"""
    try:
        output_folder = Path(output_folder)
        
        if video_side_processed and Path(video_side_processed).exists():
            new_name = output_folder / f"{name}__1_side_processed.mp4"
            shutil.move(video_side_processed, new_name)
            print(f"📹 側面處理影片已移動: {new_name.name}")
            
        if video_45_processed and Path(video_45_processed).exists():
            new_name = output_folder / f"{name}__1_45_processed.mp4"
            shutil.move(video_45_processed, new_name)
            print(f"📹 45度處理影片已移動: {new_name.name}")
            
        return True
    except Exception as e:
        print(f"⚠️ 移動處理影片失敗: {e}")
        return False

def generate_processing_summary(output_folder, name, timing_results, total_time):
    """生成處理摘要檔案"""
    try:
        summary = {
            "user_name": name,
            "processing_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_time_seconds": total_time,
            "step_times": timing_results,
            "output_folder": str(output_folder),
            "status": "completed"
        }
        
        summary_file = output_folder / f"{name}__processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
        
        print(f"📊 處理摘要已保存: {summary_file.name}")
        return True
    except Exception as e:
        print(f"⚠️ 生成處理摘要失敗: {e}")
        return False