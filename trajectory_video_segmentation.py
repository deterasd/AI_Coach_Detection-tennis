"""
影片自動分割模組 (Video Segmentation Module)
============================================
從 trajector_processing_unified.py 提取的分割邏輯

功能：
- 偵測球進入/出場時間點 (detect_ball_entries_optimized)
- 動態影片分割 (segment_video_dynamic)
- 球對對齊處理 (align_ball_segments)
- 分割片段管理 (create_ball_specific_segments)

優化：
- 多執行緒讀取 (ThreadedVideoCapture)
- 批次推理 + 稀疏掃描
- 動態跳幀邏輯
- GPU 加速 (FP16)

使用方式：
    from trajectory_video_segmentation import process_video_segmentation
    
    results = process_video_segmentation(
        video_side, video_45, tennis_ball_model, name, output_folder,
        ball_entry_direction="right", confidence_threshold=0.5
    )
"""

import time
import numpy as np
import os
import json
import shutil
import cv2
import subprocess
import threading
import queue
from pathlib import Path


# ============================================
# 設定檔載入
# ============================================
#"segmentation": {
#            "next_ball_offset": 擊球間隔提早結束時間
#            "exit_buffer_time": 出場後緩衝時間
#            "preview_start_time": 球進入前提早開始時間
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


# ============================================
# 多執行緒影片讀取類別
# ============================================

class ThreadedVideoCapture:
    """
    多執行緒影片讀取器
    用於加速影片讀取，避免 I/O 阻塞影響推理速度
    """
    def __init__(self, path, queue_size=256):
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
                time.sleep(0.001)
                
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


# ============================================
# 球追蹤與偵測函數
# ============================================

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
    
    max_tracking_distance = max(200, fps * 8)
    min_distance = float('inf')
    closest_ball_id = None
    
    for ball_id, ball_info in active_balls.items():
        if ball_info['positions']:
            last_pos = ball_info['positions'][-1]
            distance = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
            if distance < min_distance and distance <= max_tracking_distance:
                min_distance = distance
                closest_ball_id = ball_id
    
    if closest_ball_id is not None:
        active_balls[closest_ball_id]['positions'].append(position)
        active_balls[closest_ball_id]['last_seen'] = current_time


def check_ball_exits(active_balls, edges, current_time, exit_timeout):
    """檢查球是否出場"""
    exited_balls = []
    balls_to_remove = []
    
    for ball_id, ball_info in active_balls.items():
        time_since_last_seen = current_time - ball_info['last_seen']
        
        if time_since_last_seen >= exit_timeout:
            if len(ball_info['positions']) >= 2:
                is_exit, reason = is_ball_exit_right_edge(ball_info['positions'], edges)
                if is_exit:
                    exit_time = ball_info['last_seen']
                    exited_balls.append((ball_id, exit_time, reason))
                    balls_to_remove.append(ball_id)
                else:
                    balls_to_remove.append(ball_id)
            else:
                balls_to_remove.append(ball_id)
    
    for ball_id in balls_to_remove:
        del active_balls[ball_id]
    
    return exited_balls


def is_ball_exit_right_edge(positions, edges):
    """檢查是否為右邊出場"""
    if len(positions) < 2:
        return False, "軌跡太短"
    
    recent_positions = positions[-min(8, len(positions)):]
    end_pos = recent_positions[-1]
    right_boundary = edges['right'] - 100
    
    is_at_right_edge = end_pos[0] > right_boundary
    
    if not is_at_right_edge:
        return False, "不在右邊界"
    
    movement_analysis = analyze_movement_trend(recent_positions, edges)
    exit_reasons = []
    
    if movement_analysis['moving_right']:
        exit_reasons.append("向右移動")
    if movement_analysis['from_center']:
        exit_reasons.append("從中央區域出場")
    if movement_analysis['consistently_right']:
        exit_reasons.append("持續在右邊緣")
    if movement_analysis['moving_outward']:
        exit_reasons.append("向邊緣移動")
    
    if len(recent_positions) >= 2:
        x_trend = recent_positions[-1][0] - recent_positions[0][0]
        if x_trend > 10:
            exit_reasons.append(f"右邊界移動 (ΔX: {x_trend:.0f})")
    
    is_exit = len(exit_reasons) > 0
    reason = "; ".join(exit_reasons) if exit_reasons else "無明確出場跡象"
    
    return is_exit, reason


def analyze_movement_trend(positions, edges):
    """分析球的移動趨勢"""
    if len(positions) < 2:
        return {'moving_right': False, 'from_center': False, 'consistently_right': False, 'moving_outward': False}
    
    width = edges['right'] - edges['left']
    center_x_min = edges['left'] + width * 0.25
    center_x_max = edges['right'] - width * 0.25
    right_zone = edges['right'] - width * 0.3
    
    x_start = positions[0][0]
    x_end = positions[-1][0]
    x_trend = x_end - x_start
    
    from_center = center_x_min <= x_start <= center_x_max
    moving_right = x_trend > 10
    consistently_right = all(pos[0] > right_zone for pos in positions[-min(3, len(positions)):])
    moving_outward = moving_right or consistently_right or x_trend > 8
    
    return {
        'moving_right': moving_right,
        'from_center': from_center,
        'consistently_right': consistently_right,
        'moving_outward': moving_outward
    }


# ============================================
# 邊緣偵測函數
# ============================================

def _is_ball_entry_edge(x, y, edges, detection_mode, frame_width, frame_height):
    """
    檢查球是否在進入邊緣區域 - 改進版本
    
    偵測區域設計：
    - 右邊進入: 右邊緣上2/3 + 上邊緣右半邊
    - 左邊進入: 左邊緣上2/3 + 上邊緣左半邊
    """
    two_thirds_height = frame_height * (2/3)
    right_top_band = frame_width * (2/3)
    left_top_band = frame_width * (1/3)
    
    if detection_mode == "right_only":
        right_edge_y_threshold = two_thirds_height
        right_edge_in_zone = (x > edges['right'] and y < right_edge_y_threshold)
        top_edge_in_zone = (y < two_thirds_height and x > right_top_band)
        return right_edge_in_zone or top_edge_in_zone
        
    elif detection_mode == "left_only":
        left_edge_y_threshold = two_thirds_height
        left_edge_in_zone = (x < edges['left'] and y < left_edge_y_threshold)
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
    for detection in detected_balls:
        pos = detection['position']
        matched_ball_id = None
        min_distance = float('inf')
        
        for ball_id, ball_data in active_balls.items():
            if ball_data['positions']:
                last_pos = ball_data['positions'][-1]
                distance = ((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)**0.5
                if distance < tracking_distance and distance < min_distance:
                    min_distance = distance
                    matched_ball_id = ball_id
        
        if matched_ball_id is not None:
            active_balls[matched_ball_id]['positions'].append(pos)
            active_balls[matched_ball_id]['last_seen'] = current_time
        else:
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
    
    recent_positions = positions[-3:]
    for pos in recent_positions:
        x, y = pos
        if not (x < edges['left'] or x > edges['right'] or 
                y < edges['top'] or y > edges['bottom']):
            return False
    return True


# ============================================
# 核心偵測函數 (優化版)
# ============================================

def detect_ball_entries_optimized(video_path, model, confidence_threshold=0.5, 
                                detection_area="right_upper_two_thirds", 
                                enable_exit_detection=True, exit_timeout=1.5,
                                ball_entry_direction="right"):
    """
    優化的球進入偵測，支援多球追蹤和動態分割模式
    
    優化內容：
    - 多執行緒讀取 (ThreadedVideoCapture)
    - 批次推理 (batch_size=32)
    - 稀疏掃描 (不活躍時每4幀偵測一次)
    - 動態跳幀 (發現球後跳過60幀)
    - GPU FP16 加速
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
    
    # 使用配置的乘数计算追踪距离
    config = load_segmentation_config()
    seg_cfg = config.get("segmentation", {})
    frame_opt = seg_cfg.get("frame_optimization", {})
    fps_multiplier = frame_opt.get("fps_tracking_multiplier", 8)
    max_tracking_distance = max(200, fps * fps_multiplier)
    
    print(f"   🎯 球追蹤距離: {max_tracking_distance:.0f}像素 (根據{fps:.1f}FPS調整)")
    
    # 邊緣檢測參數 - 從配置讀取
    config = load_segmentation_config()
    seg_cfg = config.get("segmentation", {})
    ball_det = seg_cfg.get("ball_detection", {})
    
    edge_ratio = ball_det.get("edge_ratio", 0.15)
    max_tracking_distance = ball_det.get("max_tracking_distance", 200)
    exit_timeout_config = ball_det.get("exit_timeout", 1.5)
    
    edges = {
        'left': frame_width * edge_ratio,
        'right': frame_width * (1 - edge_ratio),
        'top': frame_height * edge_ratio,
        'bottom': frame_height * (1 - edge_ratio)
    }
    
    if ball_entry_direction == "right":
        print(f"   偵測範圍: 右邊緣上2/3區域 + 上邊緣右側2/3區域")
    else:
        print(f"   偵測範圍: 左邊緣上2/3區域 + 上邊緣左側2/3區域")
    
    # 初始化變數
    ball_entry_times = []
    ball_exit_times = []
    active_balls = {}
    next_ball_id = 0
    
    # 優化參數 - 從配置文件讀取
    config = load_segmentation_config()
    seg_cfg = config.get("segmentation", {})
    frame_opt = seg_cfg.get("frame_optimization", {})
    
    SKIP_FRAMES_AFTER_FOUND = frame_opt.get("skip_frames_after_found", 60)
    batch_size = frame_opt.get("batch_size", 32)
    sparse_scan_step = frame_opt.get("sparse_scan_step", 4)
    fps_multiplier = frame_opt.get("fps_tracking_multiplier", 8)
    
    frames_to_skip = 0
    frames_batch = []
    
    # 啟動多執行緒讀取
    cap.release() 
    video_stream = ThreadedVideoCapture(video_path).start()
    time.sleep(0.5)
    
    frame_count = 0
    while video_stream.running() and frame_count < total_frames:
        if video_stream.q.empty():
            time.sleep(0.001)
            continue
            
        frame = video_stream.read()
        if frame is None:
            break
            
        # 激進跳幀邏輯
        if frames_to_skip > 0:
            frames_to_skip -= 1
            frame_count += 1
            continue
            
        frames_batch.append(frame)
        frame_count += 1
        
        # 當批次滿了或是最後一幀時進行推論
        if len(frames_batch) == batch_size or frame_count == total_frames:
            need_full_inference = True
            
            # 稀疏掃描優化
            if not active_balls and len(frames_batch) >= 4:
                skip_step = 4
                sparse_indices = list(range(0, len(frames_batch), skip_step))
                sparse_frames = [frames_batch[i] for i in sparse_indices]
                
                sparse_results = model(sparse_frames, verbose=False, half=True)
                
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
                    need_full_inference = False
                    last_frame_idx = frame_count - 1
                    if last_frame_idx % int(fps * 10) < batch_size:
                        progress = (last_frame_idx / total_frames) * 100
                        print(f"   進度: {progress:.1f}% (跳過空白片段)")

            # 完整推論
            if need_full_inference:
                results = model(frames_batch, verbose=False, half=True)
                
                for i, result in enumerate(results):
                    if frames_to_skip > 0:
                        frames_to_skip -= 1
                        continue
                        
                    current_frame_idx = frame_count - len(frames_batch) + i
                    current_time = current_frame_idx / fps
                    
                    position = None
                    confidence = 0.0
                    
                    if result.boxes:
                        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))
                        if float(best_box.conf[0]) >= confidence_threshold:
                            confidence = float(best_box.conf[0])
                            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                            position = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    ball_detected = position is not None
                    
                    in_edge = False
                    if ball_detected:
                        x, y = position
                        detection_mode = "right_only" if ball_entry_direction == "right" else "left_only"
                        in_edge = _is_ball_entry_edge(x, y, edges, detection_mode, frame_width, frame_height)
                    
                    if ball_detected:
                        if in_edge and not active_balls:
                            active_balls[next_ball_id] = {
                                'entry_time': current_time,
                                'positions': [position],
                                'last_seen': current_time
                            }
                            ball_entry_times.append(current_time)
                            print(f"   ⚾ 球進入時間: {current_time:.2f} 秒 (幀 {current_frame_idx}) - 球#{next_ball_id}")
                            next_ball_id += 1
                            
                            frames_to_skip = SKIP_FRAMES_AFTER_FOUND
                            print(f"   🚀 發現新球！跳過接下來 {frames_to_skip} 幀...")
                            
                        elif active_balls:
                            update_ball_tracking(active_balls, position, current_time, fps)
                    
                    if enable_exit_detection:
                        exited_balls = check_ball_exits(active_balls, edges, current_time, exit_timeout)
                        for ball_id, exit_time, reason in exited_balls:
                            ball_exit_times.append(exit_time)
                            print(f"   🎯 球出場時間: {exit_time:.2f} 秒 - 球#{ball_id}: {reason}")
                    
                    if current_frame_idx % int(fps * 10) == 0:
                        progress = (current_frame_idx / total_frames) * 100
                        print(f"   進度: {progress:.1f}%")
            
            frames_batch = []
            
    video_stream.stop()
    
    # 處理最後一個球
    for ball_id, ball_info in active_balls.items():
        final_exit_time = (total_frames - 1) / fps
        ball_exit_times.append(final_exit_time)
        print(f"   🎯 最後片段延伸到影片結束: {final_exit_time:.2f} 秒")
    
    print(f"✅ 偵測完成: 找到 {len(ball_entry_times)} 個球進入時間點")
    print(f"   總出場點: {len(ball_exit_times)}")
    
    return ball_entry_times, ball_exit_times


# ============================================
# 片段合併與分割
# ============================================

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
    
    original_exits_count = len(ball_exits)
    
    # 補充缺失的出場時間
    if len(ball_exits) < len(ball_entries):
        config = load_segmentation_config()
        seg_cfg = config.get("segmentation", {})
        next_ball_offset = seg_cfg.get("next_ball_offset", 1.2)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        default_segment_duration = 2.0
        missing_exits = len(ball_entries) - len(ball_exits)
        print(f"   ⚠️ 缺少 {missing_exits} 個出場時間，進行智能補充 (offset={next_ball_offset}s)...")
        
        complete_exits = []
        
        for i, entry_time in enumerate(ball_entries):
            if i < original_exits_count:
                original_exit = ball_exits[i]
                duration = original_exit - entry_time
                
                if duration > 4.0:
                    if i + 1 < len(ball_entries):
                        next_entry_time = ball_entries[i + 1]
                        smart_exit = next_entry_time - next_ball_offset
                        corrected_exit = max(entry_time + 0.5, min(smart_exit, video_duration))
                        print(f"   🔧 球 {i+1} 原始出場時間過晚 ({original_exit:.2f}s)，修正為: {corrected_exit:.2f}s")
                    else:
                        corrected_exit = min(entry_time + default_segment_duration, video_duration)
                        print(f"   🔧 球 {i+1} 原始出場時間過晚 ({original_exit:.2f}s)，修正為: {corrected_exit:.2f}s")
                    complete_exits.append(corrected_exit)
                else:
                    complete_exits.append(original_exit)
            else:
                if i + 1 < len(ball_entries):
                    next_entry_time = ball_entries[i + 1]
                    smart_exit = next_entry_time - next_ball_offset
                    estimated_exit = max(entry_time + 0.5, min(smart_exit, video_duration))
                    complete_exits.append(estimated_exit)
                    print(f"   🎯 補充球 {i+1} 出場時間: {estimated_exit:.2f}s (下一球進入前 {next_ball_offset}s)")
                else:
                    estimated_exit = min(entry_time + default_segment_duration, video_duration)
                    complete_exits.append(estimated_exit)
                    print(f"   🎯 補充球 {i+1} 出場時間: {estimated_exit:.2f}s (最後一球，使用預設長度)")
        
        ball_exits = complete_exits
    
    # 合併短暫再進入的片段
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
        
        if duration > 4.0:
            print(f"      ❌ 球#{i+1} 片段時間仍然異常長 ({duration:.2f}s)")
        elif duration < 0.5:
            print(f"      ⚠️ 球#{i+1} 片段時間太短 ({duration:.2f}s)")
        else:
            print(f"      ✅ 球#{i+1} 片段時間正常")
    
    # 載入設定
    config = load_segmentation_config()
    seg_cfg = config.get("segmentation", {})
    cfg_preview = seg_cfg.get("preview_start_time", -0.2)
    cfg_buffer = seg_cfg.get("exit_buffer_time", 0.2)
    
    for i, (entry_time, exit_time) in enumerate(zip(ball_entries, ball_exits)):
        segment_num = i + 1
        
        start_time = max(0, entry_time + cfg_preview)
        end_time = exit_time + cfg_buffer
        duration = end_time - start_time
        
        if duration < 0.5:
            print(f"   ⚠️ 片段 {segment_num} 太短 ({duration:.2f}s)，跳過")
            continue
        
        output_file = output_folder / f"{name}__{segment_num}_{angle}_segment.mp4"
        
        print(f"   📹 創建片段 {segment_num}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        
        # FFmpeg 分割
        ffmpeg_path = 'ffmpeg'
        local_ffmpeg = Path("tools/ffmpeg.exe")
        if local_ffmpeg.exists():
            ffmpeg_path = str(local_ffmpeg)
        
        # 嘗試 GPU 編碼
        cmd = [
            ffmpeg_path, '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 10240:
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
                # GPU 失敗，嘗試 CPU copy 模式
                print(f"   ⚠️ GPU分割失敗，嘗試CPU模式")
                if output_file.exists():
                    output_file.unlink()
                    
                cmd_cpu = [
                    ffmpeg_path, '-y',
                    '-i', str(video_path),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    str(output_file)
                ]
                
                result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=60)
                if result_cpu.returncode == 0 and output_file.exists() and output_file.stat().st_size > 10240:
                    print(f"   ✅ 片段 {segment_num} 創建成功 (CPU): {output_file.name}")
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
                    # 最後嘗試軟體編碼
                    print(f"   🔧 嘗試軟體編碼模式")
                    if output_file.exists():
                        output_file.unlink()
                        
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
                            print(f"   ✅ 軟體編碼成功: {output_file.name}")
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
                                output_file.unlink()
                    except Exception as e:
                        print(f"   ❌ 軟體編碼錯誤: {e}")
                        if output_file.exists():
                            output_file.unlink()
                            
        except subprocess.TimeoutExpired:
            print(f"   ❌ 片段 {segment_num} 創建超時")
            if output_file.exists():
                output_file.unlink()
        except Exception as e:
            print(f"   ❌ 片段 {segment_num} 創建錯誤: {e}")
            if output_file.exists():
                output_file.unlink()
    
    print(f"✅ 動態分割完成: 創建了 {len(segments_created)} 個片段")
    return segments_created


# ============================================
# 球對對齊與片段管理
# ============================================

def align_ball_segments(side_ball_data, deg45_ball_data, name):
    """
    對齊側面和45度影片的球片段
    基於時間相近性進行配對
    """
    print(f"🔄 開始球對對齊...")
    print(f"   側面球數: {len(side_ball_data)}")
    print(f"   45度球數: {len(deg45_ball_data)}")
    
    print(f"\n   側面球進入時間:")
    for i, (entry, exit, _) in enumerate(side_ball_data, 1):
        print(f"      球{i}: 進入={entry:.2f}s, 離開={exit:.2f}s")
    
    print(f"\n   45度球進入時間:")
    for i, (entry, exit, _) in enumerate(deg45_ball_data, 1):
        print(f"      球{i}: 進入={entry:.2f}s, 離開={exit:.2f}s")
    
    ball_pairs = []
    time_tolerance = 2.0
    used_deg45_indices = set()
    
    for side_idx, (side_entry, side_exit, side_segment) in enumerate(side_ball_data):
        best_match_idx = None
        best_time_diff = float('inf')
        
        for deg45_idx, (deg45_entry, deg45_exit, deg45_segment) in enumerate(deg45_ball_data):
            if deg45_idx in used_deg45_indices:
                continue
            time_diff = abs(side_entry - deg45_entry)
            if time_diff < best_time_diff and time_diff <= time_tolerance:
                best_time_diff = time_diff
                best_match_idx = deg45_idx
        
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


def create_ball_specific_segments(segmentation_results, output_folder, name):
    """將分割片段複製到對應的 trajectory_N 資料夾中"""
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
                    if isinstance(deg45_segment, dict):
                        ball_pair["deg45_data"]["segment"]["file_path"] = str(target_path)
                    else:
                        ball_pair["deg45_data"]["segment"] = str(target_path)
                    print(f"      ✅ 45度片段: {source_path.name} → trajectory_{ball_number}/")
    
    print("✅ 分割片段複製完成")
    return segmentation_results


# ============================================
# 主要處理流程
# ============================================

def process_video_segmentation(video_side, video_45, yolo_tennis_ball_model, name, output_folder,
                              ball_entry_direction="right", confidence_threshold=0.5):
    """
    處理影片分割的完整流程 - 多球分析版本
    
    Args:
        video_side: 側面影片路徑
        video_45: 45度影片路徑
        yolo_tennis_ball_model: 網球偵測 YOLO 模型
        name: 輸出名稱前綴
        output_folder: 輸出資料夾
        ball_entry_direction: 球進入方向 ("right" 或 "left")
        confidence_threshold: 偵測信心度閾值
    
    Returns:
        dict: 包含分割結果的字典
    """
    print("\n📹 步驟：影片自動分割處理...")
    print("=" * 50)
    
    output_folder = Path(output_folder)
    segments_folder = output_folder / "segments"
    segments_folder.mkdir(parents=True, exist_ok=True)
    
    if ball_entry_direction == "right":
        detection_area = "right_upper_two_thirds"
    else:
        detection_area = "left_upper_two_thirds"
    
    enable_exit_detection = True
    exit_timeout = 1.5
    
    print(f"   🎯 分割設定:")
    print(f"      球進入方向: {'右邊' if ball_entry_direction == 'right' else '左邊'}")
    print(f"      偵測區域: {detection_area}")
    print(f"      球出場偵測: 啟用")
    print(f"      出場等待時間: {exit_timeout} 秒")

    segmentation_results = {
        "side_segments": [],
        "deg45_segments": [],
        "ball_pairs": [],
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
                    print(f"   ⚠️ 偵測統計與片段數量不一致，改用片段時間資料")
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
                    print(f"   ⚠️ 偵測統計與片段數量不一致，改用片段時間資料")
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
        json.dump(segmentation_results, f, ensure_ascii=False, indent=2)
    
    total_segments = len(segmentation_results["side_segments"]) + len(segmentation_results["deg45_segments"])
    total_balls = len(ball_pairs)
    
    print(f"\n✅ 影片分割完成！")
    print(f"   總共創建: {total_segments} 個片段")
    print(f"   側面片段: {len(segmentation_results['side_segments'])} 個")
    print(f"   45度片段: {len(segmentation_results['deg45_segments'])} 個")
    print(f"   對齊球對: {total_balls} 對")
    print(f"   結果保存: {results_file.name}")
    
    return segmentation_results


# ============================================
# 主程式入口 (測試用)
# ============================================

if __name__ == "__main__":
    print("🎾 影片分割模組測試")
    print("=" * 50)
    print("此模組通常由 trajector_processing_unified.py 呼叫")
    print("若要獨立測試，請使用以下方式：")
    print()
    print("  from trajectory_video_segmentation import process_video_segmentation")
    print("  from ultralytics import YOLO")
    print()
    print("  model = YOLO('model/tennisball_OD_v1.pt')")
    print("  results = process_video_segmentation(")
    print("      'video_side.mp4', 'video_45.mp4',")
    print("      model, 'test_user', 'output_folder'")
    print("  )")
