"""
影片自動分割模組
從 trajector_processing_unified.py 提取的分割邏輯
使用 detect_ball_entries_optimized 偵測球進入/出場時間點
使用 segment_video_dynamic 進行動態影片分割
"""

import cv2
import numpy as np
import subprocess
import traceback
from pathlib import Path
from ultralytics import YOLO


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
    
    # 偵測範圍設定
    if ball_entry_direction == "right":
        detection_mode = "right_upper_two_thirds"
    else:
        detection_mode = "left_upper_two_thirds"
    
    # 初始化變數
    ball_entry_times = []
    ball_exit_times = []
    active_balls = {}  # 活躍球追蹤
    next_ball_id = 0
    
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        # 偵測球
        results = model(frame, verbose=False, conf=confidence_threshold)
        
        if results[0].boxes:
            best_box = max(results[0].boxes, key=lambda box: float(box.conf[0]))
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            position = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # 檢查是否在進入邊緣
            if _is_ball_entry_edge(position[0], position[1], edges, detection_mode, frame_width, frame_height):
                # 檢查是否為新球
                is_new_ball = True
                for ball_id, ball_info in active_balls.items():
                    if len(ball_info['positions']) > 0:
                        last_pos = ball_info['positions'][-1]
                        distance = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
                        if distance < max(200, fps * 8):
                            is_new_ball = False
                            break
                
                if is_new_ball:
                    ball_entry_times.append(current_time)
                    active_balls[next_ball_id] = {
                        'entry_time': current_time,
                        'positions': [position],
                        'last_seen': current_time
                    }
                    print(f"   ⚾ 球進入時間: {current_time:.2f} 秒 (幀 {frame_count}) - 球#{next_ball_id}")
                    next_ball_id += 1
            
            # 更新活躍球追蹤
            _update_ball_tracking(active_balls, position, current_time, fps)
        
        # 檢查球出場
        if enable_exit_detection:
            exited_balls = _check_ball_exits(active_balls, edges, current_time, exit_timeout)
            for ball_id, exit_time in exited_balls:
                ball_exit_times.append(exit_time)
        
        # 顯示進度
        if frame_count % 50 == 0:
            print(f"   進度: {frame_count / total_frames * 100:.1f}%")
    
    # 處理最後一個球
    for ball_id, ball_info in active_balls.items():
        if ball_info['entry_time'] not in [t for t, _ in zip(ball_entry_times, ball_exit_times)]:
            ball_exit_times.append(total_frames / fps)
            print(f"   🎯 最後片段延伸到影片結束: {total_frames / fps:.2f} 秒")
    
    cap.release()
    
    print(f"✅ 偵測完成: 找到 {len(ball_entry_times)} 個球進入時間點")
    print(f"   總出場點: {len(ball_exit_times)}")
    
    return ball_entry_times, ball_exit_times


def _is_ball_entry_edge(x, y, edges, detection_mode, frame_width, frame_height):
    """檢查球是否在進入邊緣區域"""
    two_thirds_height = frame_height * (2/3)
    right_top_band = frame_width * (2/3)
    left_top_band = frame_width * (1/3)
    
    if detection_mode == "right_upper_two_thirds":
        # 右邊緣上2/3 + 上邊緣右側2/3
        right_edge = x > edges['right'] and y < two_thirds_height
        top_right_edge = y < edges['top'] and x > right_top_band
        return right_edge or top_right_edge
    
    elif detection_mode == "left_upper_two_thirds":
        # 左邊緣上2/3 + 上邊緣左側1/3
        left_edge = x < edges['left'] and y < two_thirds_height
        top_left_edge = y < edges['top'] and x < left_top_band
        return left_edge or top_left_edge
    
    return False


def _update_ball_tracking(active_balls, position, current_time, fps):
    """更新球追蹤資訊"""
    if not position:
        return
    
    max_tracking_distance = max(200, fps * 8)
    min_distance = float('inf')
    closest_ball_id = None
    
    for ball_id, ball_info in active_balls.items():
        if len(ball_info['positions']) > 0:
            last_pos = ball_info['positions'][-1]
            distance = np.sqrt((position[0] - last_pos[0])**2 + (position[1] - last_pos[1])**2)
            if distance < min_distance and distance < max_tracking_distance:
                min_distance = distance
                closest_ball_id = ball_id
    
    if closest_ball_id is not None:
        active_balls[closest_ball_id]['positions'].append(position)
        active_balls[closest_ball_id]['last_seen'] = current_time


def _check_ball_exits(active_balls, edges, current_time, exit_timeout):
    """檢查球是否出場"""
    exited_balls = []
    balls_to_remove = []
    
    for ball_id, ball_info in active_balls.items():
        time_since_last_seen = current_time - ball_info['last_seen']
        
        if time_since_last_seen > exit_timeout:
            if len(ball_info['positions']) >= 2:
                is_exit, reason = _is_ball_exit_right_edge(ball_info['positions'], edges)
                if is_exit:
                    exited_balls.append((ball_id, ball_info['last_seen']))
                    balls_to_remove.append(ball_id)
            else:
                balls_to_remove.append(ball_id)
    
    for ball_id in balls_to_remove:
        del active_balls[ball_id]
    
    return exited_balls


def _is_ball_exit_right_edge(positions, edges):
    """檢查是否為右邊出場"""
    if len(positions) < 2:
        return False, "軌跡點不足"
    
    recent_positions = positions[-min(8, len(positions)):]
    end_pos = recent_positions[-1]
    right_boundary = edges['right'] - 100
    
    is_at_right_edge = end_pos[0] > right_boundary
    
    if not is_at_right_edge:
        return False, "不在右邊界"
    
    movement_analysis = _analyze_movement_trend(recent_positions, edges)
    exit_reasons = []
    
    if movement_analysis['moving_right']:
        exit_reasons.append("向右移動")
    if movement_analysis['from_center']:
        exit_reasons.append("從中央開始")
    if movement_analysis['consistently_right']:
        exit_reasons.append("持續在右邊")
    if movement_analysis['moving_outward']:
        exit_reasons.append("向外移動")
    
    is_exit = len(exit_reasons) > 0
    reason = "; ".join(exit_reasons) if exit_reasons else "無明確出場跡象"
    
    return is_exit, reason


def _analyze_movement_trend(positions, edges):
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


def segment_video_dynamic(video_path, ball_entries, ball_exits, output_folder, 
                         name, angle, preview_start_time=-0.5):
    """
    動態分割影片，根據球進入和出場時間點創建片段
    支援多球分割
    """
    print(f"✂️ 開始動態分割影片: {Path(video_path).name}")
    
    if not ball_entries:
        print("⚠️ 沒有找到球進入時間點，跳過分割")
        return []
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # 配對驗證
    print(f"   📊 分割配對驗證:")
    segments_info = []
    
    for i, (entry_time, exit_time) in enumerate(zip(ball_entries, ball_exits), 1):
        segment_duration = exit_time - entry_time
        print(f"      球#{i}: 進入{entry_time:.2f}s → 出場{exit_time:.2f}s (片段{segment_duration:.2f}s)")
        
        if segment_duration < 0.5:
            print(f"      ⚠️ 球#{i} 片段過短，跳過")
            continue
        if segment_duration > 5.0:
            print(f"      ⚠️ 球#{i} 片段過長，可能有誤")
        
        segments_info.append({
            'entry': entry_time,
            'exit': exit_time,
            'duration': segment_duration,
            'ball_number': i
        })
        print(f"      ✅ 球#{i} 片段時間正常")
    
    # 執行分割
    created_segments = []
    for segment_info in segments_info:
        ball_num = segment_info['ball_number']
        start_time = max(0, segment_info['entry'] + preview_start_time)
        end_time = min(duration, segment_info['exit'] + 0.1)
        segment_duration = end_time - start_time
        
        output_path = output_folder / f"{name}_{ball_num}_{angle}_segment.mp4"
        
        print(f"   📹 創建片段 {ball_num}: {start_time:.2f}s - {end_time:.2f}s ({segment_duration:.2f}s)")
        
        success = _segment_with_ffmpeg(video_path, output_path, start_time, segment_duration)
        
        if success and output_path.exists():
            file_size = output_path.stat().st_size / 1024
            print(f"   ✅ 片段 {ball_num} 創建成功: {output_path.name} ({file_size:.1f} KB)")
            created_segments.append({
                'ball_number': ball_num,
                'output_path': str(output_path),
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration
            })
        else:
            print(f"   ❌ 片段 {ball_num} 創建失敗")
    
    print(f"✅ 動態分割完成: 創建了 {len(created_segments)} 個片段")
    return created_segments


def _segment_with_ffmpeg(input_path, output_path, start_time, duration):
    """使用 FFmpeg 分割影片"""
    try:
        # 檢查 FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            ffmpeg_cmd = 'ffmpeg'
        except:
            # 嘗試使用本地 ffmpeg.exe
            ffmpeg_cmd = 'ffmpeg.exe'
        
        cmd = [
            ffmpeg_cmd,
            '-y',  # 覆蓋輸出檔案
            '-ss', str(start_time),
            '-i', str(input_path),
            '-t', str(duration),
            '-c:v', 'copy',  # 複製視頻流，不重新編碼
            '-avoid_negative_ts', '1',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ FFmpeg 分割失敗: {e}")
        return False


def process_video_segmentation(video_side, video_45, tennis_ball_model, name, output_folder,
                              ball_entry_direction="right", confidence_threshold=0.5):
    """
    處理影片分割的完整流程
    
    Args:
        video_side: 側面影片路徑
        video_45: 45度影片路徑
        tennis_ball_model: 網球偵測模型
        name: 輸出檔案名稱前綴
        output_folder: 輸出資料夾
        ball_entry_direction: 球進入方向 ("right" 或 "left")
        confidence_threshold: 偵測信心度
    
    Returns:
        dict: 包含分割結果的字典
    """
    print(f"\n📹 步驟：影片自動分割處理...")
    print("=" * 50)
    print(f"   🎯 分割設定:")
    print(f"      球進入方向: {ball_entry_direction}")
    print(f"      偵測區域: {'right_upper_two_thirds' if ball_entry_direction == 'right' else 'left_upper_two_thirds'}")
    print(f"      球出場偵測: 啟用")
    print(f"      出場等待時間: 1.5 秒")
    
    output_folder = Path(output_folder)
    
    # 處理側面影片
    print(f"\n🎥 處理側面影片: {Path(video_side).name}")
    side_entries, side_exits = detect_ball_entries_optimized(
        video_side, tennis_ball_model, confidence_threshold,
        ball_entry_direction=ball_entry_direction,
        enable_exit_detection=True, exit_timeout=1.5
    )
    
    side_output = output_folder / "segments" / "side"
    side_segments = segment_video_dynamic(
        video_side, side_entries, side_exits, 
        side_output, name, "side"
    )
    
    # 處理45度影片
    print(f"\n🎥 處理45度影片: {Path(video_45).name}")
    deg45_entries, deg45_exits = detect_ball_entries_optimized(
        video_45, tennis_ball_model, confidence_threshold,
        ball_entry_direction=ball_entry_direction,
        enable_exit_detection=True, exit_timeout=1.5
    )
    
    deg45_output = output_folder / "segments" / "45deg"
    deg45_segments = segment_video_dynamic(
        video_45, deg45_entries, deg45_exits,
        deg45_output, name, "45"
    )
    
    # 返回結果
    results = {
        'side': {
            'entries': side_entries,
            'exits': side_exits,
            'segments': side_segments
        },
        '45deg': {
            'entries': deg45_entries,
            'exits': deg45_exits,
            'segments': deg45_segments
        }
    }
    
    print(f"\n✅ 影片分割完成！")
    print(f"   總共創建: {len(side_segments) + len(deg45_segments)} 個片段")
    print(f"   側面片段: {len(side_segments)} 個")
    print(f"   45度片段: {len(deg45_segments)} 個")
    
    return results
