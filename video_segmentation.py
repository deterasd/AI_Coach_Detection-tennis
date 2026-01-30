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
    
    # === 優化參數 ===
    SKIP_FRAMES_AFTER_FOUND = 60  # 找到球後跳過 60 幀 (約 1 秒)
    SCAN_STEP = 4  # 平常搜尋時每 4 幀檢查一次
    MOTION_THRESHOLD = 50  # 動態偵測閾值 (像素變化量)
    
    # 初始化動態偵測
    prev_gray = None
    roi_mask = _create_roi_mask(frame_width, frame_height, edges, detection_mode)
    
    current_frame_idx = 0
    
    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = current_frame_idx / fps
        
        # === 動態偵測預篩選 (Motion Filter) ===
        # 如果沒有活躍球，先檢查是否有動靜，沒有就跳過 YOLO
        should_run_yolo = True
        
        if not active_balls:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 縮小圖像以加快處理速度
            small_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            
            if prev_gray is not None:
                # 計算差異
                frame_diff = cv2.absdiff(small_gray, prev_gray)
                # 應用 ROI Mask (同樣縮小)
                small_mask = cv2.resize(roi_mask, (small_gray.shape[1], small_gray.shape[0]))
                frame_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=small_mask)
                
                # 二值化並計算變化像素
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(thresh)
                
                if motion_pixels < MOTION_THRESHOLD:
                    should_run_yolo = False
                    # print(f"   💤 靜止畫面 (變動: {motion_pixels}) - 跳過偵測")
            
            prev_gray = small_gray
        
        # 偵測球 - 自動判斷是否使用半精度
        if should_run_yolo:
            is_cuda = next(model.parameters()).is_cuda
            results = model(frame, verbose=False, conf=confidence_threshold, half=is_cuda)
            
            found_new_ball = False
            
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
                        # === 回溯檢查 (Backtracking) ===
                        # 為了提高準確度，當發現球時，往回檢查幾幀以找到精確的進入點
                        actual_entry_time = current_time
                        actual_entry_frame = current_frame_idx
                        
                        # 簡單回溯邏輯：如果我們是跳著找的，嘗試往回找
                        if SCAN_STEP > 1:
                            # 這裡可以實作真正的回溯，讀取前幾幀
                            # 為了示範，我們先標記這是一個優化點
                            pass

                        ball_entry_times.append(actual_entry_time)
                        active_balls[next_ball_id] = {
                            'entry_time': actual_entry_time,
                            'positions': [position],
                            'last_seen': actual_entry_time
                        }
                        print(f"   ⚾ 球進入時間: {actual_entry_time:.2f} 秒 (幀 {actual_entry_frame}) - 球#{next_ball_id}")
                        next_ball_id += 1
                        found_new_ball = True
                
                # 更新活躍球追蹤
                _update_ball_tracking(active_balls, position, current_time, fps)
        else:
            # 如果跳過 YOLO，視為沒有找到新球
            found_new_ball = False
            results = [] # 空結果
        
        # 檢查球出場
        if enable_exit_detection:
            exited_balls = _check_ball_exits(active_balls, edges, current_time, exit_timeout)
            for ball_id, exit_time in exited_balls:
                ball_exit_times.append(exit_time)
        
        # 顯示進度
        if current_frame_idx % 50 == 0:
            print(f"   進度: {current_frame_idx / total_frames * 100:.1f}%")
            
        # === 激進跳幀邏輯 ===
        if found_new_ball:
            print(f"   🚀 發現新球！跳過接下來 {SKIP_FRAMES_AFTER_FOUND} 幀 ({SKIP_FRAMES_AFTER_FOUND/fps:.1f}秒)...")
            current_frame_idx += SKIP_FRAMES_AFTER_FOUND
        else:
            # 如果有活躍球，我們不能跳太快，以免漏掉軌跡或出場
            if active_balls:
                current_frame_idx += 1
            else:
                # 沒有球的時候，可以跳著找
                current_frame_idx += SCAN_STEP

    
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
    """檢查球是否出場，加入高速保底機制"""
    exited_balls = []
    balls_to_remove = []
    
    for ball_id, ball_info in active_balls.items():
        time_since_last_seen = current_time - ball_info['last_seen']
        
        # 如果一段時間沒看到球了
        if time_since_last_seen > exit_timeout:
            # 1. 嘗試偵測是否從邊緣出場
            if len(ball_info['positions']) >= 2:
                is_exit, reason = _is_ball_exit_right_edge(ball_info['positions'], edges)
                if is_exit:
                    print(f"   📊 球#{ball_id} 偵測到邊緣出場 ({reason})")
                    exited_balls.append((ball_id, ball_info['last_seen']))
                else:
                    # 2. 高速保底：如果追蹤丟失但沒有明確出場，且已有進入，則給予保底時長 (預設進場後 3s)
                    fallback_exit = ball_info['entry_time'] + 2.3
                    print(f"   🕒 球#{ball_id} 高速追蹤丟失，套用 3.0s 保底時長")
                    exited_balls.append((ball_id, min(current_time, fallback_exit)))
            else:
                # 偵測點太少，可能只是雜訊
                pass
            
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
                         name, angle, preview_start_time=-0.2):
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
        end_time = min(duration, segment_info['exit'] + 0.2)
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


def _create_roi_mask(width, height, edges, detection_mode):
    """創建 ROI 遮罩，用於動態偵測"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    two_thirds_height = int(height * (2/3))
    right_top_band = int(width * (2/3))
    left_top_band = int(width * (1/3))
    
    if detection_mode == "right_upper_two_thirds":
        # 右邊緣上2/3
        cv2.rectangle(mask, (int(edges['right']), 0), (width, two_thirds_height), 255, -1)
        # 上邊緣右側2/3
        cv2.rectangle(mask, (right_top_band, 0), (width, int(edges['top'])), 255, -1)
        
    elif detection_mode == "left_upper_two_thirds":
        # 左邊緣上2/3
        cv2.rectangle(mask, (0, 0), (int(edges['left']), two_thirds_height), 255, -1)
        # 上邊緣左側1/3
        cv2.rectangle(mask, (0, 0), (left_top_band, int(edges['top'])), 255, -1)
        
    return mask


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
