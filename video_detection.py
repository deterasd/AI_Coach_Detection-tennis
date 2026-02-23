import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import gc
import json
import os

# COCO 預設 17 個關節名稱
body_parts_list = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 球拍標記點名稱（與B版本及模型輸出順序一致）
paddle_labels = ["Top", "Right", "Left", "Bottom", "Center", "Grip_Top", "Grip_Bottom"]

def resize_frame(frame, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=inter)

def process_video(
    video_path,
    ball_model_path='model/tennisball_OD_v1.pt',
    pose_model_path='model/yolov8n-pose.pt',
    paddle_model_path='model/tennispaddle.pt',
    ball_model=None,
    pose_model=None,
    paddle_model=None,
    OUTPUT_WIDTH=1280,
    OUTPUT_HEIGHT=720,
    skip_frames=1,
    yolo_batch_size=4,
    ball_conf_threshold=0.8,
    paddle_conf_threshold=0.5,
    json_path=None,
    trace_length=60
):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device_str}")

    # 檢查是否使用 JSON
    use_json = False
    trajectory_data = None
    if json_path and os.path.exists(json_path):
        try:
            print(f"[INFO] 發現軌跡檔案，將使用 JSON 資料進行繪圖: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            use_json = True
        except Exception as e:
            print(f"⚠️ 讀取 JSON 失敗，將切換回模型推論: {e}")

    # 載入模型 (如果不使用 JSON)
    if not use_json:
        if ball_model is None:
            ball_model = YOLO(ball_model_path)
            ball_model.model.to(device_str)
        if pose_model is None:
            pose_model = YOLO(pose_model_path)
            pose_model.model.to(device_str)
        if paddle_model is None:
            paddle_model = YOLO(paddle_model_path)
            paddle_model.model.to(device_str)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法讀取影片: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30
    
    # 取得原始影片解析度（用於 JSON 座標縮放）
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] FPS={original_fps:.2f}, 原始解析度={orig_w}x{orig_h}")

    frames_for_output = []
    frames_for_infer = []
    infer_indices = []
    frame_idx = 0

    # 讀取影片
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        resized_frame = resize_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        frames_for_output.append(resized_frame)
        if frame_idx % skip_frames == 0:
            frames_for_infer.append(resized_frame)
            infer_indices.append(frame_idx)
    cap.release()

    total_frames = len(frames_for_output)
    if total_frames == 0:
        print("❌ 無法擷取任何影格。")
        return

    # === 計算 JSON 座標縮放比例 ===
    # trajectory_2D_output.py 使用原始解析度推論，但這裡顯示的是 resize 後的 frame
    # 需要將 JSON 座標從原始解析度映射到輸出解析度
    # resize_frame 使用 width 參數，按寬度等比縮放
    scale_x = OUTPUT_WIDTH / orig_w if orig_w > 0 else 1.0
    scale_y = scale_x  # 等比縮放，x 和 y 使用相同比例
    if use_json:
        print(f"[INFO] JSON 座標縮放比例: scale={scale_x:.4f} ({orig_w}x{orig_h} -> {OUTPUT_WIDTH}x{int(orig_h * scale_x)})")

    # === 初始化結果容器 ===
    ball_positions = [None] * total_frames
    ball_confidences = [None] * total_frames
    keypoints_per_frame = [None] * total_frames
    keypoints_conf_per_frame = [None] * total_frames
    paddle_keypoints = [None] * total_frames
    paddle_confidences = [None] * total_frames

    if use_json:
        print(f"[INFO] 使用 JSON 資料填入結果容器...")
        for frame_data in trajectory_data:
            idx = frame_data.get("frame", 0)
            if idx >= total_frames: continue
            
            # Ball（套用座標縮放）
            ball = frame_data.get("tennis_ball", {})
            if ball and ball.get("x") is not None and ball.get("y") is not None:
                ball_positions[idx] = (int(ball["x"] * scale_x), int(ball["y"] * scale_y))
                ball_confidences[idx] = 1.0 
            
            # Pose（套用座標縮放）
            kpts = []
            has_pose = False
            for part in body_parts_list:
                p_data = frame_data.get(part, {})
                if p_data and p_data.get("x") is not None:
                    kpts.append((int(p_data["x"] * scale_x), int(p_data["y"] * scale_y)))
                    has_pose = True
                else:
                    kpts.append((0, 0))
            
            if has_pose:
                keypoints_per_frame[idx] = kpts
                keypoints_conf_per_frame[idx] = [1.0]*17
            
            # Paddle（套用座標縮放）- 動態讀取 JSON 中實際存在的 paddle 數據
            paddle = frame_data.get("paddle", {})
            if paddle:
                pts = []
                confs = []
                key_map = {"Top": "top", "Right": "right", "Left": "left", "Bottom": "bottom",
                           "Center": "center", "Grip_Top": "grip_top", "Grip_Bottom": "grip_bottom"}
                for label in paddle_labels:
                    key = key_map.get(label, label.lower())
                    pt = paddle.get(key, {})
                    if pt and pt.get("x") is not None and pt.get("y") is not None:
                        pts.append((int(pt["x"] * scale_x), int(pt["y"] * scale_y)))
                        confs.append(pt.get("conf", 1.0) if pt.get("conf") is not None else 1.0)
                if len(pts) >= 4:
                    paddle_keypoints[idx] = pts
                    paddle_confidences[idx] = confs

    else:
        print(f"[INFO] 共擷取 {total_frames} 幀，進行 YOLO 推論...")

        # YOLO 推論 - 降低 batch size 以避免 OOM
        safe_batch_size = 4  # 降低批次大小以節省記憶體
        
        with torch.no_grad():
            pose_results_batch = pose_model.predict(frames_for_infer, verbose=False, device=device_str, batch=safe_batch_size)
            ball_results_batch = ball_model.predict(frames_for_infer, verbose=False, device=device_str, batch=safe_batch_size)
            # 降低 paddle 偵測的信心度閾值到 0.1 以提高偵測率
            paddle_results_batch = paddle_model.predict(frames_for_infer, verbose=False, device=device_str, batch=safe_batch_size, conf=0.1)

        # === 逐幀整理結果 ===
        for i, fidx in enumerate(infer_indices):
            pose_result = pose_results_batch[i]
            ball_result = ball_results_batch[i]
            paddle_result = paddle_results_batch[i]

            # --- Pose ---
            if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
                kpts = pose_result.keypoints.xy[0]
                kpts_xy = [(int(x), int(y)) for x, y in kpts]
                kpts_conf = pose_result.keypoints.conf[0].cpu().numpy()  # (17,)
                kpts_conf = [float(c) for c in kpts_conf]
            else:
                kpts_xy = None
                kpts_conf = None
            # --- Ball ---
            boxes = ball_result.boxes
            ball_pos, ball_conf = None, None
            if boxes is not None and len(boxes) > 0:
                best_box = max(boxes, key=lambda b: b.conf[0])
                if float(best_box.conf[0]) >= ball_conf_threshold:
                    x1, y1, x2, y2 = best_box.xyxy[0]
                    ball_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    ball_conf = float(best_box.conf[0])

            # --- Paddle Keypoints (與B版本一致，讀取所有偵測到的點) ---
            paddle_pts, paddle_conf = None, None
            if paddle_result.keypoints is not None and len(paddle_result.keypoints) > 0:
                pts = paddle_result.keypoints.xy[0].cpu().numpy()
                confs = paddle_result.keypoints.conf[0].cpu().numpy()
                if pts.shape[0] >= 0:
                    paddle_pts = [(int(x), int(y)) for x, y in pts]
                    paddle_conf = [float(c) for c in confs]

            idx_in_list = fidx - 1
            ball_positions[idx_in_list] = ball_pos
            ball_confidences[idx_in_list] = ball_conf
            keypoints_per_frame[idx_in_list] = kpts_xy
            keypoints_conf_per_frame[idx_in_list] = kpts_conf
            paddle_keypoints[idx_in_list] = paddle_pts
            paddle_confidences[idx_in_list] = paddle_conf

    # === 影片輸出設定 ===
    output_path = video_path.replace('.mp4', '_processed.mp4')
    info_panel_width = 400
    out_w, out_h = OUTPUT_WIDTH + info_panel_width, OUTPUT_HEIGHT
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (out_w, out_h))

    # === 畫圖主迴圈 ===
    for i in range(total_frames):
        frame = frames_for_output[i].copy()
        ball_pos = ball_positions[i]
        ball_conf = ball_confidences[i]
        kpts = keypoints_per_frame[i]
        kpt_confs = keypoints_conf_per_frame[i] 

        paddle_pts = paddle_keypoints[i]
        paddle_conf = paddle_confidences[i]

        # === 軌跡繪製 ===
        start_idx = max(0, i - trace_length)
        
        # --- A. 網球軌跡 (綠 -> 紅) ---
        ball_trail_points = []
        for j in range(start_idx, i + 1):
            if ball_positions[j] is not None:
                ball_trail_points.append(ball_positions[j])
        
        if len(ball_trail_points) > 1:
            for j in range(1, len(ball_trail_points)):
                progress = j / len(ball_trail_points)
                color = (0, int(255 * (1 - progress)), int(255 * progress))  # Green->Red
                thickness = int(4 * progress) + 1
                cv2.line(frame, ball_trail_points[j-1], ball_trail_points[j], color, thickness)

        # --- B. 手腕軌跡 (藍 -> 綠) ---
        wrist_trail_points = []
        for j in range(start_idx, i + 1):
            k = keypoints_per_frame[j]
            if k is not None and len(k) > 10:
                rw = k[10]  # Right Wrist
                if rw[0] > 0 and rw[1] > 0:
                    wrist_trail_points.append(rw)
        
        if len(wrist_trail_points) > 1:
            for j in range(1, len(wrist_trail_points)):
                progress = j / len(wrist_trail_points)
                color = (int(255 * (1 - progress)), int(255 * progress), 0)  # Blue->Green
                thickness = int(4 * progress) + 1
                cv2.line(frame, wrist_trail_points[j-1], wrist_trail_points[j], color, thickness)

        # --- 畫球與人體 ---
        if ball_pos:
            cv2.circle(frame, ball_pos, 6, (0, 255, 255), -1)
        if kpts:
            for idx, (x, y) in enumerate(kpts):
                color = (0, 0, 255) if idx == 10 else (0, 255, 0)
                cv2.circle(frame, (x, y), 5, color, -1)

        # --- 畫球拍 (簡化版本，直接繪製原始點) ---
        if paddle_pts and len(paddle_pts) > 0:
            for idx, (px, py) in enumerate(paddle_pts):
                # 過濾掉座標為 (0,0) 或負值的無效點
                if px <= 0 or py <= 0:
                    continue

                # 根據實際 label 設定顏色
                if paddle_conf and idx < len(paddle_conf):
                    label = paddle_labels[idx] if idx < len(paddle_labels) else ""
                else:
                    label = paddle_labels[idx] if idx < len(paddle_labels) else ""
                
                if label == "Center":
                    color = (0, 0, 255)  # 紅色 - 中心
                elif label in ("Grip_Top", "Grip_Bottom"):
                    color = (0, 255, 0)  # 綠色 - 握把
                else:
                    color = (255, 0, 0)  # 藍色 - 邊框 (Top/Right/Left/Bottom)
                
                # 只畫實心圓點
                cv2.circle(frame, (px, py), 6, color, -1)

        # --- 資訊面板 ---
        info_panel = np.ones((out_h, info_panel_width, 3), dtype=np.uint8) * 40
        header_height = 50
        cv2.rectangle(info_panel, (0, 0), (info_panel_width, header_height), (0, 150, 0), -1)
        cv2.putText(info_panel, "Tennis Ball Detection", (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        y_text = header_height + 30
        if ball_pos is not None:
            bx, by = ball_pos
            conf_val = ball_conf if ball_conf is not None else 0.0

            # ⭐ 一行顯示：Ball + 座標 + Conf
            cv2.putText(info_panel,
                        f"Ball: ({bx}, {by})   Conf: {conf_val:.2f}",
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            y_text += 30

        else:
            cv2.putText(info_panel, "Ball: Not Detected",
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            y_text += 30

        # --- 球拍資訊 ---
        y_text += 10
        cv2.putText(info_panel, "Paddle Detection", (10, y_text),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
        y_text += 25
        if paddle_pts is not None:
            for j in range(len(paddle_pts)):
                pt = paddle_pts[j]
                label = paddle_labels[j] if j < len(paddle_labels) else f"Point_{j}"
                # 安全地獲取 conf 值，避免索引越界
                conf = paddle_conf[j] if (paddle_conf and j < len(paddle_conf)) else 0.0
                cv2.putText(info_panel, f"{label:<12}: {pt}  Conf={conf:.2f}",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                y_text += 20
        else:
            cv2.putText(info_panel, "Paddle Status: Not Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            y_text += 25

        # --- 姿勢估計 ---
        pose_header_top = y_text + 10
        pose_header_bottom = pose_header_top + 40
        cv2.rectangle(info_panel, (0, pose_header_top),
                      (info_panel_width, pose_header_bottom), (255, 100, 0), -1)
        cv2.putText(info_panel, "Pose Estimation", (10, pose_header_top + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y_text = pose_header_bottom + 30
        if kpts is not None:
            for idx, part_name in enumerate(body_parts_list):
                if idx < len(kpts):
                    xx, yy = kpts[idx]
                    if kpt_confs is not None and idx < len(kpt_confs):
                        conf_val = kpt_confs[idx]
                        text_line = f"{part_name:<15}: ({xx}, {yy})  Conf={conf_val:.2f}"
                    else:
                        text_line = f"{part_name:<15}: ({xx}, {yy})"

                    cv2.putText(info_panel, text_line, (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                    #cv2.putText(info_panel, f"{part_name:<15}: ({xx}, {yy})", (10, y_text),
                         #       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                    y_text += 22
                    if y_text >= out_h - 10:
                        break
        else:
            cv2.putText(info_panel, "No keypoints found", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        combined_frame = np.hstack((frame, info_panel))
        out.write(combined_frame)

    out.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✅ 輸出完成：{output_path}")
    return output_path


if __name__ == "__main__":
    total_start = time.time()
    video_path = '測試2__1_45_compressed.mp4'
    output_path = process_video(video_path)
    total_end = time.time()
    print(f"===== 程式總耗時: {total_end - total_start:.2f} 秒 =====")
