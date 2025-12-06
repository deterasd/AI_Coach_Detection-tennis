import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import gc

# COCO 預設 17 個關節名稱
body_parts_list = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 球拍四個標記點名稱
paddle_labels = ["Top", "Right", "Left", "Bottom"]

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
    paddle_model_path='model/best-paddlekeypoint.pt',
    OUTPUT_WIDTH=1280,
    OUTPUT_HEIGHT=720,
    skip_frames=1,
    yolo_batch_size=4,
    ball_conf_threshold=0.8,
    paddle_conf_threshold=0.5
):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device_str}")

    # 載入模型
    ball_model = YOLO(ball_model_path)
    pose_model = YOLO(pose_model_path)
    paddle_model = YOLO(paddle_model_path)
    ball_model.model.to(device_str)
    pose_model.model.to(device_str)
    paddle_model.model.to(device_str)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法讀取影片: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30
    print(f"[INFO] FPS={original_fps:.2f}")

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

    print(f"[INFO] 共擷取 {total_frames} 幀，進行 YOLO 推論...")

    # YOLO 推論
    with torch.no_grad():
        pose_results_batch = pose_model.predict(frames_for_infer, verbose=False, device=device_str, batch=yolo_batch_size)
        ball_results_batch = ball_model.predict(frames_for_infer, verbose=False, device=device_str, batch=yolo_batch_size)
        # 降低 paddle 偵測的信心度閾值到 0.1 以提高偵測率
        paddle_results_batch = paddle_model.predict(frames_for_infer, verbose=False, device=device_str, batch=yolo_batch_size, conf=0.1)

    # === 初始化結果容器 ===
    ball_positions = [None] * total_frames
    ball_confidences = [None] * total_frames
    keypoints_per_frame = [None] * total_frames
    keypoints_conf_per_frame = [None] * total_frames
    paddle_keypoints = [None] * total_frames
    paddle_confidences = [None] * total_frames

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

        # --- Paddle Keypoints ---
        paddle_pts, paddle_conf = None, None
        if paddle_result.keypoints is not None and len(paddle_result.keypoints) > 0:
            pts = paddle_result.keypoints.xy[0].cpu().numpy()
            confs = paddle_result.keypoints.conf[0].cpu().numpy()
            if pts.shape[0] >= 4:
                paddle_pts = [(int(x), int(y)) for x, y in pts[:4]]
                paddle_conf = [float(c) for c in confs[:4]]

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

        # --- 畫球與人體 ---
        if ball_pos:
            cv2.circle(frame, ball_pos, 6, (0, 255, 255), -1)
        if kpts:
            for idx, (x, y) in enumerate(kpts):
                color = (0, 0, 255) if idx == 10 else (0, 255, 0)
                cv2.circle(frame, (x, y), 5, color, -1)

        # --- 畫球拍：四點 + 外框 + 對角十字 (X) ---
        if paddle_pts and len(paddle_pts) >= 4:
            for (x, y) in paddle_pts:
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

            # 中心點
            cx, cy = int(np.mean([p[0] for p in paddle_pts])), int(np.mean([p[1] for p in paddle_pts]))
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            # 依中心排序四點，確保對角連線正確
            pts_with_ang = []
            for (x, y) in paddle_pts:
                ang = np.arctan2(y - cy, x - cx)
                pts_with_ang.append(((x, y), ang))
            pts_sorted = [p for (p, _) in sorted(pts_with_ang, key=lambda t: t[1])]

            # 外框
            cv2.polylines(frame, [np.array(pts_sorted, np.int32)], True, (255, 0, 0), 2)

            # 對角線形成十字 (X)
            cv2.line(frame, pts_sorted[0], pts_sorted[2], (0, 255, 255), 2)
            cv2.line(frame, pts_sorted[1], pts_sorted[3], (0, 255, 255), 2)

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
        y_text += 30
        if paddle_pts is not None:
            for j, label in enumerate(paddle_labels):
                if j < len(paddle_pts):
                    pt = paddle_pts[j]
                    conf = paddle_conf[j] if paddle_conf else 0
                    cv2.putText(info_panel, f"{label:<6}: {pt}  Conf={conf:.2f}",
                                (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    y_text += 25
        else:
            cv2.putText(info_panel, "Paddle Status: Not Detected", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y_text += 30

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
