import cv2
import numpy as np
import time
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# COCO default 17 body parts
body_parts_list = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Paddle keypoint labels (matches model output order)
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

def create_info_panel_template(out_h, info_panel_width):
    """Pre-create static parts of info panel to avoid recreating every frame"""
    info_panel = np.ones((out_h, info_panel_width, 3), dtype=np.uint8) * 40
    header_height = 50
    cv2.rectangle(info_panel, (0, 0), (info_panel_width, header_height), (0, 150, 0), -1)
    cv2.putText(info_panel, "Tennis Ball Detection", (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return info_panel, header_height

def draw_info_panel(template, header_height, ball_pos, ball_conf, paddle_pts, paddle_conf, kpts, kpt_confs, out_h, info_panel_width):
    """Fast info panel drawing by cloning template and adding dynamic content"""
    info_panel = template.copy()

    y_text = header_height + 30
    # Ball info
    if ball_pos is not None:
        bx, by = ball_pos
        conf_val = ball_conf if ball_conf is not None else 0.0
        cv2.putText(info_panel, f"Ball: ({bx}, {by})   Conf: {conf_val:.2f}",
                    (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_text += 30
    else:
        cv2.putText(info_panel, "Ball: Not Detected", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_text += 30

    # Paddle info
    y_text += 10
    cv2.putText(info_panel, "Paddle Detection", (10, y_text),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
    y_text += 25
    if paddle_pts is not None:
        for j in range(len(paddle_pts)):
            pt = paddle_pts[j]
            label = paddle_labels[j] if j < len(paddle_labels) else f"Point_{j}"
            conf = paddle_conf[j] if (paddle_conf and j < len(paddle_conf)) else 0.0
            cv2.putText(info_panel, f"{label:<12}: {pt}  Conf={conf:.2f}",
                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += 20
    else:
        cv2.putText(info_panel, "Paddle Status: Not Detected", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_text += 25

    # Pose info
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
                y_text += 22
                if y_text >= out_h - 10:
                    break
    else:
        cv2.putText(info_panel, "No keypoints found", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return info_panel

def process_frame(frame_data):
    """Process a single frame - designed for parallel execution"""
    frame_idx, frame, ball_pos, kpts, paddle_pts, paddle_conf, ball_conf, kpt_confs, \
    trace_length, ball_positions, keypoints_per_frame, info_template, header_height, out_h, info_panel_width = frame_data

    frame = frame.copy()

    # Draw trajectories
    start_idx = 0

    # Ball trajectory (green -> red)
    ball_trail_points = []
    for j in range(start_idx, frame_idx + 1):
        if ball_positions[j] is not None:
            ball_trail_points.append(ball_positions[j])

    if len(ball_trail_points) > 1:
        num_points = len(ball_trail_points)
        for j in range(1, num_points):
            progress = j / num_points
            color = (0, int(255 * (1 - progress)), int(255 * progress))
            cv2.line(frame, ball_trail_points[j-1], ball_trail_points[j], color, 3)

    # Wrist trajectory (blue -> green)
    wrist_trail_points = []
    for j in range(start_idx, frame_idx + 1):
        k = keypoints_per_frame[j]
        if k is not None and len(k) > 10:
            rw = k[10]  # Right Wrist
            if rw[0] > 0 and rw[1] > 0:
                wrist_trail_points.append(rw)

    if len(wrist_trail_points) > 1:
        num_points = len(wrist_trail_points)
        for j in range(1, num_points):
            progress = j / num_points
            color = (int(255 * (1 - progress)), int(255 * progress), 0)
            cv2.line(frame, wrist_trail_points[j-1], wrist_trail_points[j], color, 4)

    # Draw ball
    if ball_pos:
        cv2.circle(frame, ball_pos, 6, (0, 255, 255), -1)

    # Draw keypoints
    if kpts:
        for idx, (x, y) in enumerate(kpts):
            color = (0, 0, 255) if idx == 10 else (0, 255, 0)
            cv2.circle(frame, (x, y), 5, color, -1)

    # Draw paddle
    if paddle_pts and len(paddle_pts) > 0:
        for idx, (px, py) in enumerate(paddle_pts):
            if px <= 0 or py <= 0:
                continue

            label = paddle_labels[idx] if idx < len(paddle_labels) else ""
            if label == "Center":
                color = (0, 0, 255)  # Red - center
            elif label in ("Grip_Top", "Grip_Bottom"):
                color = (0, 255, 0)  # Green - grip
            else:
                color = (255, 0, 0)  # Blue - edges

            cv2.circle(frame, (px, py), 6, color, -1)

    # Create info panel
    info_panel = draw_info_panel(info_template, header_height, ball_pos, ball_conf,
                                   paddle_pts, paddle_conf, kpts, kpt_confs, out_h, info_panel_width)

    # Combine frame and info panel
    combined_frame = np.hstack((frame, info_panel))

    return frame_idx, combined_frame

def frame_reader_thread(video_path, OUTPUT_WIDTH, OUTPUT_HEIGHT, frame_queue, stop_event, total_frames_ref):
    """Thread to read frames from video and put into queue"""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = resize_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        frame_queue.put((frame_idx, resized_frame))
        frame_idx += 1

    cap.release()
    total_frames_ref[0] = frame_idx
    stop_event.set()
    print(f"[INFO] Frame reader finished: {frame_idx} frames read")

def frame_writer_thread(output_queue, out, stop_event, expected_frames):
    """Thread to write frames to output video in order"""
    written = 0
    buffer = {}  # Buffer for out-of-order frames
    next_frame_idx = 0

    while not stop_event.is_set() or written < expected_frames:
        try:
            frame_idx, combined_frame = output_queue.get(timeout=0.1)
            buffer[frame_idx] = combined_frame

            # Write frames in order
            while next_frame_idx in buffer:
                out.write(buffer[next_frame_idx])
                del buffer[next_frame_idx]
                next_frame_idx += 1
                written += 1
        except queue.Empty:
            continue

    print(f"[INFO] Frame writer finished: {written} frames written")

def process_video(video_path, OUTPUT_WIDTH=1280, OUTPUT_HEIGHT=720, trace_length=60, num_workers=6):
    print(f"[INFO] Starting optimized video processing with {num_workers} workers...")

    # Load JSON trajectory data
    json_path = video_path.replace('.mp4', '.json')
    if not os.path.exists(json_path):
        print(f"❌ Error: JSON trajectory file not found: {json_path}")
        return None

    try:
        print(f"[INFO] Loading trajectory data from JSON: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        print(f"[INFO] Successfully loaded {len(trajectory_data)} frames from JSON")
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return None

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file: {video_path}")
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[INFO] Video: FPS={original_fps:.2f}, Resolution={orig_w}x{orig_h}, Frames={total_frames}")

    # Calculate scaling ratio
    scale_x = OUTPUT_WIDTH / orig_w if orig_w > 0 else 1.0
    scale_y = scale_x
    print(f"[INFO] Coordinate scaling: {scale_x:.4f}")

    # Pre-process JSON data into arrays for fast access
    print(f"[INFO] Pre-processing trajectory data...")
    ball_positions = [None] * total_frames
    ball_confidences = [None] * total_frames
    keypoints_per_frame = [None] * total_frames
    keypoints_conf_per_frame = [None] * total_frames
    paddle_keypoints = [None] * total_frames
    paddle_confidences = [None] * total_frames

    for frame_data in trajectory_data:
        idx = frame_data.get("frame", 0)
        if idx >= total_frames:
            continue

        # Ball
        ball = frame_data.get("tennis_ball", {})
        if ball and ball.get("x") is not None and ball.get("y") is not None:
            ball_positions[idx] = (int(ball["x"] * scale_x), int(ball["y"] * scale_y))
            ball_confidences[idx] = 1.0

        # Pose
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

        # Paddle
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

    # Setup output video
    output_path = video_path.replace('.mp4', '_processed.mp4')
    info_panel_width = 400
    out_w, out_h = OUTPUT_WIDTH + info_panel_width, OUTPUT_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (out_w, out_h))

    # Pre-create info panel template
    print(f"[INFO] Creating info panel template...")
    info_template, header_height = create_info_panel_template(out_h, info_panel_width)

    # Setup threading pipeline
    frame_queue = queue.Queue(maxsize=32)
    output_queue = queue.Queue(maxsize=32)
    stop_event = threading.Event()
    total_frames_ref = [0]  # Mutable reference

    # Start frame reader thread
    reader_thread = threading.Thread(
        target=frame_reader_thread,
        args=(video_path, OUTPUT_WIDTH, OUTPUT_HEIGHT, frame_queue, stop_event, total_frames_ref)
    )
    reader_thread.start()

    # Start frame writer thread
    writer_stop = threading.Event()
    writer_thread = threading.Thread(
        target=frame_writer_thread,
        args=(output_queue, out, writer_stop, total_frames)
    )
    writer_thread.start()

    # Process frames with thread pool
    print(f"[INFO] Processing frames with {num_workers} worker threads...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        processed_count = 0

        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame_idx, frame = frame_queue.get(timeout=0.1)

                # Prepare frame data
                ball_pos = ball_positions[frame_idx]
                ball_conf = ball_confidences[frame_idx]
                kpts = keypoints_per_frame[frame_idx]
                kpt_confs = keypoints_conf_per_frame[frame_idx]
                paddle_pts = paddle_keypoints[frame_idx]
                paddle_conf = paddle_confidences[frame_idx]

                frame_data = (frame_idx, frame, ball_pos, kpts, paddle_pts, paddle_conf,
                             ball_conf, kpt_confs, trace_length, ball_positions,
                             keypoints_per_frame, info_template, header_height, out_h, info_panel_width)

                # Submit to thread pool
                future = executor.submit(process_frame, frame_data)
                futures.append(future)

                # Collect completed frames (non-blocking)
                if len(futures) >= num_workers * 2:
                    completed = []
                    for f in futures[:]:
                        if f.done():
                            try:
                                result_idx, result_frame = f.result()
                                output_queue.put((result_idx, result_frame))
                                processed_count += 1
                                completed.append(f)
                            except Exception as e:
                                print(f"Error processing frame: {e}")
                                completed.append(f)
                    futures = [f for f in futures if f not in completed]

                if processed_count % 50 == 0 and processed_count > 0:
                    print(f"[INFO] Processed {processed_count} frames...")

            except queue.Empty:
                continue

        # Process remaining futures with timeout
        for future in as_completed(futures, timeout=60):
            try:
                result_idx, result_frame = future.result()
                output_queue.put((result_idx, result_frame))
                processed_count += 1
            except Exception as e:
                print(f"Error in remaining futures: {e}")

    print(f"[INFO] All frames processed: {processed_count} total")

    # Wait for threads to finish
    reader_thread.join()
    writer_stop.set()
    writer_thread.join()

    out.release()
    gc.collect()

    print(f"[SUCCESS] Output completed: {output_path}")
    return output_path


if __name__ == "__main__":
    total_start = time.time()
    video_path = '1209/1209_45.mp4'

    output_path = process_video(video_path, num_workers=6)

    if output_path:
        total_end = time.time()
        print(f"===== Total execution time: {total_end - total_start:.2f} seconds =====")
