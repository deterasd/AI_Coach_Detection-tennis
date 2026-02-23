import cv2
import json
import time
import torch
from ultralytics import YOLO
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

def frame_reader(video_path, frame_queue, stop_event):
    # Continuously read video frames and put them into queue with hardware acceleration
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    frame_number = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((frame_number, frame))
        frame_number += 1
    cap.release()
    stop_event.set()

def process_single_frame(body_result, ball_result, paddle_result, keypoint_names, frame_number):
    # Process a single frame result and return frame_data dictionaryy
    frame_data = {
        "frame": frame_number,
        "tennis_ball": {"x": None, "y": None, "conf": None},
        "paddle": {
            "top": {"x": None, "y": None, "conf": None},
            "bottom": {"x": None, "y": None, "conf": None},
            "right": {"x": None, "y": None, "conf": None},
            "left": {"x": None, "y": None, "conf": None},
            "center": {"x": None, "y": None, "conf": None},
            "grip_top": {"x": None, "y": None, "conf": None},
            "grip_bottom": {"x": None, "y": None, "conf": None},
        }
    }

    for keypoint in keypoint_names:
        frame_data[keypoint] = {"x": None, "y": None, "conf": None}

    # Body keypoints
    if body_result.keypoints is not None:
        keypoints = body_result.keypoints.xy[0].cpu().numpy()
        confs = body_result.keypoints.conf[0].cpu().numpy() if body_result.keypoints.conf is not None else None

        if len(keypoints) == len(keypoint_names):
            for idx, keypoint in enumerate(keypoint_names):
                x, y = keypoints[idx][:2]
                conf = confs[idx] if confs is not None else 0.0

                if x != 0.0 or y != 0.0:
                    frame_data[keypoint].update({
                        "x": int(x),
                        "y": int(y),
                        "conf": float(conf)
                    })

    # Tennis ball position
    for box in ball_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        if confidence > 0.8:
            frame_data["tennis_ball"].update({
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "conf": confidence
            })
            break

    # Paddle position (7 keypoints)
    if paddle_result.keypoints is not None and len(paddle_result.keypoints.xy) > 0:
        kpts = paddle_result.keypoints.xy[0].cpu().numpy()
        p_confs = paddle_result.keypoints.conf[0].cpu().numpy() if paddle_result.keypoints.conf is not None else None

        if kpts.shape[0] >= 6:
            top, bottom, right, left, grip_top, grip_bottom = kpts[:6]

            c_top = float(p_confs[0]) if p_confs is not None else 0.0
            c_bottom = float(p_confs[1]) if p_confs is not None else 0.0
            c_right = float(p_confs[2]) if p_confs is not None else 0.0
            c_left = float(p_confs[3]) if p_confs is not None else 0.0
            c_grip_top = float(p_confs[4]) if p_confs is not None else 0.0
            c_grip_bottom = float(p_confs[5]) if p_confs is not None else 0.0

            frame_data["paddle"]["top"] = {"x": int(top[0]), "y": int(top[1]), "conf": c_top}
            frame_data["paddle"]["bottom"] = {"x": int(bottom[0]), "y": int(bottom[1]), "conf": c_bottom}
            frame_data["paddle"]["right"] = {"x": int(right[0]), "y": int(right[1]), "conf": c_right}
            frame_data["paddle"]["left"] = {"x": int(left[0]), "y": int(left[1]), "conf": c_left}
            frame_data["paddle"]["grip_top"] = {"x": int(grip_top[0]), "y": int(grip_top[1]), "conf": c_grip_top}
            frame_data["paddle"]["grip_bottom"] = {"x": int(grip_bottom[0]), "y": int(grip_bottom[1]), "conf": c_grip_bottom}

            cx = int((top[0] + right[0] + bottom[0] + left[0]) / 4)
            cy = int((top[1] + right[1] + bottom[1] + left[1]) / 4)
            frame_data["paddle"]["center"].update({"x": cx, "y": cy})

    return frame_data

def _infer_on_stream(model, frames, stream):
    """Execute model inference on specified CUDA stream"""
    with torch.cuda.stream(stream):
        return model(frames, verbose=False)

def _run_batch_inference(pose_model, ball_model, paddle_model, batch_frames, streams, executor):
    """Run three models in parallel using CUDA streams"""
    futures = [
        executor.submit(_infer_on_stream, pose_model, batch_frames, streams[0]),
        executor.submit(_infer_on_stream, ball_model, batch_frames, streams[1]),
        executor.submit(_infer_on_stream, paddle_model, batch_frames, streams[2]),
    ]
    body_results = futures[0].result()
    ball_results = futures[1].result()
    paddle_results = futures[2].result()
    torch.cuda.synchronize()
    return body_results, ball_results, paddle_results

def process_video_batch(pose_model, ball_model, paddle_model, video_path, batch_size=8):
    """Process video with async reading and parallel CUDA streams inference"""
    frame_queue = queue.Queue(maxsize=8 * batch_size)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(video_path, frame_queue, stop_event))
    reader_thread.start()

    use_cuda = torch.cuda.is_available()
    streams = [torch.cuda.Stream() for _ in range(3)] if use_cuda else None

    frame_json = []
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    batch_frames = []
    batch_indices = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        while not (stop_event.is_set() and frame_queue.empty()):
            try:
                frame_index, frame = frame_queue.get(timeout=0.1)
                batch_frames.append(frame)
                batch_indices.append(frame_index)
                if len(batch_frames) == batch_size:
                    with torch.no_grad():
                        if use_cuda:
                            body_results, ball_results, paddle_results = _run_batch_inference(
                                pose_model, ball_model, paddle_model, batch_frames, streams, executor
                            )
                        else:
                            body_results = pose_model(batch_frames, verbose=False)
                            ball_results = ball_model(batch_frames, verbose=False)
                            paddle_results = paddle_model(batch_frames, verbose=False)
                    for idx, (body_result, ball_result, paddle_result) in enumerate(zip(body_results, ball_results, paddle_results)):
                        frame_data = process_single_frame(body_result, ball_result, paddle_result, keypoint_names, batch_indices[idx])
                        frame_json.append(frame_data)
                    del batch_frames, batch_indices, body_results, ball_results, paddle_results
                    batch_frames = []
                    batch_indices = []
            except queue.Empty:
                continue

        # Process remaining frames
        if batch_frames:
            with torch.no_grad():
                if use_cuda:
                    body_results, ball_results, paddle_results = _run_batch_inference(
                        pose_model, ball_model, paddle_model, batch_frames, streams, executor
                    )
                else:
                    body_results = pose_model(batch_frames, verbose=False)
                    ball_results = ball_model(batch_frames, verbose=False)
                    paddle_results = paddle_model(batch_frames, verbose=False)
            for idx, (body_result, ball_result, paddle_result) in enumerate(zip(body_results, ball_results, paddle_results)):
                frame_data = process_single_frame(body_result, ball_result, paddle_result, keypoint_names, batch_indices[idx])
                frame_json.append(frame_data)

    # Fill missing keypoints in last frame with previous frame data
    if frame_json and len(frame_json) > 1:
        last_frame = frame_json[-1]
        prev_frame = frame_json[-2]
        for keypoint in keypoint_names:
            if last_frame[keypoint]["x"] is None:
                last_frame[keypoint] = prev_frame[keypoint]

    reader_thread.join()
    return frame_json

def analyze_trajectory(pose_model, ball_model, paddle_model, video_path, batch_size):
    trajectory = process_video_batch(pose_model, ball_model, paddle_model, video_path, batch_size=batch_size)
    output_path = video_path.replace('.mp4', '(2D_trajectory).json')
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    return output_path

if __name__ == "__main__":
    total_start_time = time.time()
    BATCH_SIZE = 8

    # Load TensorRT engines
    print("Loading TensorRT engines...")
    model_load_start = time.time()
    pose_model = YOLO('model/yolo11l-pose.engine', task='pose')
    ball_model = YOLO('model/tennisball_OD_v1.engine', task='detect')
    paddle_model = YOLO('model/yolov11x.engine', task='pose')
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.4f}s")

    video_path = '1209/1209_side.mp4'

    analysis_start = time.time()
    output_path = analyze_trajectory(pose_model, ball_model, paddle_model, video_path, batch_size=BATCH_SIZE)
    analysis_time = time.time() - analysis_start
    print(f"Trajectory analysis time: {analysis_time:.4f}s")

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.4f}s")
