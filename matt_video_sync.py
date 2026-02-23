import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2


def get_hit_frame(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for frame_info in data:
        if frame_info.get("tennis_ball_hit", False):
            return int(frame_info["frame"])

    return None


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = total_frames / fps
    return total_frames, fps, duration, (width, height)


def check_ffmpeg_available():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg first, then retry."
        )
    return ffmpeg_path


def get_available_encoders(ffmpeg_path):
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to query ffmpeg encoders: {e.stderr}") from e

    lines = result.stdout.splitlines()
    encoders = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def build_ffmpeg_command(
    ffmpeg_path,
    input_path,
    output_path,
    start_sec,
    frames_to_process,
    encoder,
):
    base = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        input_path,
        "-frames:v",
        str(frames_to_process),
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        encoder,
    ]

    if encoder == "h264_nvenc":
        base.extend(["-preset", "p1", "-cq", "23", "-b:v", "0"])
    elif encoder == "libx264":
        base.extend(["-preset", "ultrafast", "-crf", "23"])

    base.append(output_path)
    return base


def process_video_ffmpeg(
    ffmpeg_path,
    input_path,
    output_path,
    start_frame,
    frames_to_process,
    fps,
    preferred_encoder,
):
    start_sec = start_frame / fps

    cmd = build_ffmpeg_command(
        ffmpeg_path=ffmpeg_path,
        input_path=input_path,
        output_path=output_path,
        start_sec=start_sec,
        frames_to_process=frames_to_process,
        encoder=preferred_encoder,
    )

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed for {input_path} with encoder {preferred_encoder}: {e.stderr}"
        ) from e


def synchronize_videos(
    input_path_1,
    input_path_2,
    json_path_1,
    json_path_2,
    trim_length=60,
    overwrite_input=True,
):
    ffmpeg_path = check_ffmpeg_available()
    encoders = get_available_encoders(ffmpeg_path)

    if "h264_nvenc" in encoders:
        selected_encoder = "h264_nvenc"
        print("[INFO] Using ffmpeg encoder: h264_nvenc (GPU)")
    elif "libx264" in encoders:
        selected_encoder = "libx264"
        print("[INFO] Using ffmpeg encoder: libx264 (CPU)")
    else:
        raise RuntimeError("No supported H.264 encoder found (h264_nvenc/libx264).")

    frames1, fps1, duration1, _ = get_video_info(input_path_1)
    frames2, fps2, duration2, _ = get_video_info(input_path_2)

    print("\n[INFO] Video info:")
    print(f"Video 1: {frames1} frames, {duration1:.2f}s, fps={fps1:.3f}")
    print(f"Video 2: {frames2} frames, {duration2:.2f}s, fps={fps2:.3f}")

    hit_frame_1 = get_hit_frame(json_path_1)
    hit_frame_2 = get_hit_frame(json_path_2)

    if hit_frame_1 is None or hit_frame_2 is None:
        raise RuntimeError(
            f"Missing tennis_ball_hit frame. hit1={hit_frame_1}, hit2={hit_frame_2}"
        )

    print("\n[INFO] Hit frame:")
    print(f"Video 1 hit: frame {hit_frame_1}")
    print(f"Video 2 hit: frame {hit_frame_2}")

    max_frames_after = min(frames1 - hit_frame_1, frames2 - hit_frame_2)
    max_frames_before = min(hit_frame_1, hit_frame_2)

    max_frames_after = max(0, max_frames_after)
    max_frames_before = max(0, max_frames_before)

    frames_before = min(trim_length // 2, max_frames_before)
    frames_after = min(trim_length - frames_before, max_frames_after)

    start_frame_1 = hit_frame_1 - frames_before
    start_frame_2 = hit_frame_2 - frames_before
    frames_to_process = frames_before + frames_after

    if frames_to_process <= 0:
        raise RuntimeError("frames_to_process <= 0. Check hit frames and input videos.")

    print("\n[INFO] Sync range:")
    print(
        f"Video 1: start={start_frame_1}, end={start_frame_1 + frames_to_process - 1}"
    )
    print(
        f"Video 2: start={start_frame_2}, end={start_frame_2 + frames_to_process - 1}"
    )
    print(f"Output length: {frames_to_process} frames")

    temp_dir = tempfile.gettempdir()
    output_path_1 = os.path.join(temp_dir, f"temp_sync_1_{os.path.basename(input_path_1)}")
    output_path_2 = os.path.join(temp_dir, f"temp_sync_2_{os.path.basename(input_path_2)}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                process_video_ffmpeg,
                ffmpeg_path,
                input_path_1,
                output_path_1,
                start_frame_1,
                frames_to_process,
                fps1,
                selected_encoder,
            ),
            executor.submit(
                process_video_ffmpeg,
                ffmpeg_path,
                input_path_2,
                output_path_2,
                start_frame_2,
                frames_to_process,
                fps2,
                selected_encoder,
            ),
        ]

        for future in as_completed(futures):
            future.result()

    if overwrite_input:
        shutil.move(output_path_1, input_path_1)
        shutil.move(output_path_2, input_path_2)
        final_path_1 = input_path_1
        final_path_2 = input_path_2
    else:
        final_path_1 = input_path_1.replace(".mp4", "_synced.mp4")
        final_path_2 = input_path_2.replace(".mp4", "_synced.mp4")
        shutil.move(output_path_1, final_path_1)
        shutil.move(output_path_2, final_path_2)

    final_duration_1 = frames_to_process / fps1
    final_duration_2 = frames_to_process / fps2

    print("\n[INFO] Sync completed")
    print(f"Video 1 output: {final_path_1} ({final_duration_1:.3f}s)")
    print(f"Video 2 output: {final_path_2} ({final_duration_2:.3f}s)")

    return final_path_1, final_path_2


if __name__ == "__main__":
    start_time = time.time()

    input_video_1 = "1209/1209_45_processed.mp4"
    input_video_2 = "1209/1209_side_processed.mp4"
    json_path_1 = "1209/1209_45.json"
    json_path_2 = "1209/1209_side.json"

    print("[INFO] Starting video synchronization...")
    synchronize_videos(
        input_video_1,
        input_video_2,
        json_path_1,
        json_path_2,
        trim_length=60,
        overwrite_input=True,
    )

    print(f"[INFO] Total time: {time.time() - start_time:.4f}s")
