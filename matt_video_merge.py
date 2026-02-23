import os
import shutil
import subprocess
import tempfile
import time


ENCODER_CONFIG = {
    "h264_nvenc": ("h264_nvenc", ["-preset", "p1", "-cq", "24", "-b:v", "0"], "GPU (h264_nvenc)"),
    "libx264": ("libx264", ["-preset", "ultrafast", "-crf", "24"], "CPU (libx264)"),
    "mp4v": ("mpeg4", ["-q:v", "4", "-vtag", "mp4v"], "mp4v (mpeg4)"),
}


def derive_output_path(top_video):
    if top_video.endswith("_processed.mp4"):
        return top_video.replace("_processed.mp4", "_full_video.mp4")
    if top_video.endswith("_45.mp4"):
        return top_video.replace("_45.mp4", "_full_video.mp4")
    if top_video.endswith(".mp4"):
        return top_video.replace(".mp4", "_full_video.mp4")
    return f"{top_video}_full_video.mp4"


def check_ffmpeg_available():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")
    return ffmpeg_path


def build_merge_command(ffmpeg_path, top_video, bottom_video, output_video, encoder_name):
    if encoder_name not in ENCODER_CONFIG:
        raise RuntimeError(f"Unsupported encoder: {encoder_name}")

    codec, encoder_args, _ = ENCODER_CONFIG[encoder_name]
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        top_video,
        "-i",
        bottom_video,
        "-filter_complex",
        "[0:v][1:v]vstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        codec,
        *encoder_args,
        output_video,
    ]


def run_merge(ffmpeg_path, top_video, bottom_video, temp_output_video, preferred_encoder):
    if preferred_encoder == "auto":
        encoders_to_try = ["h264_nvenc", "libx264"]
    else:
        encoders_to_try = [preferred_encoder]

    last_error = None
    for encoder_name in encoders_to_try:
        cmd = build_merge_command(
            ffmpeg_path=ffmpeg_path,
            top_video=top_video,
            bottom_video=bottom_video,
            output_video=temp_output_video,
            encoder_name=encoder_name,
        )
        try:
            subprocess.run(cmd, check=True)
            return ENCODER_CONFIG[encoder_name][2]
        except subprocess.CalledProcessError as e:
            last_error = e

    if preferred_encoder == "auto":
        raise RuntimeError("ffmpeg merge failed for both h264_nvenc and libx264") from last_error
    raise RuntimeError(f"ffmpeg merge failed for encoder: {preferred_encoder}") from last_error


def combine_videos_ffmpeg(top_video, bottom_video, preferred_encoder="mp4v"):
    if not os.path.exists(top_video):
        raise FileNotFoundError(f"Top input not found: {top_video}")
    if not os.path.exists(bottom_video):
        raise FileNotFoundError(f"Bottom input not found: {bottom_video}")

    ffmpeg_path = check_ffmpeg_available()
    output_video = derive_output_path(top_video)
    temp_output_video = os.path.join(tempfile.gettempdir(), f"merge_{int(time.time() * 1000)}.mp4")

    print(f"[INFO] Top input: {top_video}")
    print(f"[INFO] Bottom input: {bottom_video}")
    print(f"[INFO] Output: {output_video}")

    try:
        encoder_desc = run_merge(
            ffmpeg_path=ffmpeg_path,
            top_video=top_video,
            bottom_video=bottom_video,
            temp_output_video=temp_output_video,
            preferred_encoder=preferred_encoder,
        )
        print(f"[INFO] Merge encoder: {encoder_desc}")
    except Exception:
        if os.path.exists(temp_output_video):
            try:
                os.remove(temp_output_video)
            except OSError:
                pass
        raise

    final_output = output_video
    try:
        shutil.move(temp_output_video, output_video)
    except (PermissionError, OSError):
        final_output = temp_output_video
        print(f"[WARN] Cannot write target path '{output_video}'. Keeping temp output: {final_output}")

    print(f"[SUCCESS] Merge completed: {final_output}")
    return final_output


if __name__ == "__main__":
    start_time = time.time()

    top_video = "1209/1209_45_processed.mp4"
    bottom_video = "1209/1209_side_processed.mp4"

    print("[INFO] Starting video merge...")
    merged_video = combine_videos_ffmpeg(top_video, bottom_video, preferred_encoder="mp4v")

    elapsed_time = time.time() - start_time
    print(f"[INFO] Total elapsed: {elapsed_time:.2f}s")
    print(f"[INFO] Final merged video: {merged_video}")
