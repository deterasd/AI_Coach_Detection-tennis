import os
import re
import shutil
import subprocess
import tempfile
import time


def derive_output_path(top_video):
    if top_video.endswith("_processed.mp4"):
        return top_video.replace("_processed.mp4", "_full_video.mp4")
    if top_video.endswith("_45.mp4"):
        return top_video.replace("_45.mp4", "_full_video.mp4")
    if top_video.endswith(".mp4"):
        return top_video.replace(".mp4", "_full_video.mp4")
    return f"{top_video}_full_video.mp4"


def ensure_writable_output_path(output_video):
    if not os.path.exists(output_video):
        return output_video

    try:
        os.remove(output_video)
        return output_video
    except PermissionError:
        root, ext = os.path.splitext(output_video)
        alt_output = f"{root}_{int(time.time())}{ext}"
        print(f"[WARN] Output is locked, using alternate output: {alt_output}")
        return alt_output


def check_ffmpeg_available():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")
    return ffmpeg_path


def has_video_stream(ffmpeg_path, video_path):
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-i", video_path],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False

    probe_text = f"{result.stdout}\n{result.stderr}"
    return re.search(r"Stream #\d+:\d+.*Video:", probe_text) is not None


def resolve_valid_input_video(ffmpeg_path, requested_path):
    candidates = [requested_path]

    if requested_path.endswith("_processed.mp4"):
        candidates.append(requested_path.replace("_processed.mp4", "_processed copy.mp4"))
        candidates.append(requested_path.replace("_processed.mp4", ".mp4"))

    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 1024 and has_video_stream(ffmpeg_path, candidate):
            if candidate != requested_path:
                print(f"[WARN] Fallback input selected: {candidate} (requested: {requested_path})")
            return candidate

    raise RuntimeError(
        f"No valid video stream found for input: {requested_path}. "
        f"Tried: {', '.join(candidates)}"
    )


def get_available_video_encoders(ffmpeg_path):
    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )

    encoders = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def select_encoder(ffmpeg_path):
    encoders = get_available_video_encoders(ffmpeg_path)

    if "h264_nvenc" in encoders:
        # Fastest practical NVENC setting for this workflow.
        return "h264_nvenc", ["-preset", "p1", "-cq", "23", "-b:v", "0"], "GPU (h264_nvenc)"

    if "libx264" in encoders:
        return "libx264", ["-preset", "veryfast", "-crf", "23"], "CPU (libx264)"

    raise RuntimeError("No supported encoder found: h264_nvenc/libx264")


def build_merge_command(ffmpeg_path, top_video, bottom_video, output_video, encoder, encoder_args):
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
        encoder,
        *encoder_args,
        output_video,
    ]


def combine_videos_ffmpeg(top_video, bottom_video):
    requested_top_video = top_video
    ffmpeg_path = check_ffmpeg_available()
    top_video = resolve_valid_input_video(ffmpeg_path, top_video)
    bottom_video = resolve_valid_input_video(ffmpeg_path, bottom_video)
    output_video = derive_output_path(requested_top_video)
    output_video = ensure_writable_output_path(output_video)
    temp_output_video = os.path.join(
        tempfile.gettempdir(), f"merge_{int(time.time() * 1000)}.mp4"
    )

    encoder, encoder_args, encoder_desc = select_encoder(ffmpeg_path)
    cmd = build_merge_command(
        ffmpeg_path=ffmpeg_path,
        top_video=top_video,
        bottom_video=bottom_video,
        output_video=temp_output_video,
        encoder=encoder,
        encoder_args=encoder_args,
    )

    print(f"[INFO] Merge encoder: {encoder_desc}")
    print(f"[INFO] Top input: {top_video}")
    print(f"[INFO] Bottom input: {bottom_video}")
    print(f"[INFO] Output: {output_video}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg merge failed with return code {e.returncode}") from e

    final_output = output_video
    try:
        shutil.move(temp_output_video, output_video)
    except PermissionError:
        final_output = temp_output_video
        print(
            f"[WARN] Cannot write target path '{output_video}'. "
            f"Keeping output at temp path: {final_output}"
        )

    print(f"[SUCCESS] Merge completed: {final_output}")
    return final_output


if __name__ == "__main__":
    start_time = time.time()

    top_video = "1209/1209_45_processed.mp4"
    bottom_video = "1209/1209_side_processed.mp4"

    print("[INFO] Starting video merge...")
    merged_video = combine_videos_ffmpeg(top_video, bottom_video)

    elapsed_time = time.time() - start_time
    print(f"[INFO] Total elapsed: {elapsed_time:.2f}s")
    print(f"[INFO] Final merged video: {merged_video}")
