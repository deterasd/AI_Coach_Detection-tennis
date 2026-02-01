import os
import sys
import time
import subprocess
from pathlib import Path


def combine_videos_ffmpeg(top_video, bottom_video):
    # 修復輸出檔案名稱生成邏輯，避免與輸入檔案重名
    top_path = Path(top_video)
    output_name = top_path.stem.replace('_45_segment_processed', '_full_video').replace('_45_processed', '_full_video')
    output_video = str(top_path.parent / f"{output_name}.mp4")

    # 如果輸出檔案和輸入檔案相同，則添加後綴
    if output_video == top_video:
        output_name = top_path.stem + '_full_video'
        output_video = str(top_path.parent / f"{output_name}.mp4")

    # 使用 subprocess 列表參數，避免 shell 解析造成 "Unrecognized option 'rc'" / "Error splitting the argument list"
    # macOS 無 CUDA/NVENC，改用 libx264；有 NVIDIA 的環境可改為 h264_nvenc
    use_nvenc = sys.platform != "darwin"  # 非 macOS 時可嘗試 nvenc（需本機有 NVIDIA 驅動）
    if use_nvenc:
        args = [
            "ffmpeg", "-y", "-hwaccel", "cuda",
            "-i", top_video, "-i", bottom_video,
            "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]", "-map", "[v]",
            "-c:v", "h264_nvenc", "-preset", "p7", "-profile:v", "high", "-qp", "0",
            "-pix_fmt", "yuv420p", "-threads", "8", output_video,
        ]
    else:
        # macOS / 無 NVIDIA：使用 libx264，相容性最佳
        args = [
            "ffmpeg", "-y",
            "-i", top_video, "-i", bottom_video,
            "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]", "-map", "[v]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-threads", "8", output_video,
        ]

    print(f"🎬 合併影片: {Path(top_video).name} + {Path(bottom_video).name} → {Path(output_video).name}")
    try:
        ret = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if ret.returncode == 0 and Path(output_video).exists():
            return output_video
        if ret.stderr:
            print(ret.stderr[-2000:] if len(ret.stderr) > 2000 else ret.stderr)
        print(f"❌ 影片合併失敗，返回值: {ret.returncode}")
    except FileNotFoundError:
        print("❌ 找不到 ffmpeg，請安裝 FFmpeg 後再試。")
    except subprocess.TimeoutExpired:
        print("❌ 影片合併逾時。")
    return None

if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    top_video = "testing__45.mp4"
    bottom_video = "testing__side.mp4"

    print("開始合併影片（超高畫質 + GPU 加速）...")
    combine_videos_ffmpeg(top_video, bottom_video)
    end_time = time.time()  # 記錄結束時間

    elapsed_time = end_time - start_time  # 計算執行時間

    print(f"處理時間: {elapsed_time:.2f} 秒")  # 顯示處理時間
