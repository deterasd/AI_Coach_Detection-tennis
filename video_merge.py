import subprocess
import time
from pathlib import Path

FFMPEG_PATH = r"C:/ffmpeg/ffmpeg.exe"  # ✅ 改成你實際 ffmpeg.exe 的位置

def combine_videos_ffmpeg(top_video, bottom_video):
    top_video = str(Path(top_video))
    bottom_video = str(Path(bottom_video))

    output_video = top_video.replace('_45_processed.mp4', '_full_video.mp4')
    if output_video == top_video:
        # 若檔名不含 _45_processed.mp4，就用通用命名
        output_video = str(Path(top_video).with_name(Path(top_video).stem + "_full_video.mp4"))

    cmd = [
        FFMPEG_PATH,
        "-y",  # 覆蓋輸出
        "-i", top_video,
        "-i", bottom_video,
        "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", "18",          # 品質好、檔案不會爆
        "-preset", "veryfast", # 速度
        output_video
    ]

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ 合併完成:", output_video)
    return output_video

if __name__ == "__main__":
    start_time = time.time()

    top_video = "testing__45.mp4"
    bottom_video = "testing__side.mp4"

    print("開始合併影片（CPU x264）...")
    combine_videos_ffmpeg(top_video, bottom_video)

    print(f"處理時間: {time.time() - start_time:.2f} 秒")

"""
import os
import time

def combine_videos_ffmpeg(top_video, bottom_video):
    output_video = top_video.replace('_45_processed.mp4','_full_video.mp4')
    cmd = (
        f'ffmpeg -hwaccel cuda -i "{top_video}" -i "{bottom_video}" '
        f'-filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" '
        f'-c:v h264_nvenc -preset p7 -profile:v high444p -qp 0 -b:v 50000k '
        f'-rc constqp -pix_fmt yuv444p -threads 8 -bf 2 {output_video}'
    )
    
    os.system(cmd)  # 執行 FFmpeg 指令

if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    top_video = "testing__45.mp4"
    bottom_video = "testing__side.mp4"

    print("開始合併影片（超高畫質 + GPU 加速）...")
    combine_videos_ffmpeg(top_video, bottom_video)
    end_time = time.time()  # 記錄結束時間

    elapsed_time = end_time - start_time  # 計算執行時間

    print(f"處理時間: {elapsed_time:.2f} 秒")  # 顯示處理時間
"""