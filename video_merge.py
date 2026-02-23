import os
import time
import subprocess

def combine_videos_ffmpeg(top_video, bottom_video):
    # 更靈活的輸出檔案命名：移除 _processed.mp4 或 _45.mp4，加上 _full_video.mp4
    output_video = top_video.replace('_processed.mp4', '_full_video.mp4')
    if output_video == top_video:  # 如果沒有成功替換
        # 嘗試其他替換模式
        output_video = top_video.replace('_45.mp4', '_full_video.mp4')
    if output_video == top_video:  # 如果還是沒有成功替換
        # 直接在副檔名前插入
        output_video = top_video.replace('.mp4', '_full_video.mp4')
    
    # 檢查是否有 CUDA 支援
    try:
        result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True, timeout=5)
        has_cuda = 'cuda' in result.stdout
    except:
        has_cuda = False
    
    if has_cuda:
        # 使用 GPU 加速
        cmd = (
            f'ffmpeg -y -hwaccel cuda -i "{top_video}" -i "{bottom_video}" '
            f'-filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" '
            f'-c:v h264_nvenc -preset p7 -profile:v high444p -qp 0 -b:v 50000k '
            f'-rc constqp -pix_fmt yuv444p -threads 8 -bf 2 "{output_video}"'
        )
        print("🎬 使用 GPU 加速合併影片...")
    else:
        # 使用 CPU 模式
        cmd = (
            f'ffmpeg -y -i "{top_video}" -i "{bottom_video}" '
            f'-filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" '
            f'-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -threads 8 "{output_video}"'
        )
        print("🎬 使用 CPU 模式合併影片...")
    
    # 使用 subprocess 執行並捕獲輸出
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 合併成功: {output_video}")
        else:
            print(f"❌ 合併失敗: {result.stderr}")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
    
    return output_video

if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    top_video = "testing__45.mp4"
    bottom_video = "testing__side.mp4"

    print("開始合併影片...")
    result = combine_videos_ffmpeg(top_video, bottom_video)
    end_time = time.time()  # 記錄結束時間

    elapsed_time = end_time - start_time  # 計算執行時間

    print(f"處理時間: {elapsed_time:.2f} 秒")  # 顯示處理時間
    print(f"📹 合併影片已移動到: {result}")
