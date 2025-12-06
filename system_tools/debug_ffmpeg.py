"""
調試 FFmpeg 路徑問題
"""
from pathlib import Path
import subprocess
import sys
sys.path.append('.')

from trajector_processing_with_segmentation import VideoSegmenter

def debug_ffmpeg():
    print("🔍 調試 FFmpeg 路徑問題")
    print("=" * 50)
    
    # 測試 VideoSegmenter
    print("\n1. 測試 VideoSegmenter 初始化...")
    segmenter = VideoSegmenter()
    print(f"   FFmpeg 命令: {segmenter.ffmpeg_cmd}")
    
    # 測試本地 FFmpeg 檔案
    print("\n2. 檢查本地 FFmpeg 檔案...")
    local_ffmpeg = Path("tools/ffmpeg.exe")
    print(f"   檔案存在: {local_ffmpeg.exists()}")
    if local_ffmpeg.exists():
        print(f"   絕對路徑: {local_ffmpeg.absolute()}")
        print(f"   檔案大小: {local_ffmpeg.stat().st_size / 1024 / 1024:.1f} MB")
    
    # 測試系統 FFmpeg
    print("\n3. 檢查系統 FFmpeg...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   系統 FFmpeg: 可用")
        else:
            print("   系統 FFmpeg: 不可用")
    except:
        print("   系統 FFmpeg: 不可用")
    
    # 測試 VideoSegmenter 的 FFmpeg 檢查函數
    print("\n4. 測試 VideoSegmenter 的 FFmpeg 檢查...")
    ffmpeg_cmd = segmenter._get_ffmpeg_command()
    print(f"   _get_ffmpeg_command() 返回: {ffmpeg_cmd}")
    
    # 模擬分割測試
    if ffmpeg_cmd:
        print("\n5. 模擬分割測試...")
        try:
            cmd = [ffmpeg_cmd, '-version']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("   ✅ FFmpeg 命令可執行")
                
                # 測試實際分割參數
                test_video = Path('input_videos/tennis_side.MP4')
                if test_video.exists():
                    print("   📹 找到測試影片，測試分割命令構建...")
                    
                    cmd = [
                        ffmpeg_cmd, '-y',
                        '-i', str(test_video.absolute()),
                        '-ss', '3.0',
                        '-t', '2.0',
                        '-c', 'copy',
                        '-avoid_negative_ts', 'make_zero',
                        'debug_test_output.mp4'
                    ]
                    
                    print(f"   命令: {' '.join(cmd)}")
                    
                    # 不實際執行，只檢查命令構建
                    print("   ✅ 命令構建成功")
                else:
                    print("   ⚠️ 找不到測試影片")
            else:
                print(f"   ❌ FFmpeg 命令執行失敗: {result.stderr}")
        except Exception as e:
            print(f"   ❌ 測試失敗: {e}")
    else:
        print("\n5. ❌ 無法取得 FFmpeg 命令")

if __name__ == "__main__":
    debug_ffmpeg()