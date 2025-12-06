#!/usr/bin/env python3
"""
FFmpeg 手動安裝工具
如果程式中的自動安裝失敗，可以手動執行這個腳本
"""

import requests
import zipfile
import shutil
from pathlib import Path
import subprocess
import sys

def check_ffmpeg():
    """檢查FFmpeg是否已安裝"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return True
    except:
        return False

def check_local_ffmpeg():
    """檢查本地是否有FFmpeg"""
    local_ffmpeg = Path("tools/ffmpeg.exe")
    return local_ffmpeg.exists()

def install_ffmpeg():
    """下載並安裝FFmpeg"""
    try:
        print("🔄 正在下載 FFmpeg...")
        print("⚠️  首次安裝需要下載約100MB，請稍等...")
        
        # 創建tools資料夾
        tools_dir = Path("tools")
        tools_dir.mkdir(exist_ok=True)
        
        # FFmpeg下載URL
        ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        
        # 下載FFmpeg
        response = requests.get(ffmpeg_url, stream=True)
        response.raise_for_status()
        
        zip_path = tools_dir / "ffmpeg.zip"
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"📥 下載進度: {progress:.1f}%", end='\r')
        
        print("\n📦 正在解壓縮 FFmpeg...")
        
        # 解壓縮
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tools_dir)
        
        # 找到ffmpeg.exe並複製到tools根目錄
        ffmpeg_found = False
        for item in tools_dir.glob("ffmpeg-*"):
            if item.is_dir():
                ffmpeg_exe = item / "bin" / "ffmpeg.exe"
                if ffmpeg_exe.exists():
                    target_path = tools_dir / "ffmpeg.exe"
                    shutil.copy2(ffmpeg_exe, target_path)
                    ffmpeg_found = True
                    break
        
        # 清理下載檔案和解壓縮資料夾
        zip_path.unlink(missing_ok=True)
        for item in tools_dir.glob("ffmpeg-*"):
            if item.is_dir():
                shutil.rmtree(item)
        
        if ffmpeg_found:
            print("✅ FFmpeg 安裝成功！")
            return True
        else:
            print("❌ FFmpeg 安裝失敗：找不到執行檔")
            return False
            
    except Exception as e:
        print(f"❌ FFmpeg 安裝失敗: {e}")
        return False

def main():
    print("=== FFmpeg 安裝工具 ===")
    
    # 檢查系統FFmpeg
    if check_ffmpeg():
        print("✅ 系統已安裝 FFmpeg")
        return
    
    # 檢查本地FFmpeg
    if check_local_ffmpeg():
        print("✅ 本地已有 FFmpeg (tools/ffmpeg.exe)")
        return
    
    # 安裝FFmpeg
    print("❌ 未檢測到 FFmpeg，開始安裝...")
    
    try:
        if install_ffmpeg():
            print("🎉 安裝完成！現在可以使用GPU加速的視頻分割了")
        else:
            print("❌ 安裝失敗，請檢查網路連線或手動下載")
            print("手動下載地址：https://www.gyan.dev/ffmpeg/builds/")
    except KeyboardInterrupt:
        print("\n❌ 安裝被取消")
    except Exception as e:
        print(f"❌ 安裝過程出錯: {e}")

if __name__ == "__main__":
    main()