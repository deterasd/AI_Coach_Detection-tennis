"""
VideoWriter 安全初始化工具
用於解決不同系統和編碼器兼容性問題
"""

import cv2
import os
from pathlib import Path

def safe_video_writer(output_path, fps, frame_size, fourcc_preference='mp4v'):
    """
    安全的 VideoWriter 初始化，支持多編碼器回退
    
    Args:
        output_path (str): 輸出影片路徑
        fps (float): 幀率
        frame_size (tuple): 影片尺寸 (width, height)
        fourcc_preference (str): 優先使用的編碼器
    
    Returns:
        tuple: (VideoWriter對象, 實際使用的輸出路徑)
    """
    
    # 確保輸出目錄存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 編碼器回退序列
    codec_fallbacks = [
        (fourcc_preference, cv2.VideoWriter_fourcc(*fourcc_preference)),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('X264', cv2.VideoWriter_fourcc(*'X264')),
        ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
    ]
    
    # 首先嘗試指定的編碼器
    for codec_name, fourcc in codec_fallbacks:
        print(f"🔧 嘗試 {codec_name} 編碼器...")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        
        if out.isOpened():
            print(f"✅ {codec_name} 編碼器初始化成功")
            return out, str(output_path)
        else:
            out.release()
    
    # 如果所有 MP4 編碼器都失敗，嘗試 AVI 格式
    print("🔧 嘗試 AVI 格式...")
    avi_path = output_path.with_suffix('.avi')
    out = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    
    if out.isOpened():
        print(f"✅ AVI 格式初始化成功: {avi_path.name}")
        return out, str(avi_path)
    else:
        out.release()
    
    # 最後嘗試系統預設編碼器
    print("🔧 嘗試系統預設編碼器...")
    out = cv2.VideoWriter(str(output_path), -1, fps, frame_size)
    
    if out.isOpened():
        print("✅ 系統預設編碼器初始化成功")
        return out, str(output_path)
    else:
        out.release()
    
    print("❌ 所有 VideoWriter 初始化方法都失敗")
    return None, None

def validate_video_file(file_path, min_size_kb=10):
    """
    驗證生成的影片檔案是否有效
    
    Args:
        file_path (str): 影片檔案路徑
        min_size_kb (int): 最小檔案大小（KB）
    
    Returns:
        bool: 檔案是否有效
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ 檔案不存在: {file_path}")
            return False
        
        file_size_kb = file_path.stat().st_size / 1024
        
        if file_size_kb < min_size_kb:
            print(f"❌ 檔案太小 ({file_size_kb:.1f} KB < {min_size_kb} KB): {file_path.name}")
            return False
        
        # 嘗試用 OpenCV 讀取檔案驗證完整性
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            print(f"❌ 無法讀取影片檔案: {file_path.name}")
            cap.release()
            return False
        
        # 檢查影片是否有幀
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ 影片檔案無法讀取幀: {file_path.name}")
            return False
        
        print(f"✅ 影片檔案驗證通過: {file_path.name} ({file_size_kb:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"❌ 檔案驗證錯誤: {e}")
        return False

def cleanup_failed_video(file_path):
    """
    清理失敗的影片檔案
    
    Args:
        file_path (str): 要清理的檔案路徑
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ 已清理損壞檔案: {file_path.name}")
    except Exception as e:
        print(f"⚠️ 清理檔案失敗: {e}")

# 使用範例函數
def example_usage():
    """使用範例"""
    print("VideoWriter 安全初始化工具使用範例:")
    print()
    print("# 基本使用")
    print("out, actual_path = safe_video_writer('output.mp4', 30.0, (1280, 720))")
    print("if out:")
    print("    # 寫入影片幀...")
    print("    out.release()")
    print("    # 驗證生成的檔案")
    print("    if validate_video_file(actual_path):")
    print("        print('影片生成成功')")
    print("    else:")
    print("        cleanup_failed_video(actual_path)")
    print()
    print("# 指定編碼器")
    print("out, actual_path = safe_video_writer('output.mp4', 30.0, (1280, 720), 'H264')")

if __name__ == "__main__":
    example_usage()