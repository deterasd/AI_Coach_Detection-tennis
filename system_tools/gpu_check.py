#!/usr/bin/env python3
"""
GPU 檢測工具
檢查系統GPU狀態和CUDA可用性
"""

import subprocess
import sys
from pathlib import Path

def check_nvidia_gpu():
    """檢查NVIDIA GPU"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ 檢測到 NVIDIA GPU:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi 執行失敗")
            return False
    except:
        print("❌ 未安裝 NVIDIA 驅動程式或 nvidia-smi")
        return False

def check_cuda():
    """檢查CUDA版本"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"✅ CUDA 版本: {line.strip()}")
                    return True
        return False
    except:
        print("❌ CUDA 未安裝或不在系統路徑中")
        return False

def check_opencv_gpu():
    """檢查OpenCV GPU支援"""
    try:
        import cv2
        print(f"✅ OpenCV 版本: {cv2.__version__}")
        
        # 檢查CUDA支援
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ OpenCV 支援 CUDA，可用GPU數量: {cv2.cuda.getCudaEnabledDeviceCount()}")
            return True
        else:
            print("❌ OpenCV 不支援 CUDA 或未檢測到 GPU")
            return False
    except ImportError:
        print("❌ OpenCV 未安裝")
        return False
    except:
        print("❌ OpenCV CUDA 檢查失敗")
        return False

def check_ffmpeg_gpu():
    """檢查FFmpeg GPU支援"""
    ffmpeg_paths = ['ffmpeg', str(Path('tools/ffmpeg.exe').absolute())]
    
    for ffmpeg_cmd in ffmpeg_paths:
        try:
            result = subprocess.run([ffmpeg_cmd, '-encoders'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ 找到 FFmpeg: {ffmpeg_cmd}")
                
                # 檢查NVIDIA編碼器
                encoders = result.stdout
                nvidia_encoders = []
                
                if 'h264_nvenc' in encoders:
                    nvidia_encoders.append('h264_nvenc')
                if 'hevc_nvenc' in encoders:
                    nvidia_encoders.append('hevc_nvenc')
                if 'av1_nvenc' in encoders:
                    nvidia_encoders.append('av1_nvenc')
                
                if nvidia_encoders:
                    print(f"✅ FFmpeg 支援 NVIDIA 編碼器: {', '.join(nvidia_encoders)}")
                    return True
                else:
                    print("❌ FFmpeg 不支援 NVIDIA 編碼器")
                    return False
        except:
            continue
    
    print("❌ 未找到 FFmpeg")
    return False

def check_python_gpu_libs():
    """檢查Python GPU相關函式庫"""
    gpu_libs = []
    
    # 檢查PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_libs.append(f"PyTorch (CUDA {torch.version.cuda})")
        else:
            print("❌ PyTorch 不支援 CUDA")
    except ImportError:
        pass
    
    # 檢查TensorFlow
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            gpu_libs.append("TensorFlow")
        else:
            print("❌ TensorFlow 不支援 GPU")
    except (ImportError, ModuleNotFoundError):
        pass
    
    if gpu_libs:
        print(f"✅ GPU 相關函式庫: {', '.join(gpu_libs)}")
        return True
    else:
        print("❓ 未安裝 PyTorch 或 TensorFlow")
        return False

def main():
    print("=== GPU 和 CUDA 環境檢測 ===\n")
    
    print("1. 檢查 NVIDIA GPU:")
    gpu_available = check_nvidia_gpu()
    print()
    
    print("2. 檢查 CUDA:")
    cuda_available = check_cuda()
    print()
    
    print("3. 檢查 OpenCV GPU 支援:")
    opencv_gpu = check_opencv_gpu()
    print()
    
    print("4. 檢查 FFmpeg GPU 支援:")
    ffmpeg_gpu = check_ffmpeg_gpu()
    print()
    
    print("5. 檢查 Python GPU 函式庫:")
    python_gpu = check_python_gpu_libs()
    print()
    
    print("=== 總結 ===")
    if gpu_available:
        print("✅ GPU 硬體: 可用")
    else:
        print("❌ GPU 硬體: 不可用")
    
    if cuda_available:
        print("✅ CUDA 環境: 可用")
    else:
        print("❌ CUDA 環境: 不可用")
    
    if ffmpeg_gpu:
        print("✅ FFmpeg GPU 加速: 可用")
    else:
        print("❌ FFmpeg GPU 加速: 不可用")
    
    if opencv_gpu:
        print("✅ OpenCV GPU 加速: 可用")
    else:
        print("❌ OpenCV GPU 加速: 不可用")
    
    # 給出建議
    print("\n=== 建議 ===")
    if not gpu_available:
        print("💡 請確認 NVIDIA 驅動程式已正確安裝")
    elif not cuda_available:
        print("💡 請安裝 CUDA Toolkit")
    elif not ffmpeg_gpu:
        print("💡 請使用支援 GPU 的 FFmpeg 版本")
    else:
        print("🎉 所有 GPU 加速條件都已滿足！")

if __name__ == "__main__":
    main()