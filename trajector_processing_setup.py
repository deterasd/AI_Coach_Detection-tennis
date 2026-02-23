"""
軌跡處理前置設施模組
========================================
提供 trajector_processing_unified.py 所需的：
  - DetailedLogger (日誌管理)
  - NanToNullEncoder (JSON 編碼)
  - GPU 初始化與記憶體管理
  - 輸出資料夾包裝函數 (將結果搬移到指定資料夾)
  - 處理摘要生成

此模組不包含任何核心處理邏輯，僅提供輔助工具。
"""

import time
import os
import json
import shutil
import math
import gc
import torch
import psutil
import logging
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


# ============================================
# GPU 全局變數與初始化
# ============================================

GPU_AVAILABLE = None
GPU_DEVICE = None


def init_gpu_device():
    """動態初始化 GPU 設備（延遲初始化避免模組載入時的問題）"""
    global GPU_AVAILABLE, GPU_DEVICE
    if GPU_AVAILABLE is None:
        GPU_AVAILABLE = torch.cuda.is_available()
        GPU_DEVICE = 'cuda:0' if GPU_AVAILABLE else 'cpu'
        if GPU_AVAILABLE:
            torch.backends.cudnn.benchmark = True
    return GPU_AVAILABLE, GPU_DEVICE


def clear_all_memory(aggressive=False):
    """清理記憶體 - aggressive=True 時做深度清理"""
    if aggressive and GPU_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def check_system_memory():
    """檢查系統記憶體是否足夠（至少 2GB）"""
    memory = psutil.virtual_memory()
    return memory.available > 2 * 1024**3


def check_gpu_memory():
    """檢查 GPU 記憶體是否足夠（至少 1GB）"""
    if GPU_AVAILABLE:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        return (total_memory - cached_memory) > 1.0
    return False


def get_gpu_info():
    """取得 GPU 資訊字串"""
    init_gpu_device()
    if GPU_AVAILABLE:
        props = torch.cuda.get_device_properties(0)
        return f"CUDA 可用 - {props.name} (記憶體: {props.total_memory / 1024**3:.1f}GB)"
    return "僅使用 CPU"


# ============================================
# 日誌管理
# ============================================

class DetailedLogger:
    """詳細日誌記錄器 - 同時輸出到控制台和檔案"""
    
    def __init__(self, log_folder):
        self.log_folder = Path(log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_folder / f"execution_log_{timestamp}.txt"
        
        self.logger = logging.getLogger(f'trajectory_processing_{timestamp}')
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重複添加處理器
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, msg):
        self.logger.info(msg)
        print(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
        print(f"⚠️ {msg}")
    
    def error(self, msg):
        self.logger.error(msg)
        print(f"❌ {msg}")
    
    def step(self, step_num, step_name, is_start=True):
        """步驟日誌"""
        status = "開始" if is_start else "完成"
        msg = f"{'='*60}\n步驟 {step_num}: {step_name} - {status}\n時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if GPU_AVAILABLE:
            try:
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                msg += f"\nGPU 記憶體: {gpu_mem:.2f}GB"
            except:
                pass
        msg += f"\n{'='*60}"
        self.info(msg)
    
    def get_log_path(self):
        return str(self.log_file)


# ============================================
# JSON 編碼器
# ============================================

class NanToNullEncoder(json.JSONEncoder):
    """自定義 JSON encoder，將 NaN 轉換為 null"""
    def encode(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 'null'
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        for chunk in super().iterencode(obj, _one_shot):
            yield chunk.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')


# ============================================
# 輸出資料夾包裝函數
# ============================================

def analyze_trajectory_with_output_folder(pose_model, ball_model, video_path, batch_size, output_folder, paddle_model=None):
    """分析軌跡並將結果保存到指定資料夾"""
    from trajectory_2D_output import process_video_batch
    
    init_gpu_device()
    
    if paddle_model is None:
        try:
            paddle_model = YOLO('model/yolov11x.pt')
            if GPU_AVAILABLE:
                paddle_model.to(GPU_DEVICE)
        except Exception as e:
            print(f"⚠️ 球拍模型載入失敗: {e}")
            paddle_model = ball_model
    
    if GPU_AVAILABLE:
        for model in [pose_model, ball_model, paddle_model]:
            if hasattr(model, 'to'):
                model.to(GPU_DEVICE)
    
    trajectory = process_video_batch(pose_model, ball_model, paddle_model, video_path, batch_size=batch_size)
    
    video_name = Path(video_path).stem
    output_path = Path(output_folder) / f"{video_name}(2D_trajectory).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)


def smooth_2D_trajectory_with_output_folder(trajectory_path, output_folder):
    """平滑處理2D軌跡並將結果保存到指定資料夾"""
    from trajector_2D_smoothing import smooth_2D_trajectory
    
    smoothed_trajectory_path = smooth_2D_trajectory(trajectory_path)
    
    source_path = Path(smoothed_trajectory_path)
    target_path = Path(output_folder) / source_path.name
    
    if source_path.exists() and source_path != target_path:
        shutil.move(str(source_path), str(target_path))
        return str(target_path)
    
    return smoothed_trajectory_path


def process_video_with_output_folder(video_path, output_folder, ball_model=None, pose_model=None, paddle_model=None, json_path=None):
    """處理影片並將結果保存到指定資料夾"""
    from video_detection import process_video
    
    processed_video_path = process_video(
        video_path,
        ball_model=ball_model,
        pose_model=pose_model,
        paddle_model=paddle_model,
        json_path=json_path
    )
    
    if processed_video_path and Path(processed_video_path).exists():
        source_path = Path(processed_video_path)
        target_path = Path(output_folder) / source_path.name
        if source_path != target_path:
            shutil.move(str(source_path), str(target_path))
            return str(target_path)
    
    return processed_video_path


def save_knn_feedback_with_output_folder(knn_result, output_folder, name):
    """保存KNN反饋到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment_knn_feedback.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if isinstance(knn_result, list):
            f.write(knn_result[0] if len(knn_result) > 0 else "無KNN分析結果")
        else:
            f.write(knn_result)
    
    return str(output_path)


def save_gpt_feedback_with_output_folder(gpt_result, output_folder, name):
    """保存GPT反饋到指定資料夾"""
    output_path = Path(output_folder) / f"{name}_segment_gpt_feedback.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gpt_result, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
    
    return str(output_path)


def move_to_output_folder(source_path, output_folder, target_filename):
    """將檔案搬移到輸出資料夾，回傳新路徑"""
    if source_path and Path(source_path).exists():
        target_path = Path(output_folder) / target_filename
        if Path(source_path) != target_path:
            shutil.move(str(source_path), str(target_path))
            return str(target_path)
    return source_path


# ============================================
# 處理摘要
# ============================================

def generate_processing_summary(output_folder, name, timing_results, total_time):
    """生成處理摘要檔案"""
    try:
        summary = {
            "user_name": name,
            "processing_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_time_seconds": total_time,
            "step_times": timing_results,
            "output_folder": str(output_folder),
            "status": "completed"
        }
        
        summary_file = Path(output_folder) / f"{name}__processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
        
        print(f"📊 處理摘要已保存: {summary_file.name}")
        return True
    except Exception as e:
        print(f"⚠️ 生成處理摘要失敗: {e}")
        return False


def write_ready_marker(output_folder, message="Done"):
    """建立完成標記檔案"""
    try:
        ready_file_path = os.path.join(output_folder, "ready.txt")
        with open(ready_file_path, "w", encoding='utf-8') as f:
            f.write(f"{message} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return ready_file_path
    except Exception as e:
        print(f"⚠️ 無法建立標記檔案: {e}")
        return None


def resolve_segment_video_paths(video_side, video_45, segmentation_results, logger=None):
    """從分割結果中解析實際要使用的影片路徑"""
    actual_video_side = video_side
    actual_video_45 = video_45
    
    if segmentation_results and segmentation_results.get("ball_pairs"):
        ball_pair = segmentation_results["ball_pairs"][0]
        
        if ball_pair.get("side_data") and ball_pair["side_data"].get("segment"):
            segment_value = ball_pair["side_data"]["segment"]
            segment_path = segment_value.get("file_path") if isinstance(segment_value, dict) else segment_value
            if segment_path:
                actual_video_side = os.path.abspath(segment_path) if not os.path.isabs(segment_path) else segment_path
                if logger:
                    logger.info(f"🎬 使用側面分割片段: {os.path.basename(actual_video_side)}")
            
        if ball_pair.get("deg45_data") and ball_pair["deg45_data"].get("segment"):
            segment_value = ball_pair["deg45_data"]["segment"]
            segment_path = segment_value.get("file_path") if isinstance(segment_value, dict) else segment_value
            if segment_path:
                actual_video_45 = os.path.abspath(segment_path) if not os.path.isabs(segment_path) else segment_path
                if logger:
                    logger.info(f"🎬 使用45度分割片段: {os.path.basename(actual_video_45)}")
    
    return actual_video_side, actual_video_45


def get_segment_name(name, output_folder):
    """從 output_folder 推導 segment_name"""
    folder_name = Path(output_folder).name
    if folder_name.startswith("trajectory_"):
        ball_number = folder_name.split("_")[-1]
        return f"{name}__球{ball_number}"
    return f"{name}__球1"


def print_timing_summary(name, timing_results, total_time):
    """輸出時間統計摘要"""
    print('\n' + '=' * 60)
    print(f"🎯 {name} 的軌跡分析完成！")
    print('=' * 60)
    print("⏱️ 執行時間統計:")
    print('-' * 60)
    for step, t in timing_results.items():
        print(f"{step:.<30} {t:>10.4f} 秒")
    print('-' * 60)
    print(f"{'總執行時間':.<30} {total_time:>10.4f} 秒")
    print('=' * 60)
