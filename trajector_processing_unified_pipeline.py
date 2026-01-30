#使用來測試的程式，可以直接放影片進input_video

"""
統一輸出管理的軌跡處理流程 - 專門用於 simple_test
支援多球分析，每顆球會有獨立的資料夾和完整的分析結果
整合影片自動分割功能
"""

import time
import numpy as np
import os
import json
import shutil
import torch
import gc
import psutil
import cv2
import subprocess
import math
import threading
import queue
from pathlib import Path
from ultralytics import YOLO

# 設定 PyTorch 記憶體分配策略以減少碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# === 新增：從其他模組導入分割功能 ===
try:
    from video_segmentation import detect_ball_entries_optimized, segment_video_dynamic
except ImportError:
    print("⚠️ 無法從 video_segmentation 導入功能，請確保該檔案存在")

# === 新增：全域變數與鎖 ===
clip_queue = queue.Queue()
gpu_lock = threading.Lock()
processing_complete_event = threading.Event()

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

def analyze_trajectory_with_output_folder(pose_model, ball_model, video_path, batch_size, output_folder, paddle_model=None):
    """
    分析軌跡並將結果保存到指定資料夾
    """
    from trajectory_2D_output import process_video_batch
    import json
    
    # 如果沒有提供 paddle_model，載入預設模型
    if paddle_model is None:
        try:
            paddle_model = YOLO('model/tennispaddle.pt')  # 使用正確的球拍關鍵點模型
            print("📦 已自動載入球拍模型: model/tennispaddle.pt")
        except Exception as e:
            print(f"⚠️ 球拍模型載入失敗: {e}")
            # 如果載入失敗，使用 ball_model 作為替代（雖然不會有效果）
            paddle_model = ball_model
    
    # 使用鎖保護 GPU 資源
    with gpu_lock:
        trajectory = process_video_batch(pose_model, ball_model, paddle_model, video_path, batch_size=batch_size)
    
    # 生成輸出檔案名稱（基於原始檔名）
    video_name = Path(video_path).stem
    output_path = Path(output_folder) / f"{video_name}(2D_trajectory).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def smooth_2D_trajectory_with_output_folder(trajectory_path, output_folder):
    """
    平滑處理2D軌跡並將結果保存到指定資料夾
    """
    from trajector_2D_smoothing import smooth_2D_trajectory
    import json
    
    # 執行平滑處理
    smoothed_trajectory_path = smooth_2D_trajectory(trajectory_path)
    
    # 移動結果到指定資料夾
    source_path = Path(smoothed_trajectory_path)
    target_path = Path(output_folder) / source_path.name
    
    if source_path.exists() and source_path != target_path:
        import shutil
        shutil.move(str(source_path), str(target_path))
        return str(target_path)
    
    return smoothed_trajectory_path

def process_video_with_output_folder(video_path, output_folder, ball_model=None, pose_model=None, paddle_model=None):
    """
    處理影片並將結果保存到指定資料夾
    """
    from video_detection import process_video
    import shutil
    
    # 執行影片處理 (process_video 會自動使用正確的 paddle_model_path)
    # 使用鎖保護 GPU 資源
    with gpu_lock:
        processed_video_path = process_video(
            video_path,
            ball_model=ball_model,
            pose_model=pose_model,
            paddle_model=paddle_model
        )
    
    # 移動結果到指定資料夾
    if processed_video_path and Path(processed_video_path).exists():
        source_path = Path(processed_video_path)
        target_path = Path(output_folder) / source_path.name
        
        if source_path != target_path:
            shutil.move(str(source_path), str(target_path))
            return str(target_path)
    
    return processed_video_path

def save_3d_trajectory_with_output_folder(trajectory_3d, output_folder, name):
    """
    保存3D軌跡到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_3d, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def save_3d_smoothed_trajectory_with_output_folder(trajectory_3d_smoothing, output_folder, name):
    """
    保存3D平滑軌跡到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory_smoothed).json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_3d_smoothing, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def save_3d_swing_range_with_output_folder(trajectory_3d_swing_range, output_folder, name):
    """
    保存3D擊球範圍軌跡到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment(3D_trajectory_smoothed)_only_swing.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_3d_swing_range, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)
    
    return str(output_path)

def save_knn_feedback_with_output_folder(knn_result, output_folder, name):
    """
    保存KNN反饋到指定資料夾
    """
    output_path = Path(output_folder) / f"{name}_segment_knn_feedback.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 如果是列表，取第一個元素或合併成字串
        if isinstance(knn_result, list):
            if len(knn_result) > 0:
                f.write(knn_result[0])
            else:
                f.write("無KNN分析結果")
        else:
            f.write(knn_result)
    
    return str(output_path)

def save_gpt_feedback_with_output_folder(gpt_result, output_folder, name):
    """
    保存GPT反饋到指定資料夾
    """
    import json
    
    output_path = Path(output_folder) / f"{name}_segment_gpt_feedback.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gpt_result, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
    
    return str(output_path)

def clear_all_memory():
    """清理所有記憶體（GPU + RAM）"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def check_system_memory():
    """檢查系統記憶體使用情況"""
    memory = psutil.virtual_memory()
    return memory.available > 2 * 1024**3  # 至少需要 2GB 可用

def check_gpu_memory():
    """檢查 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        return (total_memory - cached_memory) > 1.0  # 至少需要 1GB 可用
    return False

# === 新增：分析工作執行緒 ===
def analysis_worker(yolo_pose_model, yolo_tennis_ball_model, paddle_model, knn_dataset, pipeline_start_time):
    """
    背景分析執行緒，從 Queue 中取出球對進行分析
    """
    print("🚀 分析執行緒已啟動，等待影片中...")
    
    while True:
        # 1. 從佇列中拿取一個任務
        task = clip_queue.get()
        
        # 2. 檢查是否是結束信號
        if task is None:
            print("🏁 收到結束信號，分析執行緒停止。")
            clip_queue.task_done()
            break
            
        # 3. 解析任務資料
        try:
            # 解包任務參數
            (P1, P2, video_side, video_45, name, ball_folder, timing_results, ball_segmentation, ball_number) = task
            
            print(f"⚡ [分析開始] 正在處理第 {ball_number} 顆球...")
            
            # 呼叫單球處理函式
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, ball_folder, timing_results, ball_segmentation, paddle_model
            )
            
            if success:
                print(f"✅ [分析完成] 第 {ball_number} 顆球處理成功")
                
                # === 新增：計算第一顆球的完成時間 ===
                if ball_number == 1:
                    first_ball_time = time.perf_counter() - pipeline_start_time
                    print(f"\n⏱️ [效能指標] 第一顆球從開始到完成共耗時: {first_ball_time:.2f} 秒")
                    print(f"   (使用者等待時間)\n")
                # ===================================
                
            else:
                print(f"⚠️ [分析警告] 第 {ball_number} 顆球處理有部分問題")
                
        except Exception as e:
            print(f"❌ [分析失敗] 第 {ball_number} 顆球處理發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 4. 告訴佇列這個任務做完了
            clip_queue.task_done()
            # 強制清理記憶體
            clear_all_memory()

def process_single_video_set(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                           video_side, video_45, knn_dataset, 
                           name, output_folder, timing_results, segmentation_results=None, paddle_model=None):
    """處理單組影片的完整流程"""
    try:
        # 從output_folder推導球號
        output_folder_path = Path(output_folder)
        folder_name = output_folder_path.name
        if folder_name.startswith("trajectory_"):
            ball_number = folder_name.split("_")[-1]
            segment_name = f"{name}__{ball_number}"
        else:
            segment_name = f"{name}__1"
        
        # 匯入原本的處理模組
        from video_sync import synchronize_videos
        from video_merge import combine_videos_ffmpeg
        from trajector_2D_sync import sync_trajectories
        from trajector_2D_capture_swing_range import find_range
        from trajectory_3D_output import process_trajectories
        from trajector_3D_smoothing import smooth_3D_trajectory
        from trajector_3D_capture_swing_range import extract_frames
        from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
        from trajectory_gpt_single_feedback import generate_feedback_data_only
        
        # 確定要使用的影片路徑
        actual_video_side = video_side
        actual_video_45 = video_45
        
        # 如果有分割結果，使用分割後的片段
        if segmentation_results and segmentation_results.get("ball_pairs"):
            ball_pair = segmentation_results["ball_pairs"][0]  # 取第一個球對
            
            if ball_pair.get("side_data") and ball_pair["side_data"].get("segment"):
                segment_value = ball_pair["side_data"]["segment"]
                if isinstance(segment_value, dict):
                    segment_path = segment_value.get("file_path")
                else:
                    segment_path = segment_value
                if segment_path:
                    if not os.path.isabs(segment_path):
                        actual_video_side = os.path.abspath(segment_path)
                    else:
                        actual_video_side = segment_path
                    print(f"🎬 使用側面分割片段: {os.path.basename(actual_video_side)}")
                
            if ball_pair.get("deg45_data") and ball_pair["deg45_data"].get("segment"):
                segment_value = ball_pair["deg45_data"]["segment"]
                if isinstance(segment_value, dict):
                    segment_path = segment_value.get("file_path")
                else:
                    segment_path = segment_value
                if segment_path:
                    if not os.path.isabs(segment_path):
                        actual_video_45 = os.path.abspath(segment_path)
                    else:
                        actual_video_45 = segment_path
                    print(f"🎬 使用45度分割片段: {os.path.basename(actual_video_45)}")
        
        # 步驟1：分析2D軌跡
        print("\n步驟1：分析2D軌跡...")
        start = time.perf_counter()
        
        # 修改為保存到對應資料夾
        trajectory_side = analyze_trajectory_with_output_folder(yolo_pose_model, yolo_tennis_ball_model, actual_video_side, 28, output_folder, paddle_model)
        trajectory_45 = analyze_trajectory_with_output_folder(yolo_pose_model, yolo_tennis_ball_model, actual_video_45, 28, output_folder, paddle_model)
        
        timing_results['2D軌跡分析'] = time.perf_counter() - start
        print(f"✅ 2D軌跡分析完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")
        
        clear_all_memory()

        # 步驟2：2D軌跡平滑處理
        print("\n步驟2：2D軌跡平滑處理...")
        start = time.perf_counter()
        
        # 修改為保存到對應資料夾
        trajectory_side_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_side, output_folder)
        trajectory_45_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_45, output_folder)
        
        timing_results['2D平滑處理'] = time.perf_counter() - start
        print(f"✅ 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")
        
        clear_all_memory()

        # 步驟3：影片處理
        print("\n步驟3：影片物件偵測處理...")
        print("⚠️ 注意：此步驟可能消耗大量記憶體，依序處理以節省資源...")
        start = time.perf_counter()
        
        # 依序處理影片以節省記憶體，並直接保存到對應資料夾
        print("📹 處理側面影片...")
        video_side_processed = process_video_with_output_folder(
            actual_video_side, 
            output_folder,
            ball_model=yolo_tennis_ball_model,
            pose_model=yolo_pose_model,
            paddle_model=paddle_model
        )
        clear_all_memory()
        
        print("📹 處理45度影片...")
        video_45_processed = process_video_with_output_folder(
            actual_video_45, 
            output_folder,
            ball_model=yolo_tennis_ball_model,
            pose_model=yolo_pose_model,
            paddle_model=paddle_model
        )
        clear_all_memory()
        
        # 顯示處理結果
        if video_side_processed:
            print(f"📹 側面處理影片已保存: {Path(video_side_processed).name}")
        if video_45_processed:
            print(f"📹 45度處理影片已保存: {Path(video_45_processed).name}")
        
        timing_results['影片處理'] = time.perf_counter() - start
        print(f"✅ 影片處理完成，耗時：{timing_results['影片處理']:.4f} 秒")

        # 步驟4：影片同步
        print("\n步驟4：同步影片...")
        start = time.perf_counter()
        
        synchronize_videos(video_side_processed, video_45_processed, 
                          trajectory_side_smoothing, trajectory_45_smoothing)
        
        timing_results['影片同步'] = time.perf_counter() - start
        print(f"✅ 影片同步完成，耗時：{timing_results['影片同步']:.4f} 秒")

        # 步驟5：合併影片
        print("\n步驟5：合併影片...")
        start = time.perf_counter()
        
        merged_video = combine_videos_ffmpeg(video_45_processed, video_side_processed)
        
        # 移動合併後的影片到對應資料夾
        if merged_video and Path(merged_video).exists():
            final_merged_path = Path(output_folder) / f"{segment_name}_full_video.mp4"
            shutil.move(merged_video, final_merged_path)
            print(f"📹 合併影片已移動到: {final_merged_path.name}")
        
        timing_results['影片合併'] = time.perf_counter() - start
        print(f"✅ 影片合併完成，耗時：{timing_results['影片合併']:.4f} 秒")

        # 步驟6：軌跡同步
        print("\n步驟6：同步軌跡...")
        start = time.perf_counter()
        
        sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
        
        timing_results['軌跡同步'] = time.perf_counter() - start
        print(f"✅ 軌跡同步完成，耗時：{timing_results['軌跡同步']:.4f} 秒")

        # 步驟7：3D軌跡分析
        print("\n步驟7：計算3D軌跡...")
        start = time.perf_counter()
        
        trajectory_3d_path = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
        
        # 保存3D軌跡到對應資料夾（從原始位置移動）
        if trajectory_3d_path and Path(trajectory_3d_path).exists():
            source_path = Path(trajectory_3d_path)
            target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory).json"
            
            if source_path != target_path:
                shutil.move(str(source_path), str(target_path))
                trajectory_3d_path = str(target_path)
        
        timing_results['3D軌跡分析'] = time.perf_counter() - start
        print(f"✅ 3D軌跡計算完成，耗時：{timing_results['3D軌跡分析']:.4f} 秒")

        # 步驟8：3D軌跡平滑處理
        print("\n步驟8：3D軌跡平滑處理...")
        start = time.perf_counter()
        
        # 使用檔案路徑進行平滑處理
        trajectory_3d_smoothing_path = smooth_3D_trajectory(trajectory_3d_path)
        
        # 移動平滑結果到對應資料夾
        if trajectory_3d_smoothing_path and Path(trajectory_3d_smoothing_path).exists():
            source_path = Path(trajectory_3d_smoothing_path)
            target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory_smoothed).json"
            
            if source_path != target_path:
                shutil.move(str(source_path), str(target_path))
                trajectory_3d_smoothing_path = str(target_path)
        
        timing_results['3D平滑處理'] = time.perf_counter() - start
        print(f"✅ 3D平滑處理完成，耗時：{timing_results['3D平滑處理']:.4f} 秒")

        # 步驟9：有效擊球範圍判斷
        print("\n步驟9：判斷有效擊球範圍...")
        start = time.perf_counter()
        
        start_frame, end_frame = find_range(trajectory_side_smoothing)
        
        # 使用檔案路徑進行範圍擷取（extract_frames 期待檔案路徑）
        trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing_path, start_frame, end_frame)
        
        # 移動擊球範圍軌跡到對應資料夾
        if trajectory_3d_swing_range and Path(trajectory_3d_swing_range).exists():
            source_path = Path(trajectory_3d_swing_range)
            target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory_smoothed)_only_swing.json"
            
            if source_path != target_path:
                shutil.move(str(source_path), str(target_path))
                trajectory_3d_swing_range = str(target_path)
        
        timing_results['有效擊球範圍判斷'] = time.perf_counter() - start
        print(f"✅ 有效擊球範圍判斷完成，耗時：{timing_results['有效擊球範圍判斷']:.4f} 秒")

        # 步驟10：KNN分析
        print("\n步驟10：KNN分析...")
        start = time.perf_counter()
        
        # 使用3D平滑軌跡檔案路徑進行KNN分析
        trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing_path)
        
        # 保存KNN反饋到對應資料夾
        knn_feedback_path = save_knn_feedback_with_output_folder(trajectory_knn_suggestion, output_folder, segment_name)
        
        timing_results['KNN 分析'] = time.perf_counter() - start
        print(f"✅ KNN分析完成，耗時：{timing_results['KNN 分析']:.4f} 秒")

        # 步驟11：GPT反饋生成（帶錯誤容錯）
        print("\n步驟11：生成GPT反饋...")
        start = time.perf_counter()
        
        try:
            # GPT分析使用檔案路徑
            # trajectory_3d_swing_range 是 JSON 檔案路徑
            # knn_feedback_path 是 KNN 分析結果的 txt 檔案路徑
            trajectory_gpt_suggestion = generate_feedback_data_only(trajectory_3d_swing_range, knn_feedback_path)
            
            # 檢查是否有錯誤標記
            if isinstance(trajectory_gpt_suggestion, dict) and trajectory_gpt_suggestion.get('error', False):
                error_type = trajectory_gpt_suggestion.get('error_type', 'unknown')
                if error_type == 'quota_exceeded':
                    print("⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代")
                else:
                    print(f"⚠️ GPT API 發生錯誤 ({error_type})，已使用 KNN 分析結果作為替代")
            
            # 保存GPT反饋到對應資料夾（即使有錯誤也保存替代結果）
            gpt_feedback_path = save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
            
            timing_results['GPT 反饋生成'] = time.perf_counter() - start
            print(f"✅ GPT反饋生成完成，耗時：{timing_results['GPT 反饋生成']:.4f} 秒")
            
        except Exception as e:
            print(f"⚠️ GPT反饋生成失敗: {e}")
            print("⚠️ 跳過 GPT 步驟，繼續處理...")
            
            # 創建一個簡單的反饋結果
            trajectory_gpt_suggestion = {
                "problem_frame": "N/A",
                "suggestion": "GPT功能暫時無法使用，請參考KNN分析結果",
                "error": True,
                "error_type": "processing_error"
            }
            
            # 嘗試保存錯誤反饋
            try:
                gpt_feedback_path = save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
            except:
                print("⚠️ 無法保存 GPT 反饋檔案，繼續處理...")
            
            timing_results['GPT 反饋生成'] = time.perf_counter() - start

        # === 確保全部檔案都寫入後，建立完成標記檔案 ===
        try:
            ready_file_path = os.path.join(output_folder, "ready.txt")
            with open(ready_file_path, "w", encoding='utf-8') as f:
                f.write(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🚩 [完成標記] 已建立：{ready_file_path}")
        except Exception as e:
            print(f"⚠️ 無法建立標記檔案: {e}")

        return True
        
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def move_processed_videos(video_side_processed, video_45_processed, name, output_folder):
    """移動並重新命名處理後的影片檔案"""
    try:
        output_folder = Path(output_folder)
        
        if video_side_processed and Path(video_side_processed).exists():
            new_name = output_folder / f"{name}__1_side_processed.mp4"
            shutil.move(video_side_processed, new_name)
            print(f"📹 側面處理影片已移動: {new_name.name}")
            
        if video_45_processed and Path(video_45_processed).exists():
            new_name = output_folder / f"{name}__1_45_processed.mp4"
            shutil.move(video_45_processed, new_name)
            print(f"📹 45度處理影片已移動: {new_name.name}")
            
        return True
    except Exception as e:
        print(f"⚠️ 移動處理影片失敗: {e}")
        return False

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
        
        summary_file = output_folder / f"{name}__processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NanToNullEncoder)
        
        print(f"📊 處理摘要已保存: {summary_file.name}")
        return True
    except Exception as e:
        print(f"⚠️ 生成處理摘要失敗: {e}")
        return False

def align_ball_segments(side_ball_data, deg45_ball_data, name):
    """
    對齊側面和45度影片的球片段
    基於時間相近性進行配對
    """
    print(f"🔄 開始球對對齊...")
    ball_pairs = []
    time_tolerance = 2.0  # 允許的時間差異（秒）
    used_deg45_indices = set()
    
    for side_idx, (side_entry, side_exit, side_segment) in enumerate(side_ball_data):
        best_match_idx = None
        best_time_diff = float('inf')
        
        for deg45_idx, (deg45_entry, deg45_exit, deg45_segment) in enumerate(deg45_ball_data):
            if deg45_idx in used_deg45_indices:
                continue
            time_diff = abs(side_entry - deg45_entry)
            if time_diff < best_time_diff and time_diff <= time_tolerance:
                best_time_diff = time_diff
                best_match_idx = deg45_idx
        
        ball_number = side_idx + 1
        if best_match_idx is not None:
            used_deg45_indices.add(best_match_idx)
            deg45_entry, deg45_exit, deg45_segment = deg45_ball_data[best_match_idx]
            ball_pair = {
                "ball_number": ball_number,
                "side_data": {"entry_time": side_entry, "exit_time": side_exit, "segment": side_segment},
                "deg45_data": {"entry_time": deg45_entry, "exit_time": deg45_exit, "segment": deg45_segment},
                "status": "paired"
            }
        else:
            ball_pair = {
                "ball_number": ball_number,
                "side_data": {"entry_time": side_entry, "exit_time": side_exit, "segment": side_segment},
                "deg45_data": None,
                "status": "unpaired_side_only"
            }
        ball_pairs.append(ball_pair)
    
    return ball_pairs

def process_video_pipeline(video_side, video_45, name, P1, P2, knn_dataset_path):
    """
    主流程：使用生產者-消費者模式
    1. 主執行緒 (Producer): 負責分割影片，將任務丟入 Queue
    2. 背景執行緒 (Consumer): 負責從 Queue 取出任務並執行分析
    """
    print(f"🚀 開始處理流程 (Pipeline Mode)")
    print(f"   側面影片: {video_side}")
    print(f"   45度影片: {video_45}")
    
    start_total_time = time.perf_counter()
    
    # 載入模型 (只載入一次，傳遞給 worker)
    print("\n📦 載入模型中...")
    try:
        # 使用鎖保護模型載入過程
        with gpu_lock:
            # 改用較輕量的 yolov8n-pose.pt 以提升速度並減少記憶體佔用
            yolo_pose_model = YOLO('model/yolov8n-pose.pt')
            yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
            paddle_model = YOLO('model/tennispaddle.pt')
            
            # 僅移動到 GPU，不手動轉 .half() 以避免 Ultralytics 內部融合錯誤
            if torch.cuda.is_available():
                yolo_pose_model.to('cuda')
                yolo_tennis_ball_model.to('cuda')
                paddle_model.to('cuda')
                print("🚀 模型已載入至 GPU")
            else:
                print("⚠️ 未偵測到 GPU，使用 CPU 模式")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return False

    # 載入 KNN 資料集
    try:
        with open(knn_dataset_path, 'r', encoding='utf-8') as f:
            knn_dataset = json.load(f)
        print(f"📚 KNN 資料集已載入: {len(knn_dataset)} 筆資料")
    except Exception as e:
        print(f"⚠️ KNN 資料集載入失敗: {e}")
        knn_dataset = []

    # 啟動分析執行緒 (Consumer)
    worker_thread = threading.Thread(
        target=analysis_worker, 
        args=(yolo_pose_model, yolo_tennis_ball_model, paddle_model, knn_dataset, start_total_time),
        daemon=True
    )
    worker_thread.start()
    
    # 開始分割影片 (Producer)
    print("\n✂️ 開始分割影片...")
    
    # 建立基礎輸出目錄
    base_output_dir = Path("output") / f"{name}_{int(time.time())}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用鎖保護 GPU 資源進行偵測
        with gpu_lock:
            print(f"\n🎥 處理側面影片: {Path(video_side).name}")
            side_entries, side_exits = detect_ball_entries_optimized(
                video_side, yolo_tennis_ball_model, confidence_threshold=0.5,
                ball_entry_direction="right", enable_exit_detection=True, exit_timeout=1.5
            )
            
            side_segments_folder = base_output_dir / "segments" / "side"
            side_segments = segment_video_dynamic(
                video_side, side_entries, side_exits, side_segments_folder,
                name, "side", preview_start_time=-0.5
            )
            
            side_ball_data = []
            for entry, exit, segment in zip(side_entries, side_exits, side_segments):
                if segment: side_ball_data.append((entry, exit, segment))
                
            print(f"\n🎥 處理45度影片: {Path(video_45).name}")
            deg45_entries, deg45_exits = detect_ball_entries_optimized(
                video_45, yolo_tennis_ball_model, confidence_threshold=0.5,
                ball_entry_direction="right", enable_exit_detection=True, exit_timeout=1.5
            )
            
            deg45_segments_folder = base_output_dir / "segments" / "45deg"
            deg45_segments = segment_video_dynamic(
                video_45, deg45_entries, deg45_exits, deg45_segments_folder,
                name, "45", preview_start_time=-0.5
            )
            
            deg45_ball_data = []
            for entry, exit, segment in zip(deg45_entries, deg45_exits, deg45_segments):
                if segment: deg45_ball_data.append((entry, exit, segment))
        
        # 球對對齊
        ball_pairs = align_ball_segments(side_ball_data, deg45_ball_data, name)
        
        if not ball_pairs:
            print("⚠️ 未偵測到任何球對，流程結束")
            clip_queue.put(None) # 結束 worker
            return False
            
        print(f"📊 共偵測到 {len(ball_pairs)} 個球對，開始加入排程...")
        
        # === 修改：立即將任務加入 Queue，實現真正的管線化 ===
        for i, ball_pair in enumerate(ball_pairs):
            ball_number = i + 1
            
            # 為每顆球建立獨立資料夾
            ball_folder = base_output_dir / f"trajectory_{ball_number}"
            ball_folder.mkdir(exist_ok=True)
            
            # 準備計時結果字典
            timing_results = {}
            
            # 建構分割結果結構 (為了相容 process_single_video_set)
            ball_segmentation = {
                "ball_pairs": [ball_pair],
                "side_segments": [], 
                "deg45_segments": []
            }
            
            # 將任務打包丟入 Queue
            task = (P1, P2, video_side, video_45, name, str(ball_folder), timing_results, ball_segmentation, ball_number)
            
            clip_queue.put(task)
            print(f"📤 [排程] 第 {ball_number} 顆球已加入分析佇列")
            
            # 讓分析執行緒有機會在分割下一顆球之前開始工作
            time.sleep(0.1) 
            
    except Exception as e:
        print(f"❌ 分割過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    # 所有分割任務已送出，發送結束信號
    clip_queue.put(None)
    
    # 等待所有分析完成
    print("\n⏳ 等待所有分析任務完成...")
    worker_thread.join()
    
    total_time = time.perf_counter() - start_total_time
    print(f"\n🎉 所有流程結束！總耗時: {total_time:.2f} 秒")
    
    # 生成總結報告
    generate_processing_summary(base_output_dir, name, {"total_pipeline_time": total_time}, total_time)
    
    return True

if __name__ == "__main__":
    # 設定參數
    video_side = "input_videos/tennis_side.MP4"
    video_45 = "input_videos/tennis_45.MP4"
    name = "test_user"
    P1 = [0, 0, 0]  # 根據實際情況設定
    P2 = [0, 0, 0]  # 根據實際情況設定
    knn_dataset_path = "knn_dataset.json"
    
    # 檢查檔案是否存在
    if not os.path.exists(video_side) or not os.path.exists(video_45):
        print(f"❌ 找不到影片檔案: {video_side} 或 {video_45}")
        print("💡 請確保 input_videos 資料夾中有 tennis_side.MP4 和 tennis_45.MP4")
    else:
        # 執行管線化處理
        process_video_pipeline(video_side, video_45, name, P1, P2, knn_dataset_path)
