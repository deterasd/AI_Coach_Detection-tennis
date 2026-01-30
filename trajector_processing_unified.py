"""
統一輸出管理的軌跡處理流程 - 重構版
========================================
支援多球分析，每顆球會有獨立的資料夾和完整的分析結果

主要流程 (11步驟):
    1. 2D軌跡分析 → 2. 2D平滑 → 3. 影片處理 → 4. 影片同步 → 5. 影片合併
    → 6. 軌跡同步 → 7. 3D分析 → 8. 3D平滑 → 9. 擊球範圍 → 10. KNN → 11. GPT

分割模組已移至: trajectory_video_segmentation.py

使用方式:
    from trajector_processing_unified import processing_trajectory_unified
    
    success = processing_trajectory_unified(
        P1, P2, pose_model, ball_model, paddle_model,
        video_side, video_45, knn_dataset, name
    )

或直接執行:
    python trajector_processing_unified.py
"""

import time
import os
import json
import shutil
import math
import gc
import torch
import psutil
from pathlib import Path
from ultralytics import YOLO

# 從分割模組導入
from trajectory_video_segmentation import (
    process_video_segmentation,
    align_ball_segments,
    create_ball_specific_segments
)


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
# 記憶體管理
# ============================================

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


# ============================================
# 輸出資料夾包裝函數
# ============================================

def analyze_trajectory_with_output_folder(pose_model, ball_model, video_path, batch_size, output_folder, paddle_model=None):
    """分析軌跡並將結果保存到指定資料夾"""
    from trajectory_2D_output import process_video_batch
    
    if paddle_model is None:
        try:
            paddle_model = YOLO('model/tennispaddle.pt')
            print("📦 已自動載入球拍模型: model/tennispaddle.pt")
        except Exception as e:
            print(f"⚠️ 球拍模型載入失敗: {e}")
            paddle_model = ball_model
    
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


# ============================================
# 核心處理流程
# ============================================

def processing_trajectory_unified(P1, P2, yolo_pose_model, yolo_tennis_ball_model, yolo_paddle_model,
                                video_side, video_45, knn_dataset, name,
                                ball_entry_direction="right", confidence_threshold=0.5,
                                output_folder=None, segment_videos=True):
    """
    統一輸出管理的完整軌跡處理流程
    支援多球檢測，每顆球會產生獨立的資料夾
    
    Args:
        P1, P2: 相機校正參數
        yolo_pose_model: 姿勢偵測 YOLO 模型
        yolo_tennis_ball_model: 網球偵測 YOLO 模型
        yolo_paddle_model: 球拍偵測 YOLO 模型
        video_side: 側面影片路徑
        video_45: 45度影片路徑
        knn_dataset: KNN 資料集路徑
        name: 使用者名稱/輸出前綴
        ball_entry_direction: 球進入方向 ("right" 或 "left")
        confidence_threshold: 偵測信心度閾值
        output_folder: 輸出資料夾 (預設為 trajectory/{name}__trajectory)
        segment_videos: 是否啟用影片自動分割
    
    Returns:
        bool: 處理是否成功
    """
    
    if output_folder is None:
        output_folder = Path("trajectory") / f"{name}__trajectory"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timing_results = {}
    start_total = time.perf_counter()
    
    print(f"🎾 開始 {name} 的完整軌跡分析流程")
    print(f"📁 輸出資料夾: {output_folder}")
    print("=" * 60)
    
    # 檢查系統資源
    print("\n🔍 檢查系統資源...")
    clear_all_memory()
    gpu_ok = check_gpu_memory()
    ram_ok = check_system_memory()
    
    if not ram_ok:
        print("⚠️ 系統記憶體不足，將自動使用 CPU 模式")
    
    try:
        # 步驟0：影片自動分割（如果啟用）
        segmentation_results = None
        if segment_videos:
            print(f"\n📹 步驟0：影片自動分割...")
            start_segment = time.perf_counter()
            
            segmentation_results = process_video_segmentation(
                video_side, video_45, yolo_tennis_ball_model, name, output_folder,
                ball_entry_direction, confidence_threshold
            )
            
            timing_results['影片自動分割'] = time.perf_counter() - start_segment
            print(f"✅ 影片分割完成，耗時：{timing_results['影片自動分割']:.4f} 秒")
            
            clear_all_memory()
        else:
            print("\n⚠️ 影片分割功能已停用")
        
        # 根據分割結果決定處理方式
        if segmentation_results and len(segmentation_results.get("ball_pairs", [])) > 0:
            # 多球處理流程
            success = process_multiple_balls(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model,
                start_total_time=start_total
            )
        else:
            # 單球或未分割處理流程
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model
            )
        
        if success:
            total_time = time.perf_counter() - start_total
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
            
            # 生成處理摘要
            generate_processing_summary(output_folder, name, timing_results, total_time)
            
        return success
        
    except Exception as e:
        print(f"\n💥 處理過程發生錯誤: {e}")
        
        # 記錄錯誤到日誌
        error_log = output_folder / "logs" / "processing_error.log"
        error_log.parent.mkdir(exist_ok=True)
        
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"錯誤時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用者: {name}\n")
            f.write(f"錯誤訊息: {str(e)}\n")
            f.write(f"輸入影片: {video_side}, {video_45}\n")
            
        return False


def process_multiple_balls(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                          video_side, video_45, knn_dataset, 
                          name, output_folder, timing_results, segmentation_results, paddle_model=None,
                          start_total_time=None):
    """
    處理多球分析 - 為每個球對創建獨立的分析資料夾
    """
    print(f"\n開始多球分析處理 - {name}")
    print(f"偵測到 {len(segmentation_results['ball_pairs'])} 個球對")
    
    # 創建球特定的分割片段
    segmentation_results = create_ball_specific_segments(segmentation_results, output_folder, name)
    
    ball_pairs = segmentation_results["ball_pairs"]
    overall_success = True
    
    for i, ball_pair in enumerate(ball_pairs):
        ball_number = ball_pair["ball_number"]
        print(f"\n處理第 {ball_number} 顆球...")
        
        # 創建該球的專屬資料夾
        ball_folder = os.path.join(output_folder, f"trajectory_{ball_number}")
        os.makedirs(ball_folder, exist_ok=True)
        
        # 為該球創建個別的segmentation_results
        ball_segmentation = {
            "side_segments": [ball_pair["side_data"]] if ball_pair["side_data"] else [],
            "deg45_segments": [ball_pair["deg45_data"]] if ball_pair["deg45_data"] else [],
            "ball_pairs": [ball_pair]
        }
        
        try:
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, ball_folder, timing_results, ball_segmentation, paddle_model
            )
            
            if success:
                print(f"✅ 第 {ball_number} 顆球處理完成")
                
                # 記錄第一顆球完成時間
                if i == 0 and start_total_time is not None:
                    first_ball_time = time.perf_counter() - start_total_time
                    timing_results['第一顆球執行完成'] = first_ball_time
                    print(f"⏱️ 第一顆球執行完成時間: {first_ball_time:.4f} 秒")
                    
                    # 建立標記檔案，讓前端能即時通知
                    try:
                        ready_file_path = os.path.join(ball_folder, "ready.txt")
                        with open(ready_file_path, "w", encoding='utf-8') as f:
                            f.write(f"First ball ready at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"🚩 [第一球完成標記] 已建立：{ready_file_path}")
                    except Exception as e:
                        print(f"⚠️ 無法建立第一球標記檔案: {e}")
            else:
                print(f"⚠️ 第 {ball_number} 顆球處理有部分問題")
                
        except Exception as e:
            print(f"❌ 第 {ball_number} 顆球處理發生錯誤: {str(e)}")
            print(f"⚠️ 跳過第 {ball_number} 顆球，繼續處理下一顆...")
            import traceback
            traceback.print_exc()
    
    if overall_success:
        print(f"\n🎾 所有球對分析完成！共處理 {len(ball_pairs)} 個球對")
    else:
        print(f"\n⚠️ 部分球對處理失敗")
    
    return overall_success


def process_single_video_set(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                           video_side, video_45, knn_dataset, 
                           name, output_folder, timing_results, segmentation_results=None, paddle_model=None):
    """
    處理單組影片的完整流程 (11步驟)
    """
    try:
        # 從output_folder推導球號
        output_folder_path = Path(output_folder)
        folder_name = output_folder_path.name
        if folder_name.startswith("trajectory_"):
            ball_number = folder_name.split("_")[-1]
            segment_name = f"{name}__{ball_number}"
        else:
            segment_name = f"{name}__1"
        
        # 匯入處理模組
        from trajector_2D_smoothing import smooth_2D_trajectory
        from video_detection import process_video
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
            ball_pair = segmentation_results["ball_pairs"][0]
            
            if ball_pair.get("side_data") and ball_pair["side_data"].get("segment"):
                segment_value = ball_pair["side_data"]["segment"]
                segment_path = segment_value.get("file_path") if isinstance(segment_value, dict) else segment_value
                if segment_path:
                    actual_video_side = os.path.abspath(segment_path) if not os.path.isabs(segment_path) else segment_path
                    print(f"🎬 使用側面分割片段: {os.path.basename(actual_video_side)}")
                
            if ball_pair.get("deg45_data") and ball_pair["deg45_data"].get("segment"):
                segment_value = ball_pair["deg45_data"]["segment"]
                segment_path = segment_value.get("file_path") if isinstance(segment_value, dict) else segment_value
                if segment_path:
                    actual_video_45 = os.path.abspath(segment_path) if not os.path.isabs(segment_path) else segment_path
                    print(f"🎬 使用45度分割片段: {os.path.basename(actual_video_45)}")
        
        # 顯示分割結果摘要
        if segmentation_results:
            print(f"\n📊 影片分割摘要:")
            print(f"   側面片段: {len(segmentation_results['side_segments'])} 個")
            print(f"   45度片段: {len(segmentation_results['deg45_segments'])} 個")
        
        # 步驟1：分析2D軌跡
        print("\n步驟1：分析2D軌跡...")
        start = time.perf_counter()
        
        trajectory_side = analyze_trajectory_with_output_folder(
            yolo_pose_model, yolo_tennis_ball_model, actual_video_side, 28, output_folder, paddle_model)
        trajectory_45 = analyze_trajectory_with_output_folder(
            yolo_pose_model, yolo_tennis_ball_model, actual_video_45, 28, output_folder, paddle_model)
        
        timing_results['2D軌跡分析'] = time.perf_counter() - start
        print(f"✅ 2D軌跡分析完成，耗時：{timing_results['2D軌跡分析']:.4f} 秒")
        clear_all_memory()

        # 步驟2：2D軌跡平滑處理
        print("\n步驟2：2D軌跡平滑處理...")
        start = time.perf_counter()
        
        trajectory_side_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_side, output_folder)
        trajectory_45_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_45, output_folder)
        
        timing_results['2D平滑處理'] = time.perf_counter() - start
        print(f"✅ 2D平滑處理完成，耗時：{timing_results['2D平滑處理']:.4f} 秒")
        clear_all_memory()

        # 步驟3：影片處理
        print("\n步驟3：影片物件偵測處理...")
        start = time.perf_counter()
        
        print("📹 處理側面影片...")
        video_side_processed = process_video_with_output_folder(
            actual_video_side, output_folder,
            ball_model=yolo_tennis_ball_model,
            pose_model=yolo_pose_model,
            paddle_model=paddle_model,
            json_path=trajectory_side
        )
        clear_all_memory()
        
        print("📹 處理45度影片...")
        video_45_processed = process_video_with_output_folder(
            actual_video_45, output_folder,
            ball_model=yolo_tennis_ball_model,
            pose_model=yolo_pose_model,
            paddle_model=paddle_model,
            json_path=trajectory_45
        )
        clear_all_memory()
        
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
        
        trajectory_3d_smoothing_path = smooth_3D_trajectory(trajectory_3d_path)
        
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
        trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing_path, start_frame, end_frame)
        
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
        
        trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing_path)
        knn_feedback_path = save_knn_feedback_with_output_folder(trajectory_knn_suggestion, output_folder, segment_name)
        
        timing_results['KNN 分析'] = time.perf_counter() - start
        print(f"✅ KNN分析完成，耗時：{timing_results['KNN 分析']:.4f} 秒")

        # 步驟11：GPT反饋生成
        print("\n步驟11：生成GPT反饋...")
        start = time.perf_counter()
        
        try:
            trajectory_gpt_suggestion = generate_feedback_data_only(trajectory_3d_swing_range, knn_feedback_path)
            
            if isinstance(trajectory_gpt_suggestion, dict) and trajectory_gpt_suggestion.get('error', False):
                error_type = trajectory_gpt_suggestion.get('error_type', 'unknown')
                if error_type == 'quota_exceeded':
                    print("⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代")
                else:
                    print(f"⚠️ GPT API 發生錯誤 ({error_type})，已使用 KNN 分析結果作為替代")
            
            gpt_feedback_path = save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
            
            timing_results['GPT 反饋生成'] = time.perf_counter() - start
            print(f"✅ GPT反饋生成完成，耗時：{timing_results['GPT 反饋生成']:.4f} 秒")
            
        except Exception as e:
            print(f"⚠️ GPT反饋生成失敗: {e}")
            
            trajectory_gpt_suggestion = {
                "problem_frame": "N/A",
                "suggestion": "GPT功能暫時無法使用，請參考KNN分析結果",
                "error": True,
                "error_type": "processing_error"
            }
            
            try:
                gpt_feedback_path = save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
            except:
                print("⚠️ 無法保存 GPT 反饋檔案")
            
            timing_results['GPT 反饋生成'] = time.perf_counter() - start

        # 建立完成標記檔案
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


# ============================================
# 主程式入口
# ============================================

if __name__ == "__main__":
    """
    直接執行模式 - 測試用
    實際使用時請透過 trajector_processing_simple_test.py 執行
    """
    import argparse
    
    print("🎾 網球軌跡處理系統 - 統一版")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="網球軌跡處理系統")
    parser.add_argument("--video_side", type=str, help="側面影片路徑")
    parser.add_argument("--video_45", type=str, help="45度影片路徑")
    parser.add_argument("--name", type=str, default="test_user", help="使用者名稱")
    parser.add_argument("--direction", type=str, default="right", choices=["right", "left"], help="球進入方向")
    parser.add_argument("--output", type=str, default=None, help="輸出資料夾")
    parser.add_argument("--no-segment", action="store_true", help="停用影片分割")
    
    args = parser.parse_args()
    
    if not args.video_side or not args.video_45:
        print("\n⚠️ 請提供影片路徑！")
        print("\n使用方式:")
        print("  python trajector_processing_unified.py --video_side <側面影片> --video_45 <45度影片> --name <名稱>")
        print("\n範例:")
        print("  python trajector_processing_unified.py --video_side input_videos/1/side.mp4 --video_45 input_videos/1/45.mp4 --name 測試使用者")
        print("\n選項:")
        print("  --direction right/left  球進入方向 (預設: right)")
        print("  --output <路徑>         指定輸出資料夾")
        print("  --no-segment            停用影片自動分割")
        exit(1)
    
    # 載入模型
    print("\n📦 載入 YOLO 模型...")
    try:
        yolo_pose_model = YOLO('yolov8x-pose-p6.pt')
        yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
        yolo_paddle_model = YOLO('model/tennispaddle.pt')
        print("✅ 模型載入完成")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        exit(1)
    
    # 載入相機校正參數
    print("\n📷 載入相機校正參數...")
    try:
        from binocular_correction.binocular_correction import compute_projection_matrix
        P1, P2 = compute_projection_matrix()
        print("✅ 相機校正參數載入完成")
    except Exception as e:
        print(f"❌ 相機校正參數載入失敗: {e}")
        P1, P2 = None, None
    
    # 執行處理
    print(f"\n🎬 開始處理...")
    print(f"   側面影片: {args.video_side}")
    print(f"   45度影片: {args.video_45}")
    print(f"   使用者: {args.name}")
    print(f"   球進入方向: {args.direction}")
    
    success = processing_trajectory_unified(
        P1=P1,
        P2=P2,
        yolo_pose_model=yolo_pose_model,
        yolo_tennis_ball_model=yolo_tennis_ball_model,
        yolo_paddle_model=yolo_paddle_model,
        video_side=args.video_side,
        video_45=args.video_45,
        knn_dataset='knn_dataset.json',
        name=args.name,
        ball_entry_direction=args.direction,
        output_folder=args.output,
        segment_videos=not args.no_segment
    )
    
    if success:
        print("\n✅ 處理完成！")
    else:
        print("\n❌ 處理失敗")
        exit(1)
