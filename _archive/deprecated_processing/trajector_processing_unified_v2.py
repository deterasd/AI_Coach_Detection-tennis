"""
網球軌跡處理統一模組 (重構版)
==============================
整合所有軌跡處理功能的主要模組

重構改進:
- 使用 processing_config.py 集中管理配置
- 使用 processing_utils.py 統一工具函數
- 使用 timer 上下文管理器簡化計時邏輯
- 將重複的檔案移動邏輯提取到工具函數
- 保持與原版完全相容的 API

使用方式:
    python trajector_processing_unified.py --video_side <側面影片> --video_45 <45度影片>
    
或透過 trajector_processing_simple_test.py 執行
"""

import os
import json
import time
import shutil
from pathlib import Path

# 導入配置和工具模組
from processing_config import StepNames, PathConfig
from processing_utils import (
    NanToNullEncoder,
    clear_all_memory,
    check_system_memory,
    check_gpu_memory,
    timer,
    ensure_directory,
    create_ready_marker,
    save_feedback_to_folder,
    generate_processing_summary
)

# 導入影片分割模組
from trajectory_video_segmentation import (
    process_video_segmentation,
    create_ball_specific_segments
)


# ============================================
# 輸出資料夾包裝函數
# ============================================

def _move_result_to_folder(result_path, output_folder, new_name=None):
    """將處理結果移動到輸出資料夾"""
    if not result_path or not Path(result_path).exists():
        return result_path
    
    source = Path(result_path)
    target = Path(output_folder) / (new_name if new_name else source.name)
    
    if source.resolve() != target.resolve():
        ensure_directory(output_folder)
        shutil.move(str(source), str(target))
        return str(target)
    return result_path


def analyze_trajectory_with_output_folder(video_path, output_folder, segment_name):
    """2D軌跡分析並輸出到指定資料夾"""
    from trajectory_2D_output import analyze_trajectory
    result = analyze_trajectory(video_path)
    return _move_result_to_folder(
        result, output_folder, 
        f"{segment_name}_2D_trajectory_analysis.json"
    )


def smooth_2D_trajectory_with_output_folder(trajectory_path, output_path):
    """2D軌跡平滑並輸出到指定路徑"""
    from trajector_2D_smoothing import smooth_2D_trajectory
    result = smooth_2D_trajectory(trajectory_path)
    if result and Path(result).exists():
        if Path(result).resolve() != Path(output_path).resolve():
            shutil.move(result, output_path)
        return output_path
    return result


def process_video_with_output_folder(video_path, output_folder, ball_model, pose_model, paddle_model, json_path):
    """影片處理並輸出到指定資料夾"""
    from video_detection_optimized import process_video
    return process_video(
        video_path, output_folder,
        ball_model=ball_model,
        pose_model=pose_model,
        paddle_model=paddle_model,
        json_path=json_path
    )


def save_knn_feedback_with_output_folder(knn_result, output_folder, segment_name):
    """儲存KNN反饋到指定資料夾"""
    return save_feedback_to_folder(knn_result, output_folder, segment_name, "knn")


def save_gpt_feedback_with_output_folder(gpt_result, output_folder, segment_name):
    """儲存GPT反饋到指定資料夾"""
    return save_feedback_to_folder(gpt_result, output_folder, segment_name, "gpt")


# ============================================
# 主要處理函數
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
    # 設定輸出資料夾
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
    gpu_ok, _ = check_gpu_memory()
    ram_ok, _ = check_system_memory()
    
    if not ram_ok:
        print("⚠️ 系統記憶體不足，將自動使用 CPU 模式")
    
    try:
        # 步驟0：影片自動分割（如果啟用）
        segmentation_results = None
        if segment_videos:
            with timer(f"步驟0：{StepNames.SEGMENTATION}", timing_results):
                segmentation_results = process_video_segmentation(
                    video_side, video_45, yolo_tennis_ball_model, name, output_folder,
                    ball_entry_direction, confidence_threshold
                )
            clear_all_memory()
        else:
            print("\n⚠️ 影片分割功能已停用")
        
        # 根據分割結果決定處理方式
        if segmentation_results and len(segmentation_results.get("ball_pairs", [])) > 0:
            success = _process_multiple_balls(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model,
                start_total_time=start_total
            )
        else:
            success = _process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model
            )
        
        if success:
            total_time = time.perf_counter() - start_total
            _print_timing_summary(name, timing_results, total_time)
            generate_processing_summary(str(output_folder), name, timing_results, total_time)
            
        return success
        
    except Exception as e:
        print(f"\n💥 處理過程發生錯誤: {e}")
        _log_error(output_folder, name, e, video_side, video_45)
        return False


def _print_timing_summary(name, timing_results, total_time):
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


def _log_error(output_folder, name, error, video_side, video_45):
    """記錄錯誤到日誌"""
    error_log = Path(output_folder) / "logs" / "processing_error.log"
    error_log.parent.mkdir(exist_ok=True)
    
    with open(error_log, 'w', encoding='utf-8') as f:
        f.write(f"錯誤時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用者: {name}\n")
        f.write(f"錯誤訊息: {str(error)}\n")
        f.write(f"輸入影片: {video_side}, {video_45}\n")


# ============================================
# 多球處理
# ============================================

def _process_multiple_balls(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                           video_side, video_45, knn_dataset, 
                           name, output_folder, timing_results, segmentation_results, paddle_model=None,
                           start_total_time=None):
    """處理多球分析 - 為每個球對創建獨立的分析資料夾"""
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
        
        # 為該球創建個別的 segmentation_results
        ball_segmentation = {
            "side_segments": [ball_pair["side_data"]] if ball_pair["side_data"] else [],
            "deg45_segments": [ball_pair["deg45_data"]] if ball_pair["deg45_data"] else [],
            "ball_pairs": [ball_pair]
        }
        
        try:
            success = _process_single_video_set(
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
                    
                    # 建立標記檔案
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


# ============================================
# 單組影片處理 (11步驟核心流程)
# ============================================

def _process_single_video_set(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                             video_side, video_45, knn_dataset, 
                             name, output_folder, timing_results, segmentation_results=None, paddle_model=None):
    """處理單組影片的完整流程 (11步驟)"""
    try:
        # 從 output_folder 推導球號
        output_folder_path = Path(output_folder)
        folder_name = output_folder_path.name
        
        if folder_name.startswith("trajectory_"):
            ball_number = folder_name.replace("trajectory_", "")
            segment_name = f"{name}_ball_{ball_number}"
        else:
            segment_name = name
        
        # 決定實際要處理的影片
        actual_video_side = video_side
        actual_video_45 = video_45
        
        if segmentation_results:
            if segmentation_results.get("ball_pairs"):
                ball_pair = segmentation_results["ball_pairs"][0]
                if ball_pair.get("side_data", {}).get("segment_path"):
                    actual_video_side = ball_pair["side_data"]["segment_path"]
                if ball_pair.get("deg45_data", {}).get("segment_path"):
                    actual_video_45 = ball_pair["deg45_data"]["segment_path"]
        
        print(f"\n📁 處理 {segment_name}")
        print(f"   側面影片: {Path(actual_video_side).name}")
        print(f"   45度影片: {Path(actual_video_45).name}")
        print(f"   輸出資料夾: {output_folder}")
        
        # 建立輸出路徑
        trajectory_side = str(Path(output_folder) / f"{segment_name}_side_trajectory.json")
        trajectory_45 = str(Path(output_folder) / f"{segment_name}_45_trajectory.json")
        trajectory_side_smoothing = str(Path(output_folder) / f"{segment_name}_side_smoothed.json")
        trajectory_45_smoothing = str(Path(output_folder) / f"{segment_name}_45_smoothed.json")
        
        # 延遲導入處理模組
        from trajectory_2D_output import analyze_trajectory
        from trajector_2D_smoothing import smooth_2D_trajectory
        from video_detection_optimized import process_video
        from video_sync import synchronize_videos
        from video_merge import combine_videos_ffmpeg
        from trajector_2D_sync import sync_trajectories
        from trajectory_3D_output import process_trajectories
        from trajector_3D_smoothing import smooth_3D_trajectory
        from trajector_3D_capture_swing_range import find_range, extract_frames
        from trajectory_knn import analyze_trajectory_knn
        from trajectory_gpt_single_feedback import generate_feedback_data_only
        
        # ========== 步驟2：2D軌跡分析與平滑 ==========
        with timer(f"步驟2：{StepNames.PREPROCESSING}", timing_results):
            # 側面影片
            print("📊 分析側面2D軌跡...")
            result = analyze_trajectory(actual_video_side)
            _move_result_to_folder(result, output_folder, f"{segment_name}_side_trajectory.json")
            
            result = smooth_2D_trajectory(trajectory_side)
            if result and Path(result).exists() and str(Path(result).resolve()) != str(Path(trajectory_side_smoothing).resolve()):
                shutil.move(result, trajectory_side_smoothing)
            
            # 45度影片
            print("📊 分析45度2D軌跡...")
            result = analyze_trajectory(actual_video_45)
            _move_result_to_folder(result, output_folder, f"{segment_name}_45_trajectory.json")
            
            result = smooth_2D_trajectory(trajectory_45)
            if result and Path(result).exists() and str(Path(result).resolve()) != str(Path(trajectory_45_smoothing).resolve()):
                shutil.move(result, trajectory_45_smoothing)
        
        clear_all_memory()
        
        # ========== 步驟3：影片物件偵測 ==========
        with timer(f"步驟3：{StepNames.VIDEO_DETECTION}", timing_results):
            print("📹 處理側面影片...")
            video_side_processed = process_video(
                actual_video_side, output_folder,
                ball_model=yolo_tennis_ball_model,
                pose_model=yolo_pose_model,
                paddle_model=paddle_model,
                json_path=trajectory_side
            )
            clear_all_memory()
            
            print("📹 處理45度影片...")
            video_45_processed = process_video(
                actual_video_45, output_folder,
                ball_model=yolo_tennis_ball_model,
                pose_model=yolo_pose_model,
                paddle_model=paddle_model,
                json_path=trajectory_45
            )
            clear_all_memory()
        
        # ========== 步驟4：影片同步 ==========
        with timer(f"步驟4：{StepNames.VIDEO_SYNC}", timing_results):
            synchronize_videos(
                video_side_processed, video_45_processed,
                trajectory_side_smoothing, trajectory_45_smoothing
            )
        
        # ========== 步驟5：合併影片 ==========
        with timer(f"步驟5：{StepNames.VIDEO_MERGE}", timing_results):
            merged_video = combine_videos_ffmpeg(video_45_processed, video_side_processed)
            if merged_video and Path(merged_video).exists():
                final_merged_path = Path(output_folder) / f"{segment_name}_full_video.mp4"
                shutil.move(merged_video, final_merged_path)
                print(f"📹 合併影片已移動到: {final_merged_path.name}")
        
        # ========== 步驟6：軌跡同步 ==========
        with timer(f"步驟6：{StepNames.TRAJECTORY_SYNC}", timing_results):
            sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
        
        # ========== 步驟7：3D軌跡分析 ==========
        with timer(f"步驟7：{StepNames.TRAJECTORY_3D}", timing_results):
            trajectory_3d_path = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
            if trajectory_3d_path and Path(trajectory_3d_path).exists():
                target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory).json"
                if Path(trajectory_3d_path).resolve() != target_path.resolve():
                    shutil.move(str(trajectory_3d_path), str(target_path))
                    trajectory_3d_path = str(target_path)
        
        # ========== 步驟8：3D軌跡平滑 ==========
        with timer(f"步驟8：{StepNames.SMOOTHING_3D}", timing_results):
            trajectory_3d_smoothing_path = smooth_3D_trajectory(trajectory_3d_path)
            if trajectory_3d_smoothing_path and Path(trajectory_3d_smoothing_path).exists():
                target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory_smoothed).json"
                if Path(trajectory_3d_smoothing_path).resolve() != target_path.resolve():
                    shutil.move(str(trajectory_3d_smoothing_path), str(target_path))
                    trajectory_3d_smoothing_path = str(target_path)
        
        # ========== 步驟9：有效擊球範圍判斷 ==========
        with timer(f"步驟9：{StepNames.SWING_RANGE}", timing_results):
            start_frame, end_frame = find_range(trajectory_side_smoothing)
            trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing_path, start_frame, end_frame)
            if trajectory_3d_swing_range and Path(trajectory_3d_swing_range).exists():
                target_path = Path(output_folder) / f"{segment_name}_segment(3D_trajectory_smoothed)_only_swing.json"
                if Path(trajectory_3d_swing_range).resolve() != target_path.resolve():
                    shutil.move(str(trajectory_3d_swing_range), str(target_path))
                    trajectory_3d_swing_range = str(target_path)
        
        # ========== 步驟10：KNN分析 ==========
        with timer(f"步驟10：{StepNames.KNN_ANALYSIS}", timing_results):
            trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing_path)
            knn_feedback_path = save_knn_feedback_with_output_folder(
                trajectory_knn_suggestion, output_folder, segment_name
            )
        
        # ========== 步驟11：GPT反饋生成 ==========
        with timer(f"步驟11：{StepNames.GPT_FEEDBACK}", timing_results):
            try:
                trajectory_gpt_suggestion = generate_feedback_data_only(trajectory_3d_swing_range, knn_feedback_path)
                
                if isinstance(trajectory_gpt_suggestion, dict) and trajectory_gpt_suggestion.get('error', False):
                    error_type = trajectory_gpt_suggestion.get('error_type', 'unknown')
                    if error_type == 'quota_exceeded':
                        print("⚠️ GPT API 配額不足，已使用 KNN 分析結果作為替代")
                    else:
                        print(f"⚠️ GPT API 發生錯誤 ({error_type})，已使用 KNN 分析結果作為替代")
                
                save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
                
            except Exception as e:
                print(f"⚠️ GPT反饋生成失敗: {e}")
                
                trajectory_gpt_suggestion = {
                    "problem_frame": "N/A",
                    "suggestion": "GPT功能暫時無法使用，請參考KNN分析結果",
                    "error": True,
                    "error_type": "processing_error"
                }
                
                try:
                    save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
                except:
                    print("⚠️ 無法保存 GPT 反饋檔案")
        
        # 建立完成標記檔案
        create_ready_marker(str(output_folder))
        
        return True
        
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
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
    from ultralytics import YOLO
    
    print("🎾 網球軌跡處理系統 - 統一版 (重構)")
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
        yolo_pose_model = YOLO(PathConfig.POSE_MODEL_PATH)
        yolo_tennis_ball_model = YOLO(PathConfig.BALL_MODEL_PATH)
        yolo_paddle_model = YOLO(PathConfig.PADDLE_MODEL_PATH)
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
        knn_dataset=PathConfig.DEFAULT_KNN_DATASET,
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
