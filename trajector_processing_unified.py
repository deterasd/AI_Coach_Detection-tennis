"""
統一輸出管理的軌跡處理流程
========================================
支援多球分析，每顆球會有獨立的資料夾和完整的分析結果

主要流程 (11步驟):
    1. 2D軌跡分析 → 2. 2D平滑 → 3. 影片處理 → 4. 影片同步 → 5. 影片合併
    → 6. 軌跡同步 → 7. 3D分析 → 8. 3D平滑 → 9. 擊球範圍 → 10. KNN → 11. GPT

使用方式:
    from trajector_processing_unified import processing_trajectory_unified
    
    success = processing_trajectory_unified(
        P1, P2, pose_model, ball_model, paddle_model,
        video_side, video_45, knn_dataset, name
    )
"""

import time
import os
from pathlib import Path

# 前置設施（日誌、GPU、輸出資料夾包裝函數等）
from trajector_processing_setup import (
    DetailedLogger, init_gpu_device, clear_all_memory,
    check_gpu_memory, check_system_memory, get_gpu_info,
    analyze_trajectory_with_output_folder,
    smooth_2D_trajectory_with_output_folder,
    process_video_with_output_folder,
    save_knn_feedback_with_output_folder,
    save_gpt_feedback_with_output_folder,
    move_to_output_folder,
    generate_processing_summary, write_ready_marker,
    resolve_segment_video_paths, get_segment_name, print_timing_summary,
    GPU_AVAILABLE, GPU_DEVICE,
)

# 核心處理模組
from video_sync import synchronize_videos
from video_merge import combine_videos_ffmpeg
from trajector_2D_sync import sync_trajectories
from trajector_2D_capture_swing_range import find_range
from trajectory_3D_output import process_trajectories
from trajector_3D_smoothing import smooth_3D_trajectory
from trajector_3D_capture_swing_range import extract_frames
from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
from trajectory_gpt_single_feedback import generate_feedback_data_only

# 分割模組
from trajectory_video_segmentation import (
    process_video_segmentation,
    align_ball_segments,
    create_ball_specific_segments
)


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
    """
    init_gpu_device()
    
    if output_folder is None:
        output_folder = Path("trajectory") / f"{name}__trajectory"
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 初始化日誌
    logger = DetailedLogger(output_folder / "logs")
    logger.info(f"🎾 開始 {name} 的完整軌跡分析流程")
    logger.info(f"📁 輸出資料夾: {output_folder}")
    logger.info(f"📋 日誌檔案: {logger.get_log_path()}")
    
    timing_results = {}
    start_total = time.perf_counter()
    
    # 檢查系統資源
    logger.info(f"系統資訊: {get_gpu_info()}")
    clear_all_memory()
    
    if not check_system_memory():
        logger.warning("系統記憶體不足")
    
    try:
        # 步驟0：影片自動分割（如果啟用）
        segmentation_results = None
        if segment_videos:
            start = time.perf_counter()
            segmentation_results = process_video_segmentation(
                video_side, video_45, yolo_tennis_ball_model, name, output_folder,
                ball_entry_direction, confidence_threshold
            )
            timing_results['影片自動分割'] = time.perf_counter() - start
            clear_all_memory()
        
        # 根據分割結果決定處理方式
        if segmentation_results and len(segmentation_results.get("ball_pairs", [])) > 0:
            success = process_multiple_balls(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model,
                start_total_time=start_total, logger=logger
            )
        else:
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, output_folder, timing_results, segmentation_results, yolo_paddle_model,
                logger=logger
            )
        
        if success:
            total_time = time.perf_counter() - start_total
            print_timing_summary(name, timing_results, total_time)
            generate_processing_summary(output_folder, name, timing_results, total_time)
            
        return success
        
    except Exception as e:
        logger.error(f"處理過程發生錯誤: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def process_multiple_balls(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                          video_side, video_45, knn_dataset, 
                          name, output_folder, timing_results, segmentation_results, paddle_model=None,
                          start_total_time=None, logger=None):
    """處理多球分析 - 為每個球對創建獨立的分析資料夾"""
    if logger is None:
        logger = DetailedLogger(Path(output_folder) / "logs")
    
    logger.info(f"偵測到 {len(segmentation_results['ball_pairs'])} 個球對")
    segmentation_results = create_ball_specific_segments(segmentation_results, output_folder, name)
    ball_pairs = segmentation_results["ball_pairs"]
    overall_success = True
    
    for i, ball_pair in enumerate(ball_pairs):
        ball_number = ball_pair["ball_number"]
        logger.info(f"\n處理第 {ball_number} 顆球...")
        
        ball_folder = os.path.join(output_folder, f"trajectory_{ball_number}")
        os.makedirs(ball_folder, exist_ok=True)
        
        ball_segmentation = {
            "side_segments": [ball_pair["side_data"]] if ball_pair["side_data"] else [],
            "deg45_segments": [ball_pair["deg45_data"]] if ball_pair["deg45_data"] else [],
            "ball_pairs": [ball_pair]
        }
        
        try:
            success = process_single_video_set(
                P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                video_side, video_45, knn_dataset, 
                name, ball_folder, timing_results, ball_segmentation, paddle_model,
                logger=logger
            )
            
            if success:
                logger.info(f"✅ 第 {ball_number} 顆球處理完成")
                if i == 0 and start_total_time is not None:
                    timing_results['第一顆球執行完成'] = time.perf_counter() - start_total_time
                    write_ready_marker(ball_folder, "First ball ready")
            else:
                logger.warning(f"第 {ball_number} 顆球處理有部分問題")
                
        except Exception as e:
            logger.error(f"第 {ball_number} 顆球處理發生錯誤: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
    
    return overall_success


def process_single_video_set(P1, P2, yolo_pose_model, yolo_tennis_ball_model,
                           video_side, video_45, knn_dataset, 
                           name, output_folder, timing_results, 
                           segmentation_results=None, paddle_model=None, logger=None):
    """
    處理單組影片的完整流程 (11步驟)
    --------------------------------
    保持簡潔
    """
    if logger is None:
        logger = DetailedLogger(Path(output_folder) / "logs")
    
    segment_name = get_segment_name(name, output_folder)
    logger.info(f"\n開始處理: {segment_name}")
    
    # 解析實際影片路徑（如果有分割結果，使用分割後的片段）
    actual_video_side, actual_video_45 = resolve_segment_video_paths(
        video_side, video_45, segmentation_results, logger
    )

    try:
        # ------------------------------
        # 步驟1：分析2D軌跡
        # ------------------------------
        start = time.perf_counter()
        trajectory_side = analyze_trajectory_with_output_folder(
            yolo_pose_model, yolo_tennis_ball_model, actual_video_side, 28, output_folder, paddle_model)
        trajectory_45 = analyze_trajectory_with_output_folder(
            yolo_pose_model, yolo_tennis_ball_model, actual_video_45, 28, output_folder, paddle_model)
        timing_results['2D軌跡分析'] = time.perf_counter() - start
        clear_all_memory()

        # ------------------------------
        # 步驟2：2D軌跡平滑處理
        # ------------------------------
        start = time.perf_counter()
        trajectory_side_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_side, output_folder)
        trajectory_45_smoothing = smooth_2D_trajectory_with_output_folder(trajectory_45, output_folder)
        timing_results['2D平滑處理'] = time.perf_counter() - start
        clear_all_memory()

        # ------------------------------
        # 步驟3：影片物件偵測處理
        # ------------------------------
        start = time.perf_counter()
        video_side_processed = process_video_with_output_folder(
            actual_video_side, output_folder,
            ball_model=yolo_tennis_ball_model, pose_model=yolo_pose_model,
            paddle_model=paddle_model, json_path=trajectory_side)
        clear_all_memory()
        
        video_45_processed = process_video_with_output_folder(
            actual_video_45, output_folder,
            ball_model=yolo_tennis_ball_model, pose_model=yolo_pose_model,
            paddle_model=paddle_model, json_path=trajectory_45)
        clear_all_memory()
        timing_results['影片處理'] = time.perf_counter() - start

        # ------------------------------
        # 步驟4：影片同步
        # ------------------------------
        start = time.perf_counter()
        synchronize_videos(video_side_processed, video_45_processed, 
                          trajectory_side_smoothing, trajectory_45_smoothing)
        timing_results['影片同步'] = time.perf_counter() - start

        # ------------------------------
        # 步驟5：合併影片
        # ------------------------------
        start = time.perf_counter()
        merged_video = combine_videos_ffmpeg(video_45_processed, video_side_processed)
        if merged_video and Path(merged_video).exists():
            move_to_output_folder(merged_video, output_folder, f"{segment_name}_full_video.mp4")
        timing_results['影片合併'] = time.perf_counter() - start

        # ------------------------------
        # 步驟6：軌跡同步
        # ------------------------------
        start = time.perf_counter()
        sync_trajectories(trajectory_side_smoothing, trajectory_45_smoothing)
        timing_results['軌跡同步'] = time.perf_counter() - start

        # ------------------------------
        # 步驟7：3D軌跡分析
        # ------------------------------
        start = time.perf_counter()
        trajectory_3d_path = process_trajectories(trajectory_side_smoothing, trajectory_45_smoothing, P1, P2)
        trajectory_3d_path = move_to_output_folder(
            trajectory_3d_path, output_folder, f"{segment_name}_segment(3D_trajectory).json")
        timing_results['3D軌跡分析'] = time.perf_counter() - start

        # ------------------------------
        # 步驟8：3D軌跡平滑處理
        # ------------------------------
        start = time.perf_counter()
        trajectory_3d_smoothing_path = smooth_3D_trajectory(trajectory_3d_path)
        trajectory_3d_smoothing_path = move_to_output_folder(
            trajectory_3d_smoothing_path, output_folder, f"{segment_name}_segment(3D_trajectory_smoothed).json")
        timing_results['3D平滑處理'] = time.perf_counter() - start

        # ------------------------------
        # 步驟9：有效擊球範圍判斷
        # ------------------------------
        start = time.perf_counter()
        start_frame, end_frame = find_range(trajectory_side_smoothing)
        trajectory_3d_swing_range = extract_frames(trajectory_3d_smoothing_path, start_frame, end_frame)
        trajectory_3d_swing_range = move_to_output_folder(
            trajectory_3d_swing_range, output_folder, f"{segment_name}_segment(3D_trajectory_smoothed)_only_swing.json")
        timing_results['有效擊球範圍判斷'] = time.perf_counter() - start

        # ------------------------------
        # 步驟10：KNN分析
        # ------------------------------
        start = time.perf_counter()
        trajectory_knn_suggestion = analyze_trajectory_knn(knn_dataset, trajectory_3d_smoothing_path)
        knn_feedback_path = save_knn_feedback_with_output_folder(trajectory_knn_suggestion, output_folder, segment_name)
        timing_results['KNN 分析'] = time.perf_counter() - start

        # ------------------------------
        # 步驟11：GPT反饋生成
        # ------------------------------
        start = time.perf_counter()
        trajectory_gpt_suggestion = generate_feedback_data_only(trajectory_3d_swing_range, knn_feedback_path)
        save_gpt_feedback_with_output_folder(trajectory_gpt_suggestion, output_folder, segment_name)
        timing_results['GPT 反饋生成'] = time.perf_counter() - start

        # 建立完成標記
        write_ready_marker(output_folder)
        logger.info(f"✅ {segment_name} 處理完成")
        return True

    except Exception as e:
        logger.error(f"處理失敗: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


# ============================================
# 主程式入口
# ============================================

if __name__ == "__main__":
    import argparse
    from ultralytics import YOLO
    
    init_gpu_device()
    
    print("🎾 網球軌跡處理系統 - 統一版")
    print(f"🔧 GPU 狀態: {get_gpu_info()}")
    
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
        print("  python trajector_processing_unified.py --video_side <側面影片> --video_45 <45度影片> --name <名稱>")
        exit(1)
    
    # 載入模型
    yolo_pose_model = YOLO('model/yolo11l-pose.pt')
    yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
    yolo_paddle_model = YOLO('model/yolov11x.pt')
    if GPU_AVAILABLE:
        yolo_pose_model.to(GPU_DEVICE)
        yolo_tennis_ball_model.to(GPU_DEVICE)
        yolo_paddle_model.to(GPU_DEVICE)
    
    # 載入相機校正參數
    try:
        from binocular_correction.binocular_correction import compute_projection_matrix
        P1, P2 = compute_projection_matrix()
    except Exception as e:
        print(f"❌ 相機校正參數載入失敗: {e}")
        P1, P2 = None, None
    
    # 執行處理
    success = processing_trajectory_unified(
        P1=P1, P2=P2,
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
    
    exit(0 if success else 1)
