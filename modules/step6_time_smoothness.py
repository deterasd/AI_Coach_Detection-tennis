

import json
import numpy as np
from numpy.linalg import norm
from datetime import datetime
import sys
import os

# Add project root to sys.path to fix relative imports when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 引入共用模組
from modules.utils import (
    get_keypoint_safely,
    calculate_distance,
    load_json_file,
    save_json_results,
    calculate_cv,
    detect_outliers_zscore,
    generate_output_path,
    KEYPOINT_NAMES_EN
)
from config import load_config, ValidationConfig


# ========================================================
# 核心分析函數
# ========================================================

def analyze_velocity_changes(data: list, config: ValidationConfig) -> dict:
    """
    分析速度變化率（一階導數）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 速度分析結果
    """
    fps = config.fps
    results = {}
    
    keypoints_to_analyze = getattr(config, 'time_smoothness_keypoints', KEYPOINT_NAMES_EN)

    for joint_name in keypoints_to_analyze:
        positions = []
        for frame in data:
            positions.append(get_keypoint_safely(frame, joint_name))
        
        velocities = []
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i - 1] is not None:
                disp = positions[i] - positions[i - 1]
                vel = norm(disp) * fps  # mm/s
                velocities.append(vel)
        
        if velocities:
            arr = np.array(velocities, dtype=float)
            results[joint_name] = {
                "mean_velocity_mm_s": float(np.mean(arr)),
                "max_velocity_mm_s": float(np.max(arr)),
                "std_velocity_mm_s": float(np.std(arr)),
                "cv_percent": calculate_cv(arr),
                "sample_count": len(arr),
                "series_data": [float(x) for x in arr]
            }
    
    return results


def analyze_acceleration_anomalies(data: list, config: ValidationConfig) -> dict:
    """
    分析加速度異常（二階導數）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 加速度異常分析結果
    """
    fps = config.fps
    results = {}
    keypoints_to_analyze = getattr(config, 'time_smoothness_keypoints', KEYPOINT_NAMES_EN)

    for joint_name in keypoints_to_analyze:
        positions = [get_keypoint_safely(frame, joint_name) for frame in data]
        
        velocities = []
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i - 1] is not None:
                velocities.append((positions[i] - positions[i - 1]) * fps)
            else:
                velocities.append(None)
        
        accelerations = []
        for i in range(1, len(velocities)):
            if velocities[i] is not None and velocities[i - 1] is not None:
                acc = (velocities[i] - velocities[i - 1]) * fps
                accelerations.append(norm(acc))
        
        if not accelerations:
            continue
        
        arr = np.array(accelerations, dtype=float)
        outlier_indices, _ = detect_outliers_zscore(arr, config.acceleration_sigma)
        
        outlier_details = [{"frame": int(idx + 2), "acceleration": float(arr[idx])} for idx in outlier_indices]
        
        results[joint_name] = {
            "mean_acceleration_mm_s2": float(np.mean(arr)),
            "max_acceleration_mm_s2": float(np.max(arr)),
            "std_acceleration_mm_s2": float(np.std(arr)),
            "outlier_count": len(outlier_indices),
            "outlier_rate": float(len(outlier_indices) / len(arr) * 100) if arr.size > 0 else 0,
            "sample_count": len(arr),
            "outlier_details": outlier_details,
            "series_data": [float(x) for x in arr]
        }
        
    return results


def analyze_jump_anomalies(data: list, config: ValidationConfig) -> dict:
    """
    分析異常跳躍
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 跳躍異常分析結果
    """
    results = {}
    keypoints_to_analyze = getattr(config, 'time_smoothness_keypoints', KEYPOINT_NAMES_EN)

    for joint_name in keypoints_to_analyze:
        positions = [get_keypoint_safely(frame, joint_name) for frame in data]
        
        displacements = []
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i - 1] is not None:
                disp = calculate_distance(positions[i - 1], positions[i])
                displacements.append(disp if disp is not None else 0)
            else:
                displacements.append(0)

        if not displacements:
            continue
        
        arr = np.array(displacements, dtype=float)
        large_jumps = [{"frame": int(i + 1), "displacement": float(disp)} for i, disp in enumerate(arr) if disp > config.max_frame_displacement]
        
        results[joint_name] = {
            "mean_displacement_mm": float(np.mean(arr)),
            "max_displacement_mm": float(np.max(arr)),
            "std_displacement_mm": float(np.std(arr)),
            "large_jump_count": len(large_jumps),
            "large_jump_rate": float(len(large_jumps) / len(arr) * 100) if arr.size > 0 else 0,
            "sample_count": len(arr),
            "large_jumps": large_jumps,
            "series_data": [float(x) for x in arr]
        }
        
    return results

def analyze_smoothness(data: list, config: ValidationConfig) -> dict:
    """
    分析整體平滑度 (Jerk)
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 平滑度分析結果
    """
    results = {}
    keypoints_to_analyze = getattr(config, 'time_smoothness_keypoints', KEYPOINT_NAMES_EN)

    for joint_name in keypoints_to_analyze:
        positions = [p for p in (get_keypoint_safely(frame, joint_name) for frame in data) if p is not None]
        
        if len(positions) < 4:
            continue
        
        # Jerk is the third derivative
        pos_arr = np.array(positions)
        v = np.diff(pos_arr, axis=0) * config.fps
        a = np.diff(v, axis=0) * config.fps
        j = np.diff(a, axis=0) * config.fps
        jerk_magnitude = norm(j, axis=1)

        if jerk_magnitude.size == 0:
            continue
        
        results[joint_name] = {
            "mean_jerk_mm_s3": float(np.mean(jerk_magnitude)),
            "max_jerk_mm_s3": float(np.max(jerk_magnitude)),
            "std_jerk_mm_s3": float(np.std(jerk_magnitude)),
            "sample_count": len(jerk_magnitude),
            "series_data": [float(x) for x in jerk_magnitude]
        }
        
    return results


def analyze_frequency_domain(data: list, config: ValidationConfig) -> dict:
    """
    分析頻域特徵（FFT）
    """
    results = {}
    keypoints_to_analyze = getattr(config, 'fft_analysis_joint', ['right_wrist', 'left_wrist', 'nose'])

    for joint_name in keypoints_to_analyze:
        positions = [get_keypoint_safely(frame, joint_name) for frame in data]
        pos_arr = np.array([p for p in positions if p is not None])

        if pos_arr.shape[0] < 10:
            continue

        def fft_analysis_axis(signal):
            n = len(signal)
            fft_result = np.fft.fft(signal)
            freqs = np.fft.fftfreq(n, 1.0 / config.fps)
            
            positive_freqs = freqs[:n // 2]
            magnitude = np.abs(fft_result[:n // 2])
            
            dominant_freq_idx = np.argmax(magnitude[1:]) + 1 if len(magnitude) > 1 else 0
            dominant_freq = float(positive_freqs[dominant_freq_idx])
            
            high_freq_threshold = config.high_frequency_threshold
            high_freq_mask = positive_freqs > high_freq_threshold
            high_freq_energy = float(np.sum(magnitude[high_freq_mask] ** 2))
            total_energy = float(np.sum(magnitude[1:] ** 2))
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0.0
            
            return {
                "dominant_frequency_hz": dominant_freq,
                "high_frequency_ratio": high_freq_ratio
            }
        
        results[joint_name] = {
            "x_axis": fft_analysis_axis(pos_arr[:, 0]),
            "y_axis": fft_analysis_axis(pos_arr[:, 1]),
            "z_axis": fft_analysis_axis(pos_arr[:, 2]),
            "sample_count": pos_arr.shape[0]
        }
    
    return results

def print_analysis_report(
    velocity: dict,
    acceleration: dict,
    jump_anomalies: dict,
    smoothness: dict,
    config: ValidationConfig
) -> None:
    """列印分析報告"""
    
    print("\n" + "=" * 100)
    print("【時間平滑度分析報告】")
    print("=" * 100)
    
    print("\n【1. 速度分析】")
    if velocity:
        for joint, stats in list(velocity.items())[:3]: # Print top 3
            print(f"  - {joint:<15}: Mean Vel: {stats['mean_velocity_mm_s']:.1f} mm/s, Max: {stats['max_velocity_mm_s']:.1f} mm/s")
    
    print("\n【2. 加速度異常】")
    if acceleration:
        total_outliers = sum(stats.get('outlier_count', 0) for stats in acceleration.values())
        print(f"  總異常數: {total_outliers}")
        for joint, stats in list(acceleration.items())[:3]:
            print(f"  - {joint:<15}: Outliers: {stats['outlier_count']} ({stats['outlier_rate']:.1f}%)")

    print("\n【3. 異常跳躍】")
    if jump_anomalies:
        total_jumps = sum(stats.get('large_jump_count', 0) for stats in jump_anomalies.values())
        print(f"  總大跳躍數: {total_jumps}")
        for joint, stats in list(jump_anomalies.items())[:3]:
            print(f"  - {joint:<15}: Jumps: {stats['large_jump_count']} (Max: {stats['max_displacement_mm']:.1f} mm)")

    print("\n【4. 平滑度 (Jerk)】")
    if smoothness:
        for joint, stats in list(smoothness.items())[:3]:
            print(f"  - {joint:<15}: Mean Jerk: {stats['mean_jerk_mm_s3']:.1f} mm/s³")


def validate_time_smoothness_analysis(
    json_3d_path: str,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    時間平滑度驗證分析（主函數）
    """
    config = load_config(config_path)
    print(f"\n載入數據: {json_3d_path}")
    data = load_json_file(json_3d_path)
    print(f"總幀數: {len(data)}")
    
    print("\n執行速度分析...")
    velocity_analysis = analyze_velocity_changes(data, config)
    
    print("執行加速度分析...")
    acceleration_anomalies = analyze_acceleration_anomalies(data, config)
    
    print("執行跳躍異常分析...")
    jump_anomalies = analyze_jump_anomalies(data, config)
    
    print("執行平滑度分析 (Jerk)...")
    smoothness_analysis = analyze_smoothness(data, config)
    
    print("執行頻域分析...")
    frequency_analysis = analyze_frequency_domain(data, config)
    
    print_analysis_report(
        velocity_analysis, acceleration_anomalies,
        jump_anomalies, smoothness_analysis, config
    )
    
    # 整合結果
    results = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "source_file": str(json_3d_path),
            "total_frames": int(len(data)),
            "analysis_type": "Time Smoothness Analysis"
        },
        "overall_summary": {
            "total_acceleration_outliers": sum(s.get('outlier_count', 0) for s in acceleration_anomalies.values()),
            "total_large_jumps": sum(s.get('large_jump_count', 0) for s in jump_anomalies.values()),
        },
        "velocity_analysis": velocity_analysis,
        "acceleration_anomalies": acceleration_anomalies,
        "jump_anomalies": jump_anomalies,
        "smoothness_analysis": smoothness_analysis,
        "frequency_domain_analysis": frequency_analysis,
    }
    
    
    if output_json_path is None:
        # 建立 results 資料夾
        results_dir = os.path.join(os.path.dirname(json_3d_path), 'Verification Result')
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d_path))[0]
        output_json_path = os.path.join(results_dir, f"{base_name}_step6_time_smoothness_results.json")
    
    save_json_results(results, output_json_path)
    print(f"\n[OK] 結果已儲存至: {output_json_path}")
    
    return results

# ========================================================
# 主程式
# ========================================================

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        json_3d_path = sys.argv[1]
        config_path = None
        output_json_path = None
        
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
            if arg == '--output' and i + 1 < len(sys.argv):
                output_json_path = sys.argv[i + 1]
    else:
        # Default for testing
        json_3d_path = "data/trajectory__new/tsung__19_45(3D_trajectory_smoothed).json"
        config_path = None
        output_json_path = None
        print("提示: 可使用命令列參數:")
        print("  python step6_time_smoothness.py <json_path> [--config <config>] [--output <output>]")
    
    try:
        validate_time_smoothness_analysis(json_3d_path, output_json_path, config_path)
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
