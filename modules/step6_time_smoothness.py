"""
Step 6: 時間平滑度驗證分析
驗證 3D 軌跡在時間維度的連續性和平滑度

功能：
  1. 速度變化率（一階導數）
  2. 加速度異常檢測（二階導數）
  3. 方向突變檢測
  4. 異常跳躍檢測
  5. 平滑度分析
  6. 頻域分析（FFT）
  7. 方向連續性增強檢測
"""

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
    fps = config.fps  # 統一使用 fps 參數
    joint_velocities = {}
    
    for joint_name in ["left_wrist", "right_wrist", "nose"]:
        positions = []
        
        for frame in data:
            pos = get_keypoint_safely(frame, joint_name)
            if pos is not None:
                positions.append(pos)
            else:
                positions.append(None)
        
        velocities = []
        for i in range(1, len(positions)):
            if positions[i] is not None and positions[i - 1] is not None:
                disp = positions[i] - positions[i - 1]
                vel = norm(disp) * fps  # mm/s
                velocities.append(vel)
        
        if velocities:
            arr = np.array(velocities, dtype=float)
            joint_velocities[joint_name] = {
                "mean_velocity_mm_s": float(np.mean(arr)),
                "max_velocity_mm_s": float(np.max(arr)),
                "std_velocity_mm_s": float(np.std(arr)),
                "cv_percent": calculate_cv(arr),
                "sample_count": len(arr),
                "series_data": [float(x) for x in arr]
            }
    
    return joint_velocities


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
    joint_name = "right_wrist"
    
    positions = []
    for frame in data:
        pos = get_keypoint_safely(frame, joint_name)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append(None)
    
    velocities = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i - 1] is not None:
            disp = positions[i] - positions[i - 1]
            vel = disp * fps  # mm/s
            velocities.append(vel)
        else:
            velocities.append(None)
    
    accelerations = []
    for i in range(1, len(velocities)):
        if velocities[i] is not None and velocities[i - 1] is not None:
            acc = (velocities[i] - velocities[i - 1]) * fps  # mm/s²
            acc_magnitude = norm(acc)
            accelerations.append(acc_magnitude)
    
    if not accelerations:
        return {}
    
    arr = np.array(accelerations, dtype=float)
    outlier_indices, outlier_mask = detect_outliers_zscore(arr, config.acceleration_sigma)
    
    # 收集異常詳情
    outlier_details = []
    for idx in outlier_indices:
        outlier_details.append({
            "frame": int(idx + 1),  # +1 因為加速度從第2幀開始
            "acceleration": float(arr[idx])
        })
    
    return {
        "joint": joint_name,
        "mean_acceleration_mm_s2": float(np.mean(arr)),
        "max_acceleration_mm_s2": float(np.max(arr)),
        "std_acceleration_mm_s2": float(np.std(arr)),
        "outlier_count": len(outlier_indices),
        "outlier_rate": float(len(outlier_indices) / len(arr) * 100),
        "sample_count": len(arr),
        "outlier_details": outlier_details,
        "series_data": [float(x) for x in arr]
    }


def analyze_direction_changes(data: list, config: ValidationConfig) -> dict:
    """
    分析方向突變
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 方向突變分析結果
    """
    joint_name = "right_wrist"
    
    positions = []
    for frame in data:
        pos = get_keypoint_safely(frame, joint_name)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append(None)
    
    direction_vectors = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i - 1] is not None:
            vec = positions[i] - positions[i - 1]
            direction_vectors.append(vec)
        else:
            direction_vectors.append(None)
    
    direction_changes = []
    sudden_change_count = 0
    sudden_changes = []
    
    for i in range(1, len(direction_vectors)):
        if direction_vectors[i] is not None and direction_vectors[i - 1] is not None:
            v1 = direction_vectors[i - 1]
            v2 = direction_vectors[i]
            
            # 避免零向量
            n1, n2 = norm(v1), norm(v2)
            if n1 > config.epsilon and n2 > config.epsilon:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos_theta)))
                direction_changes.append(angle_deg)
                
                if angle_deg > config.direction_change_sudden:
                    sudden_change_count += 1
                    sudden_changes.append({
                        "frame": int(i + 1),  # +1 因為方向變化從第2幀開始
                        "angle_change": float(angle_deg)
                    })
    
    if not direction_changes:
        return {}
    
    arr = np.array(direction_changes, dtype=float)
    
    return {
        "joint": joint_name,
        "mean_angle_change_deg": float(np.mean(arr)),
        "max_angle_change_deg": float(np.max(arr)),
        "std_angle_change_deg": float(np.std(arr)),
        "sudden_change_count": sudden_change_count,
        "sudden_change_rate": float(sudden_change_count / len(arr) * 100),
        "sample_count": len(arr),
        "sudden_changes": sudden_changes
    }


def analyze_jump_anomalies(data: list, config: ValidationConfig) -> dict:
    """
    分析異常跳躍
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 跳躍異常分析結果
    """
    joint_name = "right_wrist"
    
    positions = []
    for frame in data:
        pos = get_keypoint_safely(frame, joint_name)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append(None)
    
    displacements = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i - 1] is not None:
            disp = calculate_distance(positions[i - 1], positions[i])
            if disp is not None:
                displacements.append(disp)
    
    if not displacements:
        return {}
    
    arr = np.array(displacements, dtype=float)
    outlier_indices, outlier_mask = detect_outliers_zscore(arr, config.jump_detection_sigma)
    
    # 直接檢測超過閾值的跳躍並收集詳情
    large_jumps = []
    for i, disp in enumerate(arr):
        if disp > config.max_frame_displacement:
            large_jumps.append({
                "frame": int(i + 1),  # +1 因為位移從第2幀開始
                "displacement": float(disp)
            })
    
    large_jump_count = len(large_jumps)
    
    return {
        "joint": joint_name,
        "mean_displacement_mm": float(np.mean(arr)),
        "max_displacement_mm": float(np.max(arr)),
        "std_displacement_mm": float(np.std(arr)),
        "zscore_outlier_count": len(outlier_indices),
        "large_jump_count": large_jump_count,
        "large_jump_rate": float(large_jump_count / len(arr) * 100),
        "sample_count": len(arr),
        "large_jumps": large_jumps
    }


def analyze_smoothness(data: list, config: ValidationConfig) -> dict:
    """
    分析整體平滑度
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 平滑度分析結果
    """
    joint_name = "right_wrist"
    
    positions = []
    for frame in data:
        pos = get_keypoint_safely(frame, joint_name)
        if pos is not None:
            positions.append(pos)
    
    if len(positions) < 3:
        return {}
    
    # 二階差分（曲率變化）
    second_diffs = []
    for i in range(1, len(positions) - 1):
        diff2 = positions[i + 1] - 2 * positions[i] + positions[i - 1]
        second_diffs.append(norm(diff2))
    
    if not second_diffs:
        return {}
    
    arr = np.array(second_diffs, dtype=float)
    
    return {
        "joint": joint_name,
        "mean_second_diff_mm": float(np.mean(arr)),
        "std_second_diff_mm": float(np.std(arr)),
        "max_second_diff_mm": float(np.max(arr)),
        "sample_count": len(arr),
        "series_data": [float(x) for x in arr]
    }


def analyze_frequency_domain(data: list, config: ValidationConfig) -> dict:
    """
    分析頻域特徵（新增 - FFT）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 頻域分析結果
    """
    # 使用配置的關節，預設為 right_wrist
    joint_name = getattr(config, 'fft_analysis_joint', 'right_wrist')
    
    # 提取 X, Y, Z 軌跡
    x_positions = []
    y_positions = []
    z_positions = []
    
    for frame in data:
        pos = get_keypoint_safely(frame, joint_name)
        if pos is not None:
            x_positions.append(pos[0])
            y_positions.append(pos[1])
            z_positions.append(pos[2])
    
    if len(x_positions) < 10:
        return {}
    
    def fft_analysis(signal):
        n = len(signal)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1.0 / config.fps)
        
        # 只取正頻率部分
        positive_freqs = freqs[:n // 2]
        magnitude = np.abs(fft_result[:n // 2])
        
        # 主頻率
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # 排除 DC 分量
        dominant_freq = float(positive_freqs[dominant_freq_idx])
        
        # 高頻能量佔比
        high_freq_threshold = config.high_frequency_threshold  # Hz
        high_freq_mask = positive_freqs > high_freq_threshold
        high_freq_energy = float(np.sum(magnitude[high_freq_mask] ** 2))
        total_energy = float(np.sum(magnitude[1:] ** 2))  # 排除 DC
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0.0
        
        return {
            "dominant_frequency_hz": dominant_freq,
            "high_frequency_ratio": high_freq_ratio
        }
    
    x_fft = fft_analysis(x_positions)
    y_fft = fft_analysis(y_positions)
    z_fft = fft_analysis(z_positions)
    
    return {
        "joint": joint_name,
        "x_axis": x_fft,
        "y_axis": y_fft,
        "z_axis": z_fft,
        "sample_count": len(x_positions)
    }


def analyze_direction_continuity_enhanced(data: list, config: ValidationConfig) -> dict:
    """
    增強的方向連續性分析（新增）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 方向連續性分析結果
    """
    joint_name = "right_wrist"
    
    positions = []
    for frame in data:
        pos = get_keypoint_safely(frame, joint_name)
        if pos is not None:
            positions.append(pos)
        else:
            positions.append(None)
    
    # 計算方向向量
    direction_vectors = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i - 1] is not None:
            vec = positions[i] - positions[i - 1]
            direction_vectors.append(vec)
        else:
            direction_vectors.append(None)
    
    # 檢測方向反轉（180° 轉向）
    reversal_count = 0
    angle_changes = []
    
    for i in range(1, len(direction_vectors)):
        if direction_vectors[i] is not None and direction_vectors[i - 1] is not None:
            v1 = direction_vectors[i - 1]
            v2 = direction_vectors[i]
            
            n1, n2 = norm(v1), norm(v2)
            if n1 > config.epsilon and n2 > config.epsilon:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos_theta)))
                angle_changes.append(angle_deg)
                
                # 檢測接近 180° 的反轉
                if angle_deg > config.direction_reversal_threshold:
                    reversal_count += 1
    
    if not angle_changes:
        return {}
    
    arr = np.array(angle_changes, dtype=float)
    
    return {
        "joint": joint_name,
        "reversal_count": reversal_count,
        "reversal_rate": float(reversal_count / len(arr) * 100),
        "mean_angle_change_deg": float(np.mean(arr)),
        "median_angle_change_deg": float(np.median(arr)),
        "sample_count": len(arr)
    }


def analyze_detailed_anomalies(
    acceleration_data: dict,
    direction_change_data: dict,
    jump_anomalies_data: dict,
    direction_continuity_data: dict,
    config: ValidationConfig
) -> dict:
    """
    詳細分析時間平滑度異常
    
    參數:
        acceleration_data: 加速度異常數據
        direction_change_data: 方向變化數據
        jump_anomalies_data: 跳躍異常數據
        direction_continuity_data: 方向連續性數據
        config: 驗證配置
    
    返回:
        dict: 詳細異常分析結果
    """
    # 加速度異常分析
    accel_outliers = acceleration_data.get('outlier_details', [])
    
    # 計算動態閾值（使用平均值）
    if accel_outliers:
        accel_values = [x['acceleration'] for x in accel_outliers]
        accel_mean = np.mean(accel_values)
        accel_threshold = accel_mean
    else:
        accel_threshold = 1000.0  # 默認值
    
    # 分級: 嚴重 (>5x平均), 中等 (3-5x), 輕微 (<3x)
    accel_severe = [x for x in accel_outliers if x['acceleration'] > accel_threshold * 5]
    accel_moderate = [x for x in accel_outliers if accel_threshold * 3 < x['acceleration'] <= accel_threshold * 5]
    accel_mild = [x for x in accel_outliers if x['acceleration'] <= accel_threshold * 3]
    
    # 方向變化異常
    dir_changes = direction_change_data.get('sudden_changes', [])
    
    # 分級: 嚴重 (>120°), 中等 (90-120°), 輕微 (<90°)
    dir_severe = [x for x in dir_changes if x['angle_change'] > 120]
    dir_moderate = [x for x in dir_changes if 90 < x['angle_change'] <= 120]
    dir_mild = [x for x in dir_changes if x['angle_change'] <= 90]
    
    # 跳躍異常
    jumps = jump_anomalies_data.get('large_jumps', [])
    
    # 計算動態閾值（使用平均值）
    if jumps:
        jump_values = [x['displacement'] for x in jumps]
        jump_threshold = np.mean(jump_values)
    else:
        jump_threshold = 100.0  # 默認值
    
    # 分級: 嚴重 (>3x平均), 中等 (2-3x), 輕微 (1-2x)
    jump_severe = [x for x in jumps if x['displacement'] > jump_threshold * 3]
    jump_moderate = [x for x in jumps if jump_threshold * 2 < x['displacement'] <= jump_threshold * 3]
    jump_mild = [x for x in jumps if jump_threshold < x['displacement'] <= jump_threshold * 2]
    
    # 方向反轉（都算嚴重）
    reversals = direction_continuity_data.get('reversal_count', 0)
    
    # 連續異常區段檢測（加速度）
    continuous_segments = []
    if accel_outliers:
        accel_sorted = sorted(accel_outliers, key=lambda x: x['frame'])
        current_segment = None
        
        for item in accel_sorted:
            if current_segment is None:
                current_segment = {
                    "start_frame": item['frame'],
                    "end_frame": item['frame'],
                    "accelerations": [item['acceleration']]
                }
            elif item['frame'] - current_segment['end_frame'] <= 2:
                current_segment['end_frame'] = item['frame']
                current_segment['accelerations'].append(item['acceleration'])
            else:
                if len(current_segment['accelerations']) >= 3:
                    current_segment['duration'] = current_segment['end_frame'] - current_segment['start_frame'] + 1
                    current_segment['mean_acceleration'] = float(np.mean(current_segment['accelerations']))
                    current_segment['max_acceleration'] = float(np.max(current_segment['accelerations']))
                    continuous_segments.append(current_segment)
                current_segment = {
                    "start_frame": item['frame'],
                    "end_frame": item['frame'],
                    "accelerations": [item['acceleration']]
                }
        
        if current_segment and len(current_segment['accelerations']) >= 3:
            current_segment['duration'] = current_segment['end_frame'] - current_segment['start_frame'] + 1
            current_segment['mean_acceleration'] = float(np.mean(current_segment['accelerations']))
            current_segment['max_acceleration'] = float(np.max(current_segment['accelerations']))
            continuous_segments.append(current_segment)
        
        for seg in continuous_segments:
            del seg['accelerations']
    
    return {
        "total_anomaly_count": len(accel_outliers) + len(dir_changes) + len(jumps) + reversals,
        "acceleration_anomalies": {
            "total": len(accel_outliers),
            "severe": accel_severe,
            "moderate": accel_moderate,
            "mild": accel_mild,
            "threshold_used": float(accel_threshold) if accel_outliers else 0.0
        },
        "direction_change_anomalies": {
            "total": len(dir_changes),
            "severe": dir_severe,
            "moderate": dir_moderate,
            "mild": dir_mild
        },
        "jump_anomalies": {
            "total": len(jumps),
            "severe": jump_severe,
            "moderate": jump_moderate,
            "mild": jump_mild,
            "threshold_used": float(jump_threshold) if jumps else 0.0
        },
        "direction_reversals": {
            "count": reversals
        },
        "continuous_segments": sorted(continuous_segments, key=lambda x: x['duration'], reverse=True)
    }


def calculate_score(
    detailed_anomalies: dict,
    smoothness: dict,
    config: ValidationConfig
) -> dict:
    """
    計算時間平滑度評分 (0-100)
    
    參數:
        detailed_anomalies: 詳細異常分析結果
        smoothness: 平滑度分析結果
        config: 驗證配置
    
    返回:
        dict: 評分結果
    """
    score = 100.0
    deductions = []
    
    # 1. 加速度異常扣分
    accel = detailed_anomalies.get('acceleration_anomalies', {})
    accel_severe_count = len(accel.get('severe', []))
    accel_moderate_count = len(accel.get('moderate', []))
    accel_mild_count = len(accel.get('mild', []))
    
    deduction_accel = 0.0
    if accel_severe_count > 0:
        d = min(30.0, accel_severe_count * 5.0)
        deduction_accel += d
        deductions.append({"reason": f"嚴重加速度異常 ({accel_severe_count} 次)", "points": d})
    
    if accel_moderate_count > 0:
        d = min(20.0, accel_moderate_count * 2.0)
        deduction_accel += d
        deductions.append({"reason": f"中等加速度異常 ({accel_moderate_count} 次)", "points": d})
        
    if accel_mild_count > 0:
        d = min(10.0, accel_mild_count * 0.5)
        deduction_accel += d
        # 輕微異常不列入詳細扣分列表，除非扣分較多
        if d > 2.0:
            deductions.append({"reason": f"輕微加速度異常 ({accel_mild_count} 次)", "points": d})
            
    score -= deduction_accel
    
    # 2. 方向突變扣分
    dir_chg = detailed_anomalies.get('direction_change_anomalies', {})
    dir_severe_count = len(dir_chg.get('severe', []))
    dir_moderate_count = len(dir_chg.get('moderate', []))
    
    deduction_dir = 0.0
    if dir_severe_count > 0:
        d = min(30.0, dir_severe_count * 5.0)
        deduction_dir += d
        deductions.append({"reason": f"嚴重方向突變 ({dir_severe_count} 次)", "points": d})
        
    if dir_moderate_count > 0:
        d = min(20.0, dir_moderate_count * 2.0)
        deduction_dir += d
        deductions.append({"reason": f"中等方向突變 ({dir_moderate_count} 次)", "points": d})
        
    score -= deduction_dir
    
    # 3. 異常跳躍扣分 (最嚴重)
    jumps = detailed_anomalies.get('jump_anomalies', {})
    jump_severe_count = len(jumps.get('severe', []))
    jump_moderate_count = len(jumps.get('moderate', []))
    
    deduction_jump = 0.0
    if jump_severe_count > 0:
        d = min(40.0, jump_severe_count * 10.0)
        deduction_jump += d
        deductions.append({"reason": f"嚴重異常跳躍 ({jump_severe_count} 次)", "points": d})
        
    if jump_moderate_count > 0:
        d = min(20.0, jump_moderate_count * 5.0)
        deduction_jump += d
        deductions.append({"reason": f"中等異常跳躍 ({jump_moderate_count} 次)", "points": d})
        
    score -= deduction_jump
    
    # 4. 方向反轉扣分
    reversals = detailed_anomalies.get('direction_reversals', {}).get('count', 0)
    if reversals > 0:
        d = min(20.0, reversals * 2.0)
        score -= d
        deductions.append({"reason": f"方向反轉 ({reversals} 次)", "points": d})
        
    # 5. 整體平滑度扣分
    mean_second_diff = smoothness.get('mean_second_diff_mm', 0.0)
    deduction_smooth = 0.0
    if mean_second_diff > 20.0:
        deduction_smooth = 20.0
        deductions.append({"reason": f"整體平滑度極差 (均值 {mean_second_diff:.1f}mm)", "points": 20.0})
    elif mean_second_diff > 10.0:
        deduction_smooth = 10.0
        deductions.append({"reason": f"整體平滑度不佳 (均值 {mean_second_diff:.1f}mm)", "points": 10.0})
    elif mean_second_diff > 5.0:
        deduction_smooth = 5.0
        deductions.append({"reason": f"整體平滑度普通 (均值 {mean_second_diff:.1f}mm)", "points": 5.0})
        
    score -= deduction_smooth
    
    # 確保分數在 0-100 之間
    score = max(0.0, min(100.0, score))
    
    # 評級
    if score >= 90:
        level = "Excellent"
    elif score >= 80:
        level = "Good"
    elif score >= 60:
        level = "Fair"
    else:
        level = "Poor"
        
    return {
        "total_score": float(score),
        "level": level,
        "deductions": deductions
    }


def print_analysis_report_enhanced(
    velocity: dict,
    acceleration: dict,
    direction_change: dict,
    jump_anomalies: dict,
    smoothness: dict,
    frequency: dict,
    direction_continuity: dict,
    detailed_anomalies: dict,
    config: ValidationConfig
) -> None:
    """列印分析報告（增強版）"""
    
    if velocity:
        print("\n" + "=" * 100)
        print("【1. 速度變化率】")
        print("=" * 100)
        for joint, stats in velocity.items():
            print(f"{joint}: 平均速度 {stats['mean_velocity_mm_s']:.2f} mm/s, CV={stats['cv_percent']:.2f}%")
    
    if acceleration:
        print("\n" + "=" * 100)
        print("【2. 加速度異常】")
        print("=" * 100)
        print(f"異常點: {acceleration['outlier_count']} / {acceleration['sample_count']}")
        print(f"異常率: {acceleration['outlier_rate']:.2f}%")
    
    if direction_change:
        print("\n" + "=" * 100)
        print("【3. 方向突變】")
        print("=" * 100)
        print(f"突變次數: {direction_change['sudden_change_count']}")
        print(f"平均角度變化: {direction_change['mean_angle_change_deg']:.2f}°")
    
    if jump_anomalies:
        print("\n" + "=" * 100)
        print("【4. 異常跳躍】")
        print("=" * 100)
        print(f"大跳躍次數: {jump_anomalies['large_jump_count']}")
        print(f"最大位移: {jump_anomalies['max_displacement_mm']:.2f} mm")
    
    if smoothness:
        print("\n" + "=" * 100)
        print("【5. 平滑度】")
        print("=" * 100)
        print(f"二階差分均值: {smoothness['mean_second_diff_mm']:.2f} mm")
    
    if frequency:
        print("\n" + "=" * 100)
        print("【6. 頻域分析 (NEW)】")
        print("=" * 100)
        print(f"X 軸主頻: {frequency['x_axis']['dominant_frequency_hz']:.2f} Hz")
        print(f"Y 軸主頻: {frequency['y_axis']['dominant_frequency_hz']:.2f} Hz")
        print(f"Z 軸主頻: {frequency['z_axis']['dominant_frequency_hz']:.2f} Hz")
        print(f"高頻能量佔比 (X): {frequency['x_axis']['high_frequency_ratio']*100:.2f}%")
    
    if direction_continuity:
        print("\n" + "=" * 100)
        print("【7. 方向連續性增強檢測 (NEW)】")
        print("=" * 100)
        print(f"方向反轉次數: {direction_continuity['reversal_count']}")
        print(f"反轉率: {direction_continuity['reversal_rate']:.2f}%")
    
    # 詳細異常分析
    if detailed_anomalies and detailed_anomalies.get('total_anomaly_count', 0) > 0:
        print("\n" + "=" * 100)
        print("【8. 異常詳細分析】")
        print("=" * 100)
        
        total = detailed_anomalies['total_anomaly_count']
        accel = detailed_anomalies['acceleration_anomalies']
        dir_chg = detailed_anomalies['direction_change_anomalies']
        jumps = detailed_anomalies['jump_anomalies']
        reversals = detailed_anomalies['direction_reversals']
        
        print(f"總異常數: {total} 個")
        print(f"  • 加速度異常: {accel['total']} 個")
        print(f"  • 方向變化: {dir_chg['total']} 個")
        print(f"  • 異常跳躍: {jumps['total']} 個")
        print(f"  • 方向反轉: {reversals['count']} 次")
        print()
        
        # 加速度異常分析
        if accel['total'] > 0:
            print("▸ 加速度異常分級:")
            accel_threshold = accel.get('threshold_used', 1000.0)
            print(f"  • 嚴重 (>{accel_threshold*5:.0f} mm/s²): {len(accel['severe']):3d} 個 ({len(accel['severe'])/accel['total']*100:5.1f}%)")
            print(f"  • 中等 ({accel_threshold*3:.0f}-{accel_threshold*5:.0f} mm/s²): {len(accel['moderate']):3d} 個 ({len(accel['moderate'])/accel['total']*100:5.1f}%)")
            print(f"  • 輕微 (<{accel_threshold*3:.0f} mm/s²): {len(accel['mild']):3d} 個 ({len(accel['mild'])/accel['total']*100:5.1f}%)")
            print()
            
            # 嚴重加速度異常 - 全部顯示
            if accel['severe']:
                print("─" * 100)
                print(f"[!] 嚴重加速度異常 (>{accel_threshold*5:.0f} mm/s^2) - 全部 {len(accel['severe'])} 個:")
                print("─" * 100)
                for idx, item in enumerate(accel['severe'], 1):
                    print(f" {idx:3d}. Frame {item['frame']:3d}: 加速度 = {item['acceleration']:8.1f} mm/s²")
                print()
            
            # 中等加速度異常 - 顯示前 10 個
            if accel['moderate']:
                print("─" * 100)
                display_count = min(10, len(accel['moderate']))
                print(f"📊 中等加速度異常 ({accel_threshold*3:.0f}-{accel_threshold*5:.0f} mm/s²) - 顯示前 {display_count} 個，共 {len(accel['moderate'])} 個:")
                print("─" * 100)
                moderate_sorted = sorted(accel['moderate'], key=lambda x: x['acceleration'], reverse=True)
                for idx, item in enumerate(moderate_sorted[:display_count], 1):
                    print(f" {idx:3d}. Frame {item['frame']:3d}: 加速度 = {item['acceleration']:8.1f} mm/s²")
                if len(accel['moderate']) > display_count:
                    print(f"\n ⋮ 其餘 {len(accel['moderate']) - display_count} 個中等異常請參閱 JSON 輸出")
                print()
            
            # 輕微加速度異常 - 統計
            if accel['mild']:
                print("─" * 100)
                print(f"📋 輕微加速度異常 (<{accel_threshold*3:.0f} mm/s²) - 共 {len(accel['mild'])} 個，詳見 JSON")
                print("─" * 100)
                print()
        
        # 方向變化異常分析
        if dir_chg['total'] > 0:
            print("▸ 方向變化異常分級:")
            print(f"  • 嚴重 (>120°):  {len(dir_chg['severe']):3d} 個 ({len(dir_chg['severe'])/dir_chg['total']*100:5.1f}%)")
            print(f"  • 中等 (90-120°): {len(dir_chg['moderate']):3d} 個 ({len(dir_chg['moderate'])/dir_chg['total']*100:5.1f}%)")
            print(f"  • 輕微 (<90°):    {len(dir_chg['mild']):3d} 個 ({len(dir_chg['mild'])/dir_chg['total']*100:5.1f}%)")
            print()
            
            # 嚴重方向變化 - 全部顯示
            if dir_chg['severe']:
                print("─" * 100)
                print(f"[!] 嚴重方向變化 (>120°) - 全部 {len(dir_chg['severe'])} 個:")
                print("─" * 100)
                for idx, item in enumerate(dir_chg['severe'], 1):
                    print(f" {idx:3d}. Frame {item['frame']:3d}: 角度變化 = {item['angle_change']:6.1f}°")
                print()
            
            # 中等方向變化 - 顯示前 10 個
            if dir_chg['moderate']:
                print("─" * 100)
                display_count = min(10, len(dir_chg['moderate']))
                print(f"📊 中等方向變化 (90-120°) - 顯示前 {display_count} 個，共 {len(dir_chg['moderate'])} 個:")
                print("─" * 100)
                moderate_sorted = sorted(dir_chg['moderate'], key=lambda x: x['angle_change'], reverse=True)
                for idx, item in enumerate(moderate_sorted[:display_count], 1):
                    print(f" {idx:3d}. Frame {item['frame']:3d}: 角度變化 = {item['angle_change']:6.1f}°")
                if len(dir_chg['moderate']) > display_count:
                    print(f"\n ⋮ 其餘 {len(dir_chg['moderate']) - display_count} 個中等異常請參閱 JSON 輸出")
                print()
        
        # 跳躍異常分析
        if jumps['total'] > 0:
            print("▸ 異常跳躍分級:")
            jump_threshold = jumps.get('threshold_used', 100.0)
            print(f"  • 嚴重 (>{jump_threshold*3:.0f} mm):  {len(jumps['severe']):3d} 個 ({len(jumps['severe'])/jumps['total']*100:5.1f}%)")
            print(f"  • 中等 ({jump_threshold*2:.0f}-{jump_threshold*3:.0f} mm): {len(jumps['moderate']):3d} 個 ({len(jumps['moderate'])/jumps['total']*100:5.1f}%)")
            print(f"  • 輕微 ({jump_threshold:.0f}-{jump_threshold*2:.0f} mm): {len(jumps['mild']):3d} 個 ({len(jumps['mild'])/jumps['total']*100:5.1f}%)")
            print()
            
            # 嚴重跳躍 - 全部顯示
            if jumps['severe']:
                print("─" * 100)
                print(f"[!] 嚴重異常跳躍 (>{jump_threshold*3:.0f} mm) - 全部 {len(jumps['severe'])} 個:")
                print("─" * 100)
                for idx, item in enumerate(jumps['severe'], 1):
                    print(f" {idx:3d}. Frame {item['frame']:3d}: 位移 = {item['displacement']:7.1f} mm")
                print()
        
        # 連續異常區段
        segments = detailed_anomalies.get('continuous_segments', [])
        if segments:
            print("▸ 連續異常區段（加速度，≥3 幀）:")
            for seg in segments[:5]:
                print(f"  • Frame {seg['start_frame']:3d}-{seg['end_frame']:3d} ({seg['duration']:2d} 幀): "
                      f"平均 {seg['mean_acceleration']:7.1f} mm/s², 最大 {seg['max_acceleration']:7.1f} mm/s²")
            print()
        
        print("💾 完整異常列表已儲存至 JSON:")
        print("   [V] detailed_anomalies.acceleration_anomalies - 所有加速度異常")
        print("   [V] detailed_anomalies.direction_change_anomalies - 所有方向變化")
        print("   [V] detailed_anomalies.jump_anomalies - 所有異常跳躍")
        print("   [V] detailed_anomalies.continuous_segments - 連續異常區段")


def validate_time_smoothness_analysis(
    json_3d_path: str,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    時間平滑度驗證分析（主函數）
    
    參數:
        json_3d_path: 3D 軌跡 JSON 檔案路徑
        output_json_path: 輸出結果 JSON 路徑（可選）
        config_path: 配置檔案路徑（可選）
    
    返回:
        dict: 完整分析結果
    """
    # 載入配置
    config = load_config(config_path)
    
    # 載入數據
    print(f"\n載入數據: {json_3d_path}")
    data = load_json_file(json_3d_path)
    print(f"總幀數: {len(data)}")
    
    # 執行各項分析
    print("\n執行速度分析...")
    velocity = analyze_velocity_changes(data, config)
    
    print("執行加速度分析...")
    acceleration = analyze_acceleration_anomalies(data, config)
    
    print("執行方向突變分析...")
    direction_change = analyze_direction_changes(data, config)
    
    print("執行跳躍異常分析...")
    jump_anomalies = analyze_jump_anomalies(data, config)
    
    print("執行平滑度分析...")
    smoothness = analyze_smoothness(data, config)
    
    print("執行頻域分析...")
    frequency = analyze_frequency_domain(data, config)
    
    print("執行方向連續性增強檢測...")
    direction_continuity = analyze_direction_continuity_enhanced(data, config)
    
    print("執行詳細異常分析...")
    detailed_anomalies = analyze_detailed_anomalies(
        acceleration, direction_change, jump_anomalies, direction_continuity, config
    )
    
    # 計算評分
    score_result = calculate_score(detailed_anomalies, smoothness, config)
    
    # 列印報告
    print_analysis_report_enhanced(
        velocity, acceleration, direction_change,
        jump_anomalies, smoothness, frequency, direction_continuity,
        detailed_anomalies, config
    )
    
    print("\n" + "=" * 100)
    print(f"【評分結果】 {score_result['total_score']:.1f} / 100 ({score_result['level']})")
    print("=" * 100)
    for d in score_result['deductions']:
        print(f"  - {d['reason']}: -{d['points']:.1f}")
    
    # 整合結果
    results = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "source_file": str(json_3d_path),
            "total_frames": int(len(data)),
            "analysis_type": "Time Smoothness Analysis"
        },
        "score": score_result,
        "overall_summary": {
            "total_acceleration_outliers": acceleration.get('outlier_count', 0) if acceleration else 0,
            "total_large_jumps": jump_anomalies.get('large_jump_count', 0) if jump_anomalies else 0,
            "total_direction_reversals": direction_continuity.get('reversal_count', 0) if direction_continuity else 0
        },
        "velocity_analysis": velocity,
        "acceleration_anomalies": acceleration,
        "direction_changes": direction_change,
        "jump_anomalies": jump_anomalies,
        "smoothness_analysis": smoothness,
        "frequency_domain_analysis": frequency,
        "direction_continuity_enhanced": direction_continuity,
        "detailed_anomalies": {
            "summary": {
                "total_anomaly_count": detailed_anomalies.get('total_anomaly_count', 0),
                "acceleration_count": detailed_anomalies.get('acceleration_anomalies', {}).get('total', 0),
                "direction_change_count": detailed_anomalies.get('direction_change_anomalies', {}).get('total', 0),
                "jump_count": detailed_anomalies.get('jump_anomalies', {}).get('total', 0),
                "reversal_count": detailed_anomalies.get('direction_reversals', {}).get('count', 0)
            },
            "acceleration_anomalies": {
                "severe": detailed_anomalies.get('acceleration_anomalies', {}).get('severe', []),
                "moderate": detailed_anomalies.get('acceleration_anomalies', {}).get('moderate', []),
                "mild": detailed_anomalies.get('acceleration_anomalies', {}).get('mild', [])
            },
            "direction_change_anomalies": {
                "severe": detailed_anomalies.get('direction_change_anomalies', {}).get('severe', []),
                "moderate": detailed_anomalies.get('direction_change_anomalies', {}).get('moderate', []),
                "mild": detailed_anomalies.get('direction_change_anomalies', {}).get('mild', [])
            },
            "jump_anomalies": {
                "severe": detailed_anomalies.get('jump_anomalies', {}).get('severe', []),
                "moderate": detailed_anomalies.get('jump_anomalies', {}).get('moderate', []),
                "mild": detailed_anomalies.get('jump_anomalies', {}).get('mild', [])
            },
            "continuous_segments": detailed_anomalies.get('continuous_segments', [])
        } if detailed_anomalies.get('total_anomaly_count', 0) > 0 else {}
    }
    
    # 保存結果
    if output_json_path is None:
        output_json_path = generate_output_path(json_3d_path, '_step6_time_smoothness_results')
    
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
        json_3d_path = "0306_3__trajectory/trajectory__2/0306_3__2(3D_trajectory_smoothed).json"
        config_path = None
        output_json_path = None
        print("提示: 可使用命令列參數:")
        print("  python step6_time_smoothness_v2.py <json_path> [--config <config>] [--output <output>]")
    
    try:
        results = validate_time_smoothness_analysis(
            json_3d_path,
            output_json_path,
            config_path
        )
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)