"""
Step 4: 物理運動邏輯驗證分析
驗證 3D 重建結果的物理合理性（速度、加速度、角度、能量等）

功能：
  1. 速度/加速度/Jerk 分析
  2. 關節角度合理性檢查
  3. 軀幹穩定性分析
  4. 碰撞檢測（球拍接觸）
  5. 重力加速度檢查
  6. 運動連續性檢查
"""

import json
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入共用模組
try:
    from .utils import (
        get_keypoint_safely,
        calculate_distance,
        calculate_angle,
        load_json_file,
        save_json_results,
        generate_output_path,
        get_keypoint_name_zh,
    )
except ImportError:
    from utils import (
        get_keypoint_safely,
        calculate_distance,
        calculate_angle,
        load_json_file,
        save_json_results,
        generate_output_path,
        get_keypoint_name_zh,
    )

from config import load_config, ValidationConfig


# ========================================================
# 運動學計算函數
# ========================================================

def calculate_velocity(
    positions: np.ndarray,
    fps: int,
    frame_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """計算速度（一階導數），並保留對應幀索引"""
    if len(positions) < 2:
        return np.array([]), np.array([])

    if frame_indices is not None and len(frame_indices) == len(positions):
        time_deltas = np.diff(frame_indices) / float(fps)
    else:
        time_deltas = np.full(len(positions) - 1, 1.0 / fps)

    time_deltas[time_deltas == 0] = 1e-6  # 避免除以零
    diffs = np.diff(positions, axis=0)
    velocities = diffs / time_deltas[:, None]
    velocity_indices = frame_indices[1:] if frame_indices is not None else np.arange(1, len(positions))
    return velocities, velocity_indices


def calculate_acceleration(
    velocities: np.ndarray,
    fps: int,
    velocity_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """計算加速度（二階導數），並保留對應幀索引"""
    if len(velocities) < 2:
        return np.array([]), np.array([])

    if velocity_indices is not None and len(velocity_indices) == len(velocities):
        time_deltas = np.diff(velocity_indices) / float(fps)
    else:
        time_deltas = np.full(len(velocities) - 1, 1.0 / fps)

    time_deltas[time_deltas == 0] = 1e-6
    diffs = np.diff(velocities, axis=0)
    accelerations = diffs / time_deltas[:, None]
    acceleration_indices = velocity_indices[1:] if velocity_indices is not None else np.arange(1, len(velocities))
    return accelerations, acceleration_indices


def calculate_jerk(
    accelerations: np.ndarray,
    fps: int,
    acceleration_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """計算 Jerk（三階導數），並保留對應幀索引"""
    if len(accelerations) < 2:
        return np.array([]), np.array([])

    if acceleration_indices is not None and len(acceleration_indices) == len(accelerations):
        time_deltas = np.diff(acceleration_indices) / float(fps)
    else:
        time_deltas = np.full(len(accelerations) - 1, 1.0 / fps)

    time_deltas[time_deltas == 0] = 1e-6
    diffs = np.diff(accelerations, axis=0)
    jerks = diffs / time_deltas[:, None]
    jerk_indices = acceleration_indices[1:] if acceleration_indices is not None else np.arange(1, len(accelerations))
    return jerks, jerk_indices


# ========================================================
# 核心分析函數
# ========================================================

def analyze_motion_kinematics(data: list, config: ValidationConfig) -> dict:
    """
    分析運動學（速度/加速度/Jerk）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 運動學分析結果
    """
    keypoints_to_analyze = [
        "left_wrist", "right_wrist",
        "left_ankle", "right_ankle",
        "tennis_ball"
    ]
    
    results = {}
    
    for kp in keypoints_to_analyze:
        positions = []
        for frame in data:
            point = get_keypoint_safely(frame, kp)
            if point is None:
                positions.append([np.nan, np.nan, np.nan])
            else:
                positions.append(point)
        
        positions = np.array(positions)
        valid_mask = ~np.isnan(positions[:, 0])
        
        if np.sum(valid_mask) < 3:
            continue
        
        pos = positions[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # 速度
        vel, vel_indices = calculate_velocity(pos, config.fps, valid_indices)
        if len(vel) == 0:
            continue
        speed = np.linalg.norm(vel, axis=1)
        
        # 加速度
        acc, acc_indices = calculate_acceleration(vel, config.fps, vel_indices)
        acc_mag = np.linalg.norm(acc, axis=1) if len(acc) > 0 else np.array([])
        
        # Jerk
        jerk_mag = np.array([])
        if len(acc) > 1:
            jerk, _ = calculate_jerk(acc, config.fps, acc_indices)
            jerk_mag = np.linalg.norm(jerk, axis=1) if len(jerk) > 0 else np.array([])
        
        # 檢查速度合理性
        if kp in ["left_wrist", "right_wrist"]:
            max_reasonable_speed = config.max_wrist_speed_ms * 1000  # 轉為 mm/s
        elif kp == "tennis_ball":
            max_reasonable_speed = config.max_ball_speed_ms * 1000
        else:
            max_reasonable_speed = config.max_wrist_speed_ms * 1000
        
        unreasonable_speed_count = int(np.sum(speed > max_reasonable_speed))
        
        results[kp] = {
            "max_speed_m_s": float(np.max(speed) / 1000),
            "mean_speed_m_s": float(np.mean(speed) / 1000),
            "max_acc_mm_s2": float(np.max(acc_mag)) if len(acc_mag) > 0 else None,
            "max_jerk_mm_s3": float(np.max(jerk_mag)) if len(jerk_mag) > 0 else None,
            "unreasonable_speed_count": unreasonable_speed_count,
            "unreasonable_speed_rate": float(unreasonable_speed_count / len(speed) * 100) if len(speed) > 0 else 0.0,
            "series": {
                "frames": valid_indices.tolist(),
                "speed_mm_s": speed.tolist(),
                "acc_mm_s2": acc_mag.tolist() if len(acc_mag) > 0 else []
            }
        }
    
    return results


def analyze_joint_angles(data: list, config: ValidationConfig) -> dict:
    """
    分析關節角度合理性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 關節角度分析結果
    """
    joint_defs = {
        "左肘": ("left_shoulder", "left_elbow", "left_wrist"),
        "右肘": ("right_shoulder", "right_elbow", "right_wrist"),
        "左膝": ("left_hip", "left_knee", "left_ankle"),
        "右膝": ("right_hip", "right_knee", "right_ankle"),
        "左髖": ("left_shoulder", "left_hip", "left_knee"),
        "右髖": ("right_shoulder", "right_hip", "right_knee"),
    }
    
    results = {}
    
    for joint_name, (j1, j2, j3) in joint_defs.items():
        angles = []
        frame_indices = []
        for i, frame in enumerate(data):
            p1 = get_keypoint_safely(frame, j1)
            p2 = get_keypoint_safely(frame, j2)
            p3 = get_keypoint_safely(frame, j3)
            
            if all(p is not None for p in [p1, p2, p3]):
                angle = calculate_angle(p1, p2, p3)
                if angle is not None:
                    angles.append(angle)
                    frame_indices.append(i)
        
        if not angles:
            continue
        
        arr = np.array(angles, dtype=float)
        abnormal = int(np.sum((arr < config.joint_angle_min) | (arr > config.joint_angle_max)))
        
        results[joint_name] = {
            "min_angle": float(arr.min()),
            "max_angle": float(arr.max()),
            "mean_angle": float(arr.mean()),
            "std_angle": float(arr.std()),
            "abnormal_count": abnormal,
            "abnormal_rate": float(abnormal / len(arr) * 100),
            "series": {
                "frames": frame_indices,
                "angles": angles
            }
        }
    
    return results


def analyze_torso_stability(data: list, config: ValidationConfig) -> dict:
    """
    分析軀幹穩定性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 軀幹穩定性分析結果
    """
    torso_centers = []
    frame_indices = []
    
    for i, frame in enumerate(data):
        ls = get_keypoint_safely(frame, "left_shoulder")
        rs = get_keypoint_safely(frame, "right_shoulder")
        lh = get_keypoint_safely(frame, "left_hip")
        rh = get_keypoint_safely(frame, "right_hip")
        
        if all(p is not None for p in [ls, rs, lh, rh]):
            center = (ls + rs + lh + rh) / 4
            torso_centers.append(center)
            frame_indices.append(i)
    
    if len(torso_centers) < 2:
        return {}
    
    centers = np.array(torso_centers)
    displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    
    mean_displacement = float(np.mean(displacements))
    max_displacement = float(np.max(displacements))
    
    if mean_displacement < config.torso_stability_threshold:
        assessment = "[OK] 軀幹穩定"
    else:
        assessment = "[!] 軀幹晃動較大"
    
    return {
        "sample_count": len(displacements),
        "mean_displacement_mm": mean_displacement,
        "max_displacement_mm": max_displacement,
        "std_displacement_mm": float(np.std(displacements)),
        "assessment": assessment,
        "series": {
            "frames": frame_indices[1:], # diff reduces length by 1
            "displacement": displacements.tolist()
        }
    }


def analyze_ball_racket_contact(data: list, config: ValidationConfig) -> dict:
    """
    分析球拍接觸檢測（碰撞檢測 - 新增）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 球拍接觸分析結果
    """
    contact_frames = []
    min_distances = []
    all_distances = [] # Store all for plotting
    all_frames = []
    contact_threshold = getattr(config, 'racket_contact_threshold', 200.0)  # mm
    
    for frame_idx, frame in enumerate(data):
        ball = get_keypoint_safely(frame, "tennis_ball")
        left_wrist = get_keypoint_safely(frame, "left_wrist")
        right_wrist = get_keypoint_safely(frame, "right_wrist")
        
        if ball is None:
            continue
        
        # 檢查球與手腕的距離
        distances = []
        if left_wrist is not None:
            dist = calculate_distance(ball, left_wrist)
            if dist is not None:
                distances.append(dist)
        
        if right_wrist is not None:
            dist = calculate_distance(ball, right_wrist)
            if dist is not None:
                distances.append(dist)
        
        if distances:
            min_dist = min(distances)
            min_distances.append(min_dist)
            all_distances.append(min_dist)
            all_frames.append(frame_idx)
            
            # 使用配置的接觸距離閾值
            if min_dist < contact_threshold:
                contact_frames.append({
                    "frame": frame_idx,
                    "distance_mm": float(min_dist)
                })
    
    if not min_distances:
        return {}
    
    return {
        "total_frames_analyzed": len(min_distances),
        "contact_count": len(contact_frames),
        "min_distance_mm": float(np.min(min_distances)),
        "mean_distance_mm": float(np.mean(min_distances)),
        "contact_frames": contact_frames[:10],  # 只返回前 10 個接觸幀
        "series": {
            "frames": all_frames,
            "min_distance": all_distances
        }
    }


def analyze_gravity_compliance(data: list, config: ValidationConfig) -> dict:
    """
    分析重力加速度合理性（新增）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 重力分析結果
    """
    ball_positions = []
    frame_indices = []
    
    for frame_idx, frame in enumerate(data):
        ball = get_keypoint_safely(frame, "tennis_ball")
        left_wrist = get_keypoint_safely(frame, "left_wrist")
        right_wrist = get_keypoint_safely(frame, "right_wrist")
        
        if ball is not None:
            # 檢查是否在飛行階段（遠離手腕）
            is_flying = True
            if left_wrist is not None:
                dist = calculate_distance(ball, left_wrist)
                if dist is not None and dist < 300:  # 300mm 閾值
                    is_flying = False
            
            if right_wrist is not None:
                dist = calculate_distance(ball, right_wrist)
                if dist is not None and dist < 300:
                    is_flying = False
            
            if is_flying:
                ball_positions.append(ball)
                frame_indices.append(frame_idx)
    
    if len(ball_positions) < 5:
        return {}
    
    positions = np.array(ball_positions)
    indices = np.array(frame_indices)
    
    # 計算垂直軸加速度（Y 軸通常是垂直方向，正向向下）
    # 注意：根據座標系統，Y 軸正向可能向下，因此重力加速度為正值
    vertical_axis = getattr(config, 'gravity_axis', 1)  # 預設 Y 軸 (index 1)
    vel_y, vel_indices = calculate_velocity(positions[:, [vertical_axis]], config.fps, indices)
    acc_y, _ = calculate_acceleration(vel_y, config.fps, vel_indices)
    
    if len(acc_y) == 0:
        return {}
    
    mean_acc_y = float(np.mean(acc_y))
    
    # 檢查是否接近重力加速度（考慮座標系統方向）
    expected_gravity = config.gravity_acceleration  # -9810 mm/s² 或 +9810 mm/s²
    deviation = abs(abs(mean_acc_y) - abs(expected_gravity)) / abs(expected_gravity)
    
    if deviation < config.gravity_tolerance:
        assessment = "[OK] 重力加速度合理"
    else:
        assessment = "[!] 重力加速度偏差較大"
    
    return {
        "sample_count": len(acc_y),
        "mean_y_acceleration_mm_s2": mean_acc_y,
        "expected_gravity_mm_s2": float(expected_gravity),
        "deviation_ratio": float(deviation),
        "assessment": assessment,
        "series": {
            "frames": vel_indices[1:].tolist(), # acc indices
            "acc_y": acc_y.flatten().tolist()
        }
    }


def print_analysis_report(
    kinematics: dict,
    joint_angles: dict,
    torso_stability: dict,
    ball_contact: dict,
    gravity_analysis: dict,
    config: ValidationConfig
) -> None:
    """列印分析報告"""
    
    print("\n" + "=" * 100)
    print("【1. 速度/加速度/Jerk 分析】")
    print("=" * 100)
    
    for kp, stats in kinematics.items():
        zh_name = get_keypoint_name_zh(kp)
        print(f"\n{zh_name}:")
        print(f"  最大速度: {stats['max_speed_m_s']:.2f} m/s")
        print(f"  平均速度: {stats['mean_speed_m_s']:.2f} m/s")
        if stats['max_acc_mm_s2'] is not None:
            print(f"  最大加速度: {stats['max_acc_mm_s2']:.1f} mm/s^2")
        if stats['unreasonable_speed_count'] > 0:
            print(f"  [WARNING] 異常速度次數: {stats['unreasonable_speed_count']} ({stats['unreasonable_speed_rate']:.1f}%)")
    
    print("\n" + "=" * 100)
    print("【2. 關節角度檢查】")
    print("=" * 100)
    
    for joint_name, stats in joint_angles.items():
        print(f"{joint_name}: 範圍 {stats['min_angle']:.1f}° ~ {stats['max_angle']:.1f}°, "
              f"平均 {stats['mean_angle']:.1f}°")
        if stats['abnormal_count'] > 0:
            print(f"  [WARNING] 異常角度: {stats['abnormal_count']} ({stats['abnormal_rate']:.1f}%)")
    
    if torso_stability:
        print("\n" + "=" * 100)
        print("【3. 軀幹穩定性】")
        print("=" * 100)
        print(f"平均位移: {torso_stability['mean_displacement_mm']:.2f} mm")
        print(f"最大位移: {torso_stability['max_displacement_mm']:.2f} mm")
        print(f"評估: {torso_stability['assessment']}")
    
    if ball_contact:
        print("\n" + "=" * 100)
        print("【4. 球拍接觸檢測】")
        print("=" * 100)
        print(f"檢測到接觸次數: {ball_contact['contact_count']}")
        print(f"最小距離: {ball_contact['min_distance_mm']:.2f} mm")
    
    if gravity_analysis:
        print("\n" + "=" * 100)
        print("【5. 重力加速度檢查】")
        print("=" * 100)
        print(f"平均垂直軸加速度: {gravity_analysis['mean_y_acceleration_mm_s2']:.1f} mm/s^2")
        print(f"預期重力加速度: {abs(gravity_analysis['expected_gravity_mm_s2']):.1f} mm/s^2")
        print(f"偏差: {gravity_analysis['deviation_ratio']*100:.1f}%")
        print(f"評估: {gravity_analysis['assessment']}")


def validate_physical_motion_analysis(
    json_3d_path: str,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    物理運動邏輯驗證分析（主函數）
    
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
    print("\n執行運動學分析...")
    kinematics = analyze_motion_kinematics(data, config)
    
    print("執行關節角度分析...")
    joint_angles = analyze_joint_angles(data, config)
    
    print("執行軀幹穩定性分析...")
    torso_stability = analyze_torso_stability(data, config)
    
    print("執行球拍接觸檢測...")
    ball_contact = analyze_ball_racket_contact(data, config)
    
    print("執行重力檢查...")
    gravity_analysis = analyze_gravity_compliance(data, config)
    
    # 列印報告
    print_analysis_report(
        kinematics, joint_angles, torso_stability,
        ball_contact, gravity_analysis, config
    )
    
    # 整合結果
    results = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "source_file": str(json_3d_path),
            "total_frames": int(len(data)),
            "analysis_type": "Physical Motion Logic Analysis"
        },
        "overall_summary": {
            "total_joints_analyzed": len(kinematics),
            "total_abnormal_angles": sum(stats.get('abnormal_count', 0) for stats in joint_angles.values()),
            "gravity_compliant": gravity_analysis.get('assessment', '').startswith('[OK]') if gravity_analysis else None
        },
        "motion_kinematics": kinematics,
        "joint_angles": joint_angles,
        "torso_stability": torso_stability,
        "ball_racket_contact": ball_contact,
        "gravity_compliance": gravity_analysis
    }
    
    # 保存結果
    if output_json_path is None:
        # 建立 results 資料夾
        results_dir = os.path.join(os.path.dirname(json_3d_path), 'Verification Result')
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d_path))[0]
        output_json_path = os.path.join(results_dir, f"{base_name}_step4_physical_motion_results.json")
    
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
        print("  python step4_physical_motion_v2.py <json_path> [--config <config>] [--output <output>]")
    
    try:
        results = validate_physical_motion_analysis(
            json_3d_path,
            output_json_path,
            config_path
        )
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
