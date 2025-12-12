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
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "neck", "left_hand", "right_hand", "tennis_ball"
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
    diffs = np.diff(centers, axis=0)
    displacements = np.linalg.norm(diffs, axis=1)
    
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
            "displacement": displacements.tolist(),
            "displacement_x": diffs[:, 0].tolist(),
            "displacement_y": diffs[:, 1].tolist(),
            "displacement_z": diffs[:, 2].tolist()
        }
    }


def analyze_ball_racket_contact(data: list, config: ValidationConfig) -> dict:
    """
    分析球拍接觸檢測（碰撞檢測 - 增強版 V2）
    檢查球與手腕或球拍子點（paddle）的距離
    
    返回:
        dict: 球拍接觸分析結果
    """
    contact_frames = []
    min_distances = []
    all_distances = [] 
    all_frames = []
    contact_threshold = getattr(config, 'racket_contact_threshold', 200.0)

    for frame_idx, frame in enumerate(data):
        ball = get_keypoint_safely(frame, "tennis_ball")
        if ball is None:
            continue

        # 檢查點：手腕和球拍
        contact_keypoints = ["left_wrist", "right_wrist"]
        current_distances = []

        # 處理手腕
        for kp_name in contact_keypoints:
            keypoint = get_keypoint_safely(frame, kp_name)
            if keypoint is not None:
                dist = calculate_distance(ball, keypoint)
                if dist is not None:
                    current_distances.append(dist)
        
        # 專門處理 'paddle' 物件
        paddle_obj = frame.get("paddle")
        if isinstance(paddle_obj, dict):
            for sub_point_name, sub_point_data in paddle_obj.items():
                if isinstance(sub_point_data, dict) and all(k in sub_point_data for k in ['x', 'y', 'z']):
                    paddle_point = np.array([sub_point_data['x'], sub_point_data['y'], sub_point_data['z']])
                    dist = calculate_distance(ball, paddle_point)
                    if dist is not None:
                        current_distances.append(dist)

        if current_distances:
            min_dist = min(current_distances)
            min_distances.append(min_dist)
            all_distances.append(min_dist)
            all_frames.append(frame_idx)
            
            if min_dist < contact_threshold:
                contact_frames.append({
                    "frame": frame_idx,
                    "distance_mm": float(min_dist)
                })
    
    if not min_distances:
        return {}
    
    min_overall_distance = float(np.min(min_distances)) if min_distances else None

    return {
        "total_frames_analyzed": len(min_distances),
        "contact_count": len(contact_frames),
        "min_distance_mm": min_overall_distance,
        "mean_distance_mm": float(np.mean(min_distances)) if min_distances else None,
        "contact_frames": contact_frames[:10],
        "series": {
            "frames": all_frames,
            "min_distance": all_distances
        }
    }


def analyze_gravity_compliance(data: list, config: ValidationConfig) -> dict:
    """
    分析重力加速度合理性（增強版，自動檢測重力軸）
    
    返回:
        dict: 重力分析結果
    """
    ball_positions = []
    frame_indices = []
    
    for frame_idx, frame in enumerate(data):
        ball = get_keypoint_safely(frame, "tennis_ball")
        if ball is not None:
            # 簡化邏輯，只要有球就納入計算，噪聲會在平均中被平滑
            ball_positions.append(ball)
            frame_indices.append(frame_idx)
    
    if len(ball_positions) < 5:
        return {"assessment": "數據不足"}
    
    positions = np.array(ball_positions)
    indices = np.array(frame_indices)
    
    vel, vel_indices = calculate_velocity(positions, config.fps, indices)
    acc, acc_indices = calculate_acceleration(vel, config.fps, vel_indices)
    
    if len(acc) == 0:
        return {"assessment": "無法計算加速度"}
    
    expected_gravity = abs(config.gravity_acceleration)
    axes = ['X', 'Y', 'Z']
    best_axis_info = {
        "deviation": float('inf'),
        "axis_index": -1,
        "mean_acc": 0,
        "acc_series": []
    }

    # 遍歷三個軸，找到最接近重力的那一個
    for i in range(3):
        acc_axis = acc[:, i]
        mean_acc = np.mean(acc_axis)
        deviation = abs(abs(mean_acc) - expected_gravity)
        
        if deviation < best_axis_info["deviation"]:
            best_axis_info = {
                "deviation": deviation,
                "axis_index": i,
                "mean_acc": float(mean_acc),
                "acc_series": acc_axis.flatten().tolist()
            }
            
    deviation_ratio = best_axis_info["deviation"] / expected_gravity
    
    if deviation_ratio < config.gravity_tolerance:
        assessment = f"[OK] 重力加速度合理 (檢測到垂直軸: {axes[best_axis_info['axis_index']]})"
    else:
        assessment = f"[!] 重力加速度偏差較大 (檢測到垂直軸: {axes[best_axis_info['axis_index']]})"
    
    return {
        "sample_count": len(acc),
        "detected_gravity_axis": axes[best_axis_info["axis_index"]],
        "mean_acceleration_mm_s2": best_axis_info["mean_acc"],
        "expected_gravity_mm_s2": float(config.gravity_acceleration),
        "deviation_ratio": float(deviation_ratio),
        "assessment": assessment,
        "series": {
            "frames": acc_indices.tolist(),
            "acceleration": best_axis_info["acc_series"]
        }
    }


def analyze_motion_continuity(data: list, config: ValidationConfig) -> dict:
    """
    分析運動連續性，檢測幀之間的跳躍
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 連續性分析結果
    """
    # 從 utils 或 config 引入所有關節點名稱
    try:
        from .utils import ALL_KEYPOINTS
    except ImportError:
        from utils import ALL_KEYPOINTS

    results = {}
    total_jumps = 0
    
    for kp in ALL_KEYPOINTS:
        positions = []
        frame_indices = []
        for i, frame in enumerate(data):
            point = get_keypoint_safely(frame, kp)
            if point is not None:
                positions.append(point)
                frame_indices.append(i)
        
        if len(positions) < 2:
            continue
            
        pos_arr = np.array(positions)
        idx_arr = np.array(frame_indices)
        
        # 僅計算連續幀之間的位移
        displacements = []
        jump_frames = []
        
        for i in range(len(pos_arr) - 1):
            # 檢查幀號是否連續
            if idx_arr[i+1] == idx_arr[i] + 1:
                dist = np.linalg.norm(pos_arr[i+1] - pos_arr[i])
                displacements.append(dist)
                if dist > config.max_inter_frame_distance:
                    jump_frames.append(idx_arr[i+1])

        if not displacements:
            continue
        
        jump_count = len(jump_frames)
        total_jumps += jump_count
        
        results[kp] = {
            "max_displacement_mm": float(np.max(displacements)) if displacements else 0.0,
            "mean_displacement_mm": float(np.mean(displacements)) if displacements else 0.0,
            "jump_count": jump_count,
            "jump_rate": float(jump_count / len(displacements) * 100) if displacements else 0.0,
            "jump_frames": jump_frames[:20], # 最多顯示 20 個
            "series": {
                "frames": idx_arr[1:].tolist(),
                "displacements": displacements
            }
        }
        
    return {"total_jumps": total_jumps, "details": results}


def print_analysis_report(
    kinematics: dict,
    joint_angles: dict,
    torso_stability: dict,
    ball_contact: dict,
    gravity_analysis: dict,
    motion_continuity: dict,
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
        if ball_contact.get('min_distance_mm') is not None:
            print(f"偵測到的最小距離: {ball_contact['min_distance_mm']:.2f} mm")
    
    if gravity_analysis and "mean_acceleration_mm_s2" in gravity_analysis:
        print("\n" + "=" * 100)
        print("【5. 重力加速度檢查】")
        print("=" * 100)
        print(f"檢測到的垂直軸: {gravity_analysis['detected_gravity_axis']}")
        print(f"平均垂直軸加速度: {gravity_analysis['mean_acceleration_mm_s2']:.1f} mm/s^2")
        print(f"預期重力加速度: {abs(gravity_analysis['expected_gravity_mm_s2']):.1f} mm/s^2")
        print(f"偏差: {gravity_analysis['deviation_ratio']*100:.1f}%")
        print(f"評估: {gravity_analysis['assessment']}")

    print("\n" + "=" * 100)
    print("【6. 運動連續性檢查 (跳躍檢測)】")
    print("=" * 100)
    if motion_continuity and motion_continuity['total_jumps'] > 0:
        print(f"偵測到總跳躍次數: {motion_continuity['total_jumps']}")
        for kp, stats in motion_continuity['details'].items():
            if stats['jump_count'] > 0:
                zh_name = get_keypoint_name_zh(kp)
                print(f"  - {zh_name}: {stats['jump_count']} 次跳躍 (最大位移: {stats['max_displacement_mm']:.1f} mm)")
    else:
        print("未偵測到連續性跳躍，軌跡平滑。")


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
    
    print("執行運動連續性檢查...")
    motion_continuity = analyze_motion_continuity(data, config)

    # 列印報告
    print_analysis_report(
        kinematics, joint_angles, torso_stability,
        ball_contact, gravity_analysis, motion_continuity, config
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
            "total_unreasonable_speed": sum(stats.get('unreasonable_speed_count', 0) for stats in kinematics.values()),
            "total_abnormal_angles": sum(stats.get('abnormal_count', 0) for stats in joint_angles.values()),
            "total_continuity_jumps": motion_continuity.get('total_jumps', 0),
            "gravity_compliant": gravity_analysis.get('assessment', '').startswith('[OK]') if gravity_analysis else None,
            "contact_detected": ball_contact.get('contact_count', 0) > 0 if ball_contact else None,
        },
        "motion_kinematics": kinematics,
        "joint_angles": joint_angles,
        "torso_stability": torso_stability,
        "ball_racket_contact": ball_contact,
        "gravity_compliance": gravity_analysis,
        "motion_continuity": motion_continuity
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
        json_3d_path = "data/trajectory__new/tsung__19_45(3D_trajectory_smoothed).json"
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
