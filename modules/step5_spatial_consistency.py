"""
Step 5: 空間一致性驗證分析
驗證 3D 重建結果的空間拓撲結構和一致性

功能：
  1. 地面平面一致性
  2. 身體左右對稱性
  3. 相對位置一致性
  4. 重心穩定性
  5. 距離恆定性（剛體組）
  6. 地面平面一致性檢查
  7. 拓撲結構驗證
  8. 穿透檢測強化
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入共用模組
try:
    from .utils import (
        get_keypoint_safely,
        calculate_distance,
        load_json_file,
        save_json_results,
        calculate_cv,
        detect_outliers_zscore,
        generate_output_path,
        KEYPOINT_NAMES_EN,
    )
except ImportError:
    from utils import (
        get_keypoint_safely,
        calculate_distance,
        load_json_file,
        save_json_results,
        calculate_cv,
        detect_outliers_zscore,
        generate_output_path,
        KEYPOINT_NAMES_EN,
    )

from config import load_config, ValidationConfig

AXIS_TO_INDEX: Dict[str, int] = {"x": 0, "y": 1, "z": 2}


def safe_percentage(numerator: float, denominator: float, epsilon: float) -> float:
    """避免除零錯誤的百分比計算"""
    if denominator <= epsilon:
        return 0.0
    return float(numerator / denominator * 100.0)


# ========================================================
# 核心分析函數
# ========================================================

def detect_vertical_orientation(data: list) -> str:
    """
    自動檢測垂直軸方向 (Y-Up 或 Y-Down)
    
    返回:
        'y_up': Y 軸向上 (頭部 Y > 腳部 Y)
        'y_down': Y 軸向下 (頭部 Y < 腳部 Y)
    """
    nose_ys = []
    ankle_ys = []
    
    for frame in data:
        nose = get_keypoint_safely(frame, "nose")
        la = get_keypoint_safely(frame, "left_ankle")
        
        if nose is not None: nose_ys.append(nose[1])
        if la is not None: ankle_ys.append(la[1])
        
    if not nose_ys or not ankle_ys:
        return 'y_down' # 預設
        
    mean_nose = np.mean(nose_ys)
    mean_ankle = np.mean(ankle_ys)
    
    return 'y_up' if mean_nose > mean_ankle else 'y_down'


def analyze_ground_plane_consistency(data: list, config: ValidationConfig) -> dict:
    """
    分析地面平面一致性
    """
    orientation = detect_vertical_orientation(data)
    print(f"檢測到垂直座標方向: {orientation}")
    
    ankle_diffs = []
    frame_indices = []
    all_ankle_ys = []
    
    left_ankle_ys = []
    right_ankle_ys = []
    
    # 1. 收集數據與計算高度差
    for i, frame in enumerate(data):
        left_ankle = get_keypoint_safely(frame, "left_ankle")
        right_ankle = get_keypoint_safely(frame, "right_ankle")
        
        if left_ankle is not None: 
            all_ankle_ys.append(left_ankle[1])
            left_ankle_ys.append(left_ankle[1])
        else:
            left_ankle_ys.append(None)
            
        if right_ankle is not None: 
            all_ankle_ys.append(right_ankle[1])
            right_ankle_ys.append(right_ankle[1])
        else:
            right_ankle_ys.append(None)
        
        if left_ankle is not None and right_ankle is not None:
            diff = abs(left_ankle[1] - right_ankle[1])
            ankle_diffs.append(diff)
            frame_indices.append(i)
    
    if not ankle_diffs or not all_ankle_ys:
        return {}
    
    # 2. 估計地面高度
    if orientation == 'y_down':
        # Y 向下，地面在數值最大處 (95%)
        ground_y_est = float(np.percentile(all_ankle_ys, 95))
    else:
        # Y 向上，地面在數值最小處 (5%)
        ground_y_est = float(np.percentile(all_ankle_ys, 5))
    
    # ... (其餘統計代碼不變)
    arr = np.array(ankle_diffs, dtype=float)
    mean_diff = float(np.mean(arr))
    
    assessment_parts = []
    if mean_diff < config.ground_plane_tolerance:
        assessment_parts.append("地面平整")
    else:
        assessment_parts.append("地面起伏大")
        
    assessment = f"[{' | '.join(assessment_parts)}]"
    
    return {
        "sample_count": len(arr),
        "mean_diff_mm": mean_diff,
        "std_diff_mm": float(np.std(arr)),
        "max_diff_mm": float(np.max(arr)),
        "estimated_ground_y": ground_y_est,
        "orientation": orientation,
        "assessment": assessment,
        "series": {
            "frames": frame_indices,
            "diff_mm": ankle_diffs,
            "left_ankle_y": left_ankle_ys,
            "right_ankle_y": right_ankle_ys
        }
    }


def analyze_body_symmetry(data: list, config: ValidationConfig) -> list:
    """分析身體左右對稱性（與 Step2 對齊的百分比邏輯）"""
    symmetry_pairs = [
        ("left_shoulder", "right_shoulder", "肩膀"),
        ("left_elbow", "right_elbow", "肘部"),
        ("left_wrist", "right_wrist", "手腕"),
        ("left_hip", "right_hip", "髖部"),
        ("left_knee", "right_knee", "膝蓋"),
        ("left_ankle", "right_ankle", "腳踝"),
        ("left_eye", "right_eye", "眼睛"),
    ]
    midline_refs = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    symmetry_results = []
    seen_pairs = set()
    
    for left_kp, right_kp, zh_name in symmetry_pairs:
        if zh_name in seen_pairs:
            continue
        seen_pairs.add(zh_name)
        asymmetry_percentages = []
        frame_indices = []
        
        for i, frame in enumerate(data):
            ref_points = [get_keypoint_safely(frame, ref) for ref in midline_refs]
            if any(p is None for p in ref_points):
                continue
            mid_x = float(np.mean([p[0] for p in ref_points]))
            left_point = get_keypoint_safely(frame, left_kp)
            right_point = get_keypoint_safely(frame, right_kp)
            if left_point is None or right_point is None:
                continue
            dl = abs(left_point[0] - mid_x)
            dr = abs(right_point[0] - mid_x)
            avg = (dl + dr) / 2.0
            diff_rate = safe_percentage(abs(dl - dr), avg, config.epsilon)
            asymmetry_percentages.append(diff_rate)
            frame_indices.append(i)
        
        if asymmetry_percentages:
            arr = np.array(asymmetry_percentages, dtype=float)
            mean_percent = float(np.mean(arr))
            symmetry_results.append({
                "pair_name": zh_name,
                "mean_asymmetry_percent": mean_percent,
                "std_asymmetry_percent": float(np.std(arr)),
                "max_asymmetry_percent": float(np.max(arr)),
                "sample_count": len(arr),
                "assessment": config.get_symmetry_assessment(mean_percent),
                "series": {
                    "frames": frame_indices,
                    "asymmetry_percent": asymmetry_percentages
                }
            })
    
    return symmetry_results


def analyze_relative_position_consistency(data: list, config: ValidationConfig) -> dict:
    """
    分析相對位置一致性
    """
    orientation = detect_vertical_orientation(data)
    
    violations = {
        "head_below_shoulder": 0,
        "shoulder_below_hip": 0,
        "hip_below_knee": 0,
        "knee_below_ankle": 0
    }
    
    total = len(data)
    series_data = {
        "frames": [],
        "head_shoulder_diff": [],
        "shoulder_hip_diff": [],
        "hip_knee_diff": [],
        "knee_ankle_diff": []
    }

    for i, frame in enumerate(data):
        nose = get_keypoint_safely(frame, "nose")
        ls = get_keypoint_safely(frame, "left_shoulder")
        rs = get_keypoint_safely(frame, "right_shoulder")
        lh = get_keypoint_safely(frame, "left_hip")
        rh = get_keypoint_safely(frame, "right_hip")
        lk = get_keypoint_safely(frame, "left_knee")
        rk = get_keypoint_safely(frame, "right_knee")
        la = get_keypoint_safely(frame, "left_ankle")
        ra = get_keypoint_safely(frame, "right_ankle")
        
        series_data["frames"].append(i)
        
        # Helper to check "A below B"
        def is_below(y_a, y_b):
            if orientation == 'y_down':
                return y_a > y_b # Y越大越低
            else:
                return y_a < y_b # Y越小越低

        # Head vs Shoulder
        if not any(p is None for p in [nose, ls, rs]):
            shoulder_y = (ls[1] + rs[1]) / 2
            if is_below(nose[1], shoulder_y):
                violations["head_below_shoulder"] += 1
            series_data["head_shoulder_diff"].append(shoulder_y - nose[1])
        else:
            series_data["head_shoulder_diff"].append(None)
        
        # Shoulder vs Hip
        if not any(p is None for p in [ls, rs, lh, rh]):
            shoulder_y = (ls[1] + rs[1]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            if is_below(shoulder_y, hip_y):
                violations["shoulder_below_hip"] += 1
            series_data["shoulder_hip_diff"].append(hip_y - shoulder_y)
        else:
            series_data["shoulder_hip_diff"].append(None)
        
        # Hip vs Knee
        if not any(p is None for p in [lh, rh, lk, rk]):
            hip_y = (lh[1] + rh[1]) / 2
            knee_y = (lk[1] + rk[1]) / 2
            if is_below(hip_y, knee_y):
                violations["hip_below_knee"] += 1
            series_data["hip_knee_diff"].append(knee_y - hip_y)
        else:
            series_data["hip_knee_diff"].append(None)
        
        # Knee vs Ankle
        if not any(p is None for p in [lk, rk, la, ra]):
            knee_y = (lk[1] + rk[1]) / 2
            ankle_y = (la[1] + ra[1]) / 2
            if is_below(knee_y, ankle_y):
                violations["knee_below_ankle"] += 1
            series_data["knee_ankle_diff"].append(ankle_y - knee_y)
        else:
            series_data["knee_ankle_diff"].append(None)
    
    return {
        "total_frames": total,
        "orientation": orientation,
        "head_below_shoulder_count": violations["head_below_shoulder"],
        "head_below_shoulder_rate": float(violations["head_below_shoulder"] / total * 100),
        "shoulder_below_hip_count": violations["shoulder_below_hip"],
        "shoulder_below_hip_rate": float(violations["shoulder_below_hip"] / total * 100),
        "hip_below_knee_count": violations["hip_below_knee"],
        "hip_below_knee_rate": float(violations["hip_below_knee"] / total * 100),
        "knee_below_ankle_count": violations["knee_below_ankle"],
        "knee_below_ankle_rate": float(violations["knee_below_ankle"] / total * 100),
        "series": series_data
    }


def analyze_center_stability(data: list, config: ValidationConfig) -> dict:
    """
    分析重心穩定性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 重心穩定性分析結果
    """
    centers = []
    frame_indices = []
    
    for i, frame in enumerate(data):
        ls = get_keypoint_safely(frame, "left_shoulder")
        rs = get_keypoint_safely(frame, "right_shoulder")
        lh = get_keypoint_safely(frame, "left_hip")
        rh = get_keypoint_safely(frame, "right_hip")
        
        if all(p is not None for p in [ls, rs, lh, rh]):
            center = (ls + rs + lh + rh) / 4
            centers.append(center)
            frame_indices.append(i)
    
    if not centers:
        return {}
    
    centers_arr = np.array(centers)
    axes: List[str] = []
    if isinstance(config.center_stability_axes, list):
        for axis in config.center_stability_axes:
            axis_lower = str(axis).lower()
            if axis_lower in AXIS_TO_INDEX and axis_lower not in axes:
                axes.append(axis_lower)
    if not axes:
        axes = ["x", "y", "z"]
    axis_stats: Dict[str, Dict[str, float]] = {}
    for axis in axes:
        idx = AXIS_TO_INDEX[axis]
        axis_values = centers_arr[:, idx]
        std_val = float(np.std(axis_values))
        mean_val = float(np.mean(axis_values))
        outlier_indices, stats = detect_outliers_zscore(axis_values, config.center_stability_sigma)
        axis_stats[axis] = {
            "mean_mm": mean_val,
            "std_mm": std_val,
            "outlier_count": int(len(outlier_indices)),
            "outlier_rate_percent": stats.get('outlier_rate', 0.0),
            "threshold_sigma": float(config.center_stability_sigma)
        }
    
    return {
        "sample_count": len(centers),
        "axes": axis_stats,
        "mean_center": centers_arr.mean(axis=0).tolist(),
        "series": {
            "frames": frame_indices,
            "x": centers_arr[:, 0].tolist(),
            "y": centers_arr[:, 1].tolist(),
            "z": centers_arr[:, 2].tolist()
        }
    }


def analyze_rigid_body_groups(data: list, config: ValidationConfig) -> dict:
    """
    分析剛體組距離恆定性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 剛體組分析結果
    """
    # 定義剛體組（這些關節之間的距離應該恆定）
    rigid_groups = {
        "頭部": [("left_eye", "right_eye"), ("left_ear", "right_ear")],
        "軀幹": [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")],
        "左上臂": [("left_shoulder", "left_elbow")],
        "右上臂": [("right_shoulder", "right_elbow")],
        "左前臂": [("left_elbow", "left_wrist")],
        "右前臂": [("right_elbow", "right_wrist")],
        "左大腿": [("left_hip", "left_knee")],
        "右大腿": [("right_hip", "right_knee")],
        "左小腿": [("left_knee", "left_ankle")],
        "右小腿": [("right_knee", "right_ankle")],
    }
    
    results = {}
    
    for group_name, pairs in rigid_groups.items():
        group_results = []
        
        for j1, j2 in pairs:
            distances = []
            frame_indices = []
            
            for i, frame in enumerate(data):
                p1 = get_keypoint_safely(frame, j1)
                p2 = get_keypoint_safely(frame, j2)
                
                if p1 is not None and p2 is not None:
                    dist = calculate_distance(p1, p2)
                    if dist is not None:
                        distances.append(dist)
                        frame_indices.append(i)
            
            if distances:
                arr = np.array(distances, dtype=float)
                cv = calculate_cv(arr)
                
                group_results.append({
                    "pair": f"{j1}-{j2}",
                    "mean_distance_mm": float(np.mean(arr)),
                    "cv_percent": cv,
                    "quality": config.get_quality_level_cv(cv),
                    "series": {
                        "frames": frame_indices,
                        "distance_mm": distances
                    }
                })
        
        results[group_name] = group_results
    
    return results


def analyze_topology_validation(data: list, config: ValidationConfig) -> dict:
    """
    分析拓撲結構驗證（新增）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 拓撲驗證結果
    """
    # 檢查骨架連接性
    connections = [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    
    broken_frames = 0
    connection_stats = {f"{j1}-{j2}": 0 for j1, j2 in connections}
    
    for frame in data:
        frame_broken = False
        
        for j1, j2 in connections:
            p1 = get_keypoint_safely(frame, j1)
            p2 = get_keypoint_safely(frame, j2)
            
            if p1 is None or p2 is None:
                connection_stats[f"{j1}-{j2}"] += 1
                frame_broken = True
        
        if frame_broken:
            broken_frames += 1
    
    total_frames = len(data)
    completeness = float((total_frames - broken_frames) / total_frames * 100)
    
    return {
        "total_frames": total_frames,
        "broken_frames": broken_frames,
        "completeness_rate": completeness,
        "connection_missing_counts": connection_stats
    }


def analyze_penetration_detection(data: list, config: ValidationConfig, ground_y_ref: float = None, orientation: str = 'y_down') -> dict:
    """
    分析穿透檢測（新增強化）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
        ground_y_ref: 外部傳入的參考地面高度 (可選)
        orientation: 垂直軸方向 ('y_up' or 'y_down')
    
    返回:
        dict: 穿透檢測結果
    """
    penetration_count = 0
    penetration_frames = []
    penetration_events = []  # 記錄穿透事件起始
    
    # 如果沒有外部參考，則自行估計
    if ground_y_ref is None:
        # 這裡簡單處理，若無外部傳入則重新檢測方向
        orientation = detect_vertical_orientation(data)
        all_ys = []
        for frame in data:
            la = get_keypoint_safely(frame, "left_ankle")
            ra = get_keypoint_safely(frame, "right_ankle")
            if la is not None: all_ys.append(la[1])
            if ra is not None: all_ys.append(ra[1])
            
        if orientation == 'y_down':
            ground_y_ref = float(np.percentile(all_ys, 95)) if all_ys else 0.0
        else:
            ground_y_ref = float(np.percentile(all_ys, 5)) if all_ys else 0.0
    
    # For plotting
    all_frames = []
    all_depths = [] # 0 if no penetration, >0 if penetration
    
    # New: Store signed height from ground (Positive = Above, Negative = Penetration)
    left_ankle_heights = []
    right_ankle_heights = []
    tennis_ball_heights = []

    # 檢查所有關鍵點
    check_joints = ["left_ankle", "right_ankle", "tennis_ball"]
    
    for frame_idx, frame in enumerate(data):
        all_frames.append(frame_idx)
        max_depth = 0.0
        frame_penetrated = False
        
        # Individual heights for plotting
        la_height = 0.0
        ra_height = 0.0
        tb_height = 0.0

        for joint_name in check_joints:
            kp = get_keypoint_safely(frame, joint_name)
            
            current_height = 0.0
            if kp is not None:
                # Calculate signed height from ground
                if orientation == 'y_down':
                    # Y increases downwards. Ground is at ground_y_ref.
                    # Height = Ground - Y
                    # If Y < Ground (smaller Y), it is above ground (positive height).
                    # If Y > Ground (larger Y), it is below ground (negative height).
                    current_height = float(ground_y_ref - kp[1])
                else:
                    # Y increases upwards. Ground is at ground_y_ref.
                    # Height = Y - Ground
                    # If Y > Ground, it is above ground (positive height).
                    # If Y < Ground, it is below ground (negative height).
                    current_height = float(kp[1] - ground_y_ref)

                # Penetration logic (remains same for scoring)
                is_penetrating = False
                penetration_depth = 0.0
                
                if current_height < -config.penetration_tolerance:
                    is_penetrating = True
                    penetration_depth = abs(current_height)
                
                if is_penetrating:
                    max_depth = max(max_depth, penetration_depth)
                    frame_penetrated = True
                    
                    # 記錄嚴重穿透 (深度 > 50mm)
                    if penetration_depth > 50:
                        # 檢查是否為新的穿透事件
                        is_new_event = True
                        for prev_frame, prev_joint in penetration_events:
                            if prev_joint == joint_name and frame_idx - prev_frame < 10:
                                is_new_event = False
                                break
                        
                        if is_new_event:
                            penetration_events.append((frame_idx, joint_name))
                            penetration_frames.append({
                                "frame": frame_idx,
                                "joint": joint_name,
                                "penetration_depth_mm": round(penetration_depth, 1)
                            })
            
            if joint_name == "left_ankle":
                la_height = current_height
            elif joint_name == "right_ankle":
                ra_height = current_height
            elif joint_name == "tennis_ball":
                tb_height = current_height
        
        if frame_penetrated:
            penetration_count += 1
            
        all_depths.append(max_depth)
        left_ankle_heights.append(la_height)
        right_ankle_heights.append(ra_height)
        tennis_ball_heights.append(tb_height)
    
    return {
        "total_penetrations": penetration_count,
        "penetration_rate": float(penetration_count / len(data) * 100) if data else 0.0,
        "ground_y_ref": ground_y_ref,
        "orientation": orientation,
        "penetration_frames": penetration_frames[:15],
        "series": {
            "frames": all_frames,
            "penetration_depth_mm": all_depths,
            "left_ankle_height": left_ankle_heights,
            "right_ankle_height": right_ankle_heights,
            "tennis_ball_height": tennis_ball_heights
        }
    }


def print_analysis_report(
    ground_plane: dict,
    topology: dict,
    penetration: dict,
    config: ValidationConfig
) -> None:
    """列印分析報告"""
    
    if ground_plane:
        print("\n" + "=" * 100)
        print("【1. 地面平面一致性】")
        print("=" * 100)
        print(f"平均高度差: {ground_plane['mean_diff_mm']:.2f} mm")
        print(f"估計地面高度 (Y): {ground_plane.get('estimated_ground_y', 0):.1f}")
        assessment = ground_plane['assessment'].replace('⚠️', '[WARNING]').replace('❌', '[ERROR]').replace('✅', '[OK]')
        print(f"評估: {assessment}")
    
    if topology:
        print("\n" + "=" * 100)
        print("【2. 拓撲結構驗證】")
        print("=" * 100)
        print(f"完整性: {topology['completeness_rate']:.1f}%")
        print(f"骨架斷裂幀數: {topology['broken_frames']}")
    
    if penetration:
        print("\n" + "=" * 100)
        print("【3. 穿透檢測】")
        print("=" * 100)
        print(f"穿透次數: {penetration['total_penetrations']}")
        print(f"穿透率: {penetration['penetration_rate']:.2f}%")


def calculate_overall_score(
    ground_plane: dict,
    penetration: dict,
    topology: dict
) -> dict:
    """
    計算總體評分 (0-100)
    """
    score = 100.0
    deductions = {}

    # 1. 地面平面 (Max 40)
    gp_diff = ground_plane.get('mean_diff_mm', 0)
    
    gp_deduction = 0
    if gp_diff > 100:
        gp_deduction += 20
    elif gp_diff > 50:
        gp_deduction += 10
        
    gp_deduction = min(40, gp_deduction)
    score -= gp_deduction
    deductions['ground_plane'] = gp_deduction

    # 2. 穿透 (Max 30)
    # 評分標準：基於穿透幀數佔總幀數的比例 (Penetration Rate)
    # 係數 2.0: 即 1% 穿透扣 2 分，15% 穿透即扣滿 30 分
    pen_rate = penetration.get('penetration_rate', 0)
    pen_deduction = min(30, pen_rate * 2.0) 
    score -= pen_deduction
    deductions['penetration'] = pen_deduction

    # 3. 拓撲 (Max 30)
    completeness = topology.get('completeness_rate', 100)
    topo_deduction = min(30, (100 - completeness) * 2)
    score -= topo_deduction
    deductions['topology'] = topo_deduction

    return {
        "total_score": max(0, round(score, 1)),
        "deductions": deductions
    }


def validate_spatial_consistency_analysis(
    json_3d_path: str,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    空間一致性驗證分析（主函數）
    
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
    print("\n執行地面平面分析...")
    ground_plane = analyze_ground_plane_consistency(data, config)
    
    # 獲取估計的地面高度，傳給穿透檢測
    ground_y_ref = ground_plane.get('estimated_ground_y')
    orientation = ground_plane.get('orientation', 'y_down')
    
    print("執行拓撲結構驗證...")
    topology = analyze_topology_validation(data, config)
    
    print("執行穿透檢測...")
    penetration = analyze_penetration_detection(data, config, ground_y_ref, orientation)
    
    # 列印報告
    print_analysis_report(
        ground_plane, topology, penetration, config
    )
    
    # 計算評分
    score_data = calculate_overall_score(
        ground_plane, penetration, topology
    )
    print(f"\n總體評分: {score_data['total_score']} / 100")

    # 整合結果
    results = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "source_file": str(json_3d_path),
            "total_frames": int(len(data)),
            "analysis_type": "Spatial Consistency Analysis"
        },
        "score": score_data,
        "overall_summary": {
            "ground_plane_consistent": ground_plane.get('assessment', '').startswith('[OK]') if ground_plane else None,
            "total_penetrations": penetration.get('total_penetrations', 0),
            "topology_completeness": topology.get('completeness_rate', 0.0)
        },
        "ground_plane_consistency": ground_plane,
        "topology_validation": topology,
        "penetration_detection": penetration
    }
    
    # 保存結果
    if output_json_path is None:
        # 建立 results 資料夾
        results_dir = os.path.join(os.path.dirname(json_3d_path), 'Verification Result')
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d_path))[0]
        output_json_path = os.path.join(results_dir, f"{base_name}_step5_spatial_consistency_results.json")
    
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
        print("  python step5_spatial_consistency_v2.py <json_path> [--config <config>] [--output <output>]")
    
    try:
        results = validate_spatial_consistency_analysis(
            json_3d_path,
            output_json_path,
            config_path
        )
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
