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

def analyze_ground_plane_consistency(data: list, config: ValidationConfig) -> dict:
    """
    分析地面平面一致性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 地面平面分析結果
    """
    ankle_diffs = []
    frame_indices = []
    
    for i, frame in enumerate(data):
        left_ankle = get_keypoint_safely(frame, "left_ankle")
        right_ankle = get_keypoint_safely(frame, "right_ankle")
        
        if left_ankle is not None and right_ankle is not None:
            # Y 軸差異（高度差）
            diff = abs(left_ankle[1] - right_ankle[1])
            ankle_diffs.append(diff)
            frame_indices.append(i)
    
    if not ankle_diffs:
        return {}
    
    arr = np.array(ankle_diffs, dtype=float)
    mean_diff = float(np.mean(arr))
    
    if mean_diff < config.ground_plane_tolerance:
        assessment = "[OK] 地面平面一致"
    else:
        assessment = "[!] 地面平面偏差較大"
    
    return {
        "sample_count": len(arr),
        "mean_diff_mm": mean_diff,
        "std_diff_mm": float(np.std(arr)),
        "max_diff_mm": float(np.max(arr)),
        "median_diff_mm": float(np.median(arr)),
        "assessment": assessment,
        "series": {
            "frames": frame_indices,
            "diff_mm": ankle_diffs
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
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 相對位置分析結果
    """
    violations = {
        "head_below_shoulder": 0,
        "shoulder_below_hip": 0,
        "hip_below_knee": 0,
        "knee_below_ankle": 0
    }
    
    total = len(data)
    
    # For plotting, we can track the vertical distance (positive = good, negative = violation)
    # e.g. Head - Shoulder Y (since Y is down, Head Y < Shoulder Y is good. So Shoulder Y - Head Y > 0 is good)
    series_data = {
        "frames": [],
        "head_shoulder_diff": [], # Shoulder Y - Head Y
        "shoulder_hip_diff": [],  # Hip Y - Shoulder Y
        "hip_knee_diff": [],      # Knee Y - Hip Y
        "knee_ankle_diff": []     # Ankle Y - Knee Y
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
        
        # Head vs Shoulder
        if not any(p is None for p in [nose, ls, rs]):
            shoulder_y = (ls[1] + rs[1]) / 2
            if nose[1] > shoulder_y:
                violations["head_below_shoulder"] += 1
            series_data["head_shoulder_diff"].append(shoulder_y - nose[1])
        else:
            series_data["head_shoulder_diff"].append(None)
        
        # Shoulder vs Hip
        if not any(p is None for p in [ls, rs, lh, rh]):
            shoulder_y = (ls[1] + rs[1]) / 2
            hip_y = (lh[1] + rh[1]) / 2
            if shoulder_y > hip_y:
                violations["shoulder_below_hip"] += 1
            series_data["shoulder_hip_diff"].append(hip_y - shoulder_y)
        else:
            series_data["shoulder_hip_diff"].append(None)
        
        # Hip vs Knee
        if not any(p is None for p in [lh, rh, lk, rk]):
            hip_y = (lh[1] + rh[1]) / 2
            knee_y = (lk[1] + rk[1]) / 2
            if hip_y > knee_y:
                violations["hip_below_knee"] += 1
            series_data["hip_knee_diff"].append(knee_y - hip_y)
        else:
            series_data["hip_knee_diff"].append(None)
        
        # Knee vs Ankle
        if not any(p is None for p in [lk, rk, la, ra]):
            knee_y = (lk[1] + rk[1]) / 2
            ankle_y = (la[1] + ra[1]) / 2
            if knee_y > ankle_y:
                violations["knee_below_ankle"] += 1
            series_data["knee_ankle_diff"].append(ankle_y - knee_y)
        else:
            series_data["knee_ankle_diff"].append(None)
    
    return {
        "total_frames": total,
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


def analyze_penetration_detection(data: list, config: ValidationConfig) -> dict:
    """
    分析穿透檢測（新增強化）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 穿透檢測結果
    """
    penetration_count = 0
    penetration_frames = []
    penetration_events = []  # 記錄穿透事件起始
    
    # For plotting
    all_frames = []
    all_depths = [] # 0 if no penetration, >0 if penetration
    
    for frame_idx, frame in enumerate(data):
        all_frames.append(frame_idx)
        max_depth = 0.0
        
        # 檢查手腕是否穿透地面
        for wrist_name in ["left_wrist", "right_wrist"]:
            wrist = get_keypoint_safely(frame, wrist_name)
            left_ankle = get_keypoint_safely(frame, "left_ankle")
            right_ankle = get_keypoint_safely(frame, "right_ankle")
            
            if wrist is not None and left_ankle is not None and right_ankle is not None:
                ground_y = (left_ankle[1] + right_ankle[1]) / 2
                
                # 如果手腕低於地面（Y 軸更大）
                if wrist[1] > ground_y + config.penetration_tolerance:
                    penetration_depth = float(wrist[1] - ground_y)
                    max_depth = max(max_depth, penetration_depth)
                    
                    # 檢查是否為新的穿透事件（與前一個事件間隔超過 5 幀）
                    is_new_event = True
                    for prev_frame, prev_joint in penetration_events:
                        if prev_joint == wrist_name and frame_idx - prev_frame < 5:
                            is_new_event = False
                            break
                    
                    if is_new_event:
                        penetration_events.append((frame_idx, wrist_name))
                        penetration_count += 1
                        penetration_frames.append({
                            "frame": frame_idx,
                            "joint": wrist_name,
                            "penetration_depth_mm": penetration_depth
                        })
        
        all_depths.append(max_depth)
    
    return {
        "total_penetrations": penetration_count,
        "penetration_rate": float(penetration_count / len(data) * 100) if data else 0.0,
        "penetration_frames": penetration_frames[:10],  # 只返回前 10 個
        "series": {
            "frames": all_frames,
            "penetration_depth_mm": all_depths
        }
    }


def print_analysis_report(
    ground_plane: dict,
    symmetry_results: list,
    relative_position: dict,
    center_stability: dict,
    rigid_groups: dict,
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
        assessment = ground_plane['assessment'].replace('⚠️', '[WARNING]').replace('❌', '[ERROR]').replace('✅', '[OK]')
        print(f"評估: {assessment}")
    
    if symmetry_results:
        print("\n" + "=" * 100)
        print("【2. 身體左右對稱性】")
        print("=" * 100)
        for result in symmetry_results:
            print(
                f"{result['pair_name']:<10} 不對稱度:{result['mean_asymmetry_percent']:.2f}% "
                f"{result['assessment']}"
            )
    
    if relative_position:
        print("\n" + "=" * 100)
        print("【3. 相對位置一致性】")
        print("=" * 100)
        if relative_position['head_below_shoulder_count'] > 0:
            print(f"[!] 頭部低於肩膀: {relative_position['head_below_shoulder_count']} 次")
        if relative_position['shoulder_below_hip_count'] > 0:
            print(f"[!] 肩膀低於髖部: {relative_position['shoulder_below_hip_count']} 次")
    
    if center_stability:
        print("\n" + "=" * 100)
        print("【4. 重心穩定性】")
        print("=" * 100)
        axes_stats = center_stability.get('axes', {})
        for axis_name, stats in axes_stats.items():
            axis_label = axis_name.upper()
            print(
                f"{axis_label} 軸標準差: {stats['std_mm']:.2f} mm"
                f" (異常 {stats['outlier_count']}, 門檻 {stats['threshold_sigma']:.1f}σ)"
            )
    
    if rigid_groups:
        print("\n" + "=" * 100)
        print("【5. 剛體組距離恆定性】")
        print("=" * 100)
        for group_name, pairs in rigid_groups.items():
            print(f"\n{group_name}:")
            for pair in pairs:
                print(f"  {pair['pair']}: CV={pair['cv_percent']:.2f}% ({pair['quality']})")
    
    if topology:
        print("\n" + "=" * 100)
        print("【6. 拓撲結構驗證】")
        print("=" * 100)
        print(f"完整性: {topology['completeness_rate']:.1f}%")
        print(f"骨架斷裂幀數: {topology['broken_frames']}")
    
    if penetration:
        print("\n" + "=" * 100)
        print("【7. 穿透檢測】")
        print("=" * 100)
        print(f"穿透次數: {penetration['total_penetrations']}")
        print(f"穿透率: {penetration['penetration_rate']:.2f}%")


def calculate_overall_score(
    ground_plane: dict,
    symmetry: list,
    relative: dict,
    rigid: dict,
    penetration: dict,
    topology: dict
) -> dict:
    """
    計算總體評分 (0-100)
    """
    score = 100.0
    deductions = {}

    # 1. 地面平面 (Max 20)
    gp_diff = ground_plane.get('mean_diff_mm', 0)
    gp_deduction = 0
    if gp_diff > 100:
        gp_deduction = 20
    elif gp_diff > 50:
        gp_deduction = 10
    elif gp_diff > 20:
        gp_deduction = 5
    score -= gp_deduction
    deductions['ground_plane'] = gp_deduction

    # 2. 對稱性 (Max 20)
    avg_asym = 0
    if symmetry:
        avg_asym = sum(s.get('mean_asymmetry_percent', 0) for s in symmetry) / len(symmetry)
    
    sym_deduction = min(20, avg_asym * 0.5) # e.g. 40% asym -> 20 pts
    score -= sym_deduction
    deductions['symmetry'] = sym_deduction

    # 3. 相對位置 (Max 20)
    total_frames = relative.get('total_frames', 1)
    violations = (
        relative.get('head_below_shoulder_count', 0) +
        relative.get('shoulder_below_hip_count', 0) +
        relative.get('hip_below_knee_count', 0) +
        relative.get('knee_below_ankle_count', 0)
    )
    violation_rate = (violations / total_frames) 
    rel_deduction = min(20, violation_rate * 20)
    score -= rel_deduction
    deductions['relative_position'] = rel_deduction

    # 4. 剛體組 (Max 20)
    cv_sum = 0
    cv_count = 0
    for group, pairs in rigid.items():
        for pair in pairs:
            cv_sum += pair.get('cv_percent', 0)
            cv_count += 1
    
    avg_cv = cv_sum / cv_count if cv_count > 0 else 0
    rigid_deduction = min(20, avg_cv * 2) # e.g. 10% CV -> 20 pts
    score -= rigid_deduction
    deductions['rigid_body'] = rigid_deduction

    # 5. 穿透 (Max 10)
    pen_rate = penetration.get('penetration_rate', 0)
    pen_deduction = min(10, pen_rate * 0.2) # 50% penetration -> 10 pts
    score -= pen_deduction
    deductions['penetration'] = pen_deduction

    # 6. 拓撲 (Max 10)
    completeness = topology.get('completeness_rate', 100)
    topo_deduction = min(10, (100 - completeness))
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
    
    print("執行身體對稱性分析...")
    symmetry_results = analyze_body_symmetry(data, config)
    
    print("執行相對位置分析...")
    relative_position = analyze_relative_position_consistency(data, config)
    
    print("執行重心穩定性分析...")
    center_stability = analyze_center_stability(data, config)
    
    print("執行剛體組分析...")
    rigid_groups = analyze_rigid_body_groups(data, config)
    
    print("執行拓撲結構驗證...")
    topology = analyze_topology_validation(data, config)
    
    print("執行穿透檢測...")
    penetration = analyze_penetration_detection(data, config)
    
    # 列印報告
    print_analysis_report(
        ground_plane, symmetry_results, relative_position,
        center_stability, rigid_groups, topology, penetration, config
    )
    
    # 計算評分
    score_data = calculate_overall_score(
        ground_plane, symmetry_results, relative_position,
        rigid_groups, penetration, topology
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
        "body_symmetry": symmetry_results,
        "relative_position_consistency": relative_position,
        "center_stability": center_stability,
        "rigid_body_groups": rigid_groups,
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
        json_3d_path = "0306_3__trajectory/trajectory__2/0306_3__2(3D_trajectory_smoothed).json"
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
