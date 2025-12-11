"""
Step 2: 骨骼一致性驗證分析
驗證 3D 重建結果的骨骼長度一致性、對稱性和穩定性

功能：
  1. 骨骼長度統計分析（均值/標準差/CV）
  2. 左右對稱性檢查
  3. 幀間跳動（Spike）檢測
  4. 全身身高分析（Scaling Drift）
  5. 各關節深度穩定性
  6. 關節角度合理性檢查
  7. 骨向量方向穩定性分析
  8. 骨長分布統計檢驗
"""

import numpy as np
from scipy import stats
from datetime import datetime
import os
import sys

# 引入共用模組
from .utils import (
    get_keypoint_safely,
    calculate_distance,
    calculate_angle,
    calculate_unit_vector,
    load_json_file,
    save_json_results,
    calculate_cv,
    detect_outliers_iqr,
    detect_outliers_zscore,
    generate_output_path,
    get_keypoint_name_zh,
)
from config import load_config, ValidationConfig


# ========================================================
# 骨骼定義
# ========================================================

BONE_DEFINITIONS = {
    # 頭部（注意：head 使用單耳僅作參考，實際應使用雙耳中點）
    "頭部": ("left_eye", "right_eye"),
    
    # 軀幹
    "脊柱": ("left_shoulder", "left_hip"),
    "肩寬": ("left_shoulder", "right_shoulder"),
    "骨盆": ("left_hip", "right_hip"),
    
    # 左上肢
    "左上臂": ("left_shoulder", "left_elbow"),
    "左前臂": ("left_elbow", "left_wrist"),
    "左整臂": ("left_shoulder", "left_wrist"),
    
    # 右上肢
    "右上臂": ("right_shoulder", "right_elbow"),
    "右前臂": ("right_elbow", "right_wrist"),
    "右整臂": ("right_shoulder", "right_wrist"),
    
    # 左下肢
    "左大腿": ("left_hip", "left_knee"),
    "左小腿": ("left_knee", "left_ankle"),
    "左整腿": ("left_hip", "left_ankle"),
    
    # 右下肢
    "右大腿": ("right_hip", "right_knee"),
    "右小腿": ("right_knee", "right_ankle"),
    "右整腿": ("right_hip", "right_ankle"),
}

BONE_NAMES_ZH = {}

SYMMETRY_PAIRS = [
    ("左上臂", "右上臂", "上臂"),
    ("左前臂", "右前臂", "前臂"),
    ("左整臂", "右整臂", "整臂"),
    ("左大腿", "右大腿", "大腿"),
    ("左小腿", "右小腿", "小腿"),
    ("左整腿", "右整腿", "整腿"),
]


# ========================================================
# 核心分析函數
# ========================================================

def analyze_bone_lengths(data: list, config: ValidationConfig) -> dict:
    """
    分析骨骼長度統計
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 骨骼長度統計結果
    """
    bone_lengths = {name: [] for name in BONE_DEFINITIONS.keys()}
    bone_lengths_frames = {name: [] for name in BONE_DEFINITIONS.keys()}
    
    # 收集所有幀的骨骼長度
    for frame_idx, frame in enumerate(data):
        for bone_name, (j1, j2) in BONE_DEFINITIONS.items():
            if j1 in frame and j2 in frame:
                p1 = get_keypoint_safely(frame, j1)
                p2 = get_keypoint_safely(frame, j2)
                
                if p1 is not None and p2 is not None:
                    length = calculate_distance(p1, p2)
                    if length is not None:
                        # Convert mm to cm
                        length_cm = length / 10.0
                        bone_lengths[bone_name].append(length_cm)
                        bone_lengths_frames[bone_name].append((frame_idx, length_cm))
    
    # 計算統計量
    bone_stats = {}
    spikes = []
    
    for bone_name, lengths in bone_lengths.items():
        if not lengths:
            bone_stats[bone_name] = None
            continue
        
        arr = np.array(lengths, dtype=float)
        mean_L = float(np.mean(arr))
        std_L = float(np.std(arr))
        cv = calculate_cv(arr)
        
        # 幀間跳動分析
        if len(arr) >= 2:
            diffs = np.abs(np.diff(arr))
            mean_diff = float(np.mean(diffs))
            max_diff = float(np.max(diffs))
            
            # Spike 檢測 (Convert config threshold from mm to cm)
            threshold = max(config.bone_spike_ratio * mean_L, config.bone_spike_min_mm / 10.0)
            frames_for_bone = bone_lengths_frames[bone_name]
            
            for i, d in enumerate(diffs):
                if d > threshold:
                    f_prev, L_prev = frames_for_bone[i]
                    f_curr, L_curr = frames_for_bone[i + 1]
                    spikes.append({
                        "bone": bone_name,
                        "bone_zh": BONE_NAMES_ZH.get(bone_name, bone_name),
                        "frame_prev": int(f_prev),
                        "frame_curr": int(f_curr),
                        "diff": float(d),
                        "L_prev": float(L_prev),
                        "L_curr": float(L_curr),
                    })
        else:
            mean_diff = 0.0
            max_diff = 0.0
        
        # 統計檢驗（新增）
        shapiro_stat, shapiro_p = stats.shapiro(arr) if len(arr) >= 3 else (None, None)
        
        # 異常值檢測（新增）
        outlier_indices_iqr, iqr_stats = detect_outliers_iqr(arr)
        outlier_indices_zscore, zscore_stats = detect_outliers_zscore(arr)
        
        bone_stats[bone_name] = {
            "mean": mean_L,
            "std": std_L,
            "cv": cv,
            "count": len(arr),
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "quality_level": config.get_quality_level_cv(cv),
            "shapiro_test": {
                "statistic": float(shapiro_stat) if shapiro_stat else None,
                "p_value": float(shapiro_p) if shapiro_p else None,
                "is_normal": bool(shapiro_p > 0.05) if shapiro_p else None
            },
            "outliers_iqr": {
                "count": iqr_stats["outlier_count"],
                "rate": iqr_stats["outlier_rate"],
                "bounds": [iqr_stats["lower_bound"], iqr_stats["upper_bound"]]
            },
            "outliers_zscore": {
                "count": zscore_stats["outlier_count"],
                "rate": zscore_stats["outlier_rate"]
            },
            "series_data": [float(x) for x in arr]  # Add raw series data for plotting
        }
    
    return {
        "bone_stats": bone_stats,
        "spikes": sorted(spikes, key=lambda s: s["diff"], reverse=True)
    }


def analyze_symmetry(bone_stats: dict, config: ValidationConfig) -> list:
    """
    分析左右對稱性
    
    參數:
        bone_stats: 骨骼統計結果
        config: 驗證配置
    
    返回:
        list: 對稱性分析結果
    """
    symmetry_results = []
    
    for left_bone, right_bone, zh_name in SYMMETRY_PAIRS:
        left_stat = bone_stats.get(left_bone)
        right_stat = bone_stats.get(right_bone)
        
        if not left_stat or not right_stat:
            continue
        
        lm = left_stat["mean"]
        rm = right_stat["mean"]
        diff = abs(lm - rm)
        avg = (lm + rm) / 2
        diff_rate = float(diff / avg * 100) if avg > config.epsilon else 0.0
        
        assessment = config.get_symmetry_assessment(diff_rate)
        
        # 新增：參考 CV 值來判斷哪一邊可能有問題
        l_cv = left_stat["cv"]
        r_cv = right_stat["cv"]
        cv_diff = abs(l_cv - r_cv)
        
        # 如果對稱性差且 CV 差異大，標記 CV 高的那一邊
        if diff_rate > config.bone_symmetry_acceptable and cv_diff > 5.0:
            if l_cv > r_cv:
                assessment += f" (左側不穩 CV:{l_cv:.1f}%)"
            else:
                assessment += f" (右側不穩 CV:{r_cv:.1f}%)"
        
        symmetry_results.append({
            "pair_name": zh_name,
            "left_mean_cm": float(lm),
            "right_mean_cm": float(rm),
            "difference_cm": float(diff),
            "difference_percent": diff_rate,
            "assessment": assessment,
            "left_cv": float(l_cv),
            "right_cv": float(r_cv)
        })
    
    return symmetry_results


def analyze_body_height(data: list, config: ValidationConfig) -> dict:
    """
    分析全身身高（Scaling Drift）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 身高分析結果
    """
    heights = []
    
    for frame in data:
        nose = get_keypoint_safely(frame, "nose")
        left_ankle = get_keypoint_safely(frame, "left_ankle")
        right_ankle = get_keypoint_safely(frame, "right_ankle")
        
        if all(p is not None for p in [nose, left_ankle, right_ankle]):
            ankle_mid = (left_ankle + right_ankle) / 2
            height = float(np.linalg.norm(nose - ankle_mid))
            # Convert mm to cm
            heights.append(height / 10.0)
    
    if not heights:
        return {}
    
    H = np.array(heights, dtype=float)
    mean_h = float(np.mean(H))
    std_h = float(np.std(H))
    cv_h = calculate_cv(H)
    
    if cv_h < config.bone_cv_excellent:
        assessment = "[OK] 身高比例穩定（幾乎沒有 scaling 漂移）"
    elif cv_h < config.bone_cv_good:
        assessment = "[!] 身高比例尚可（有輕微 scaling 變化）"
    else:
        assessment = "❌ 身高比例波動明顯，可能有三角化/標定問題"
    
    # 新增：身高絕對值合理性檢查
    # 假設正常成人身高範圍 140cm ~ 210cm
    HEIGHT_MIN = 140.0
    HEIGHT_MAX = 210.0
    scale_warning = None
    
    if mean_h < HEIGHT_MIN or mean_h > HEIGHT_MAX:
        scale_warning = f"⚠️ 警告：平均身高 ({mean_h:.1f}cm) 超出正常範圍 ({HEIGHT_MIN}-{HEIGHT_MAX}cm)，請檢查相機標定或單位。"
        assessment += f" | {scale_warning}"

    return {
        'sample_count': len(H),
        'mean_cm': mean_h,
        'std_cm': std_h,
        'cv_percent': cv_h,
        'min_cm': float(np.min(H)),
        'max_cm': float(np.max(H)),
        'max_mean_ratio': float(np.max(H) / mean_h) if mean_h > config.epsilon else 0.0,
        'assessment': assessment,
        'scale_warning': scale_warning
    }


def analyze_joint_depths(data: list, config: ValidationConfig) -> dict:
    """
    分析各關節深度穩定性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 關節深度分析結果
    """
    key_joints = [
        "nose", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_wrist", "right_wrist",
        "left_ankle", "right_ankle"
    ]
    
    joint_depths = {j: [] for j in key_joints}
    
    for frame_idx, frame in enumerate(data):
        for joint in key_joints:
            point = get_keypoint_safely(frame, joint)
            if point is not None:
                # Convert mm to cm
                joint_depths[joint].append((frame_idx, point[2] / 10.0))  # z 值
    
    depth_stats = {}
    for joint, depths in joint_depths.items():
        if not depths:
            continue
        
        zs = np.array([z for (_, z) in depths], dtype=float)
        mean_z = float(np.mean(zs))
        std_z = float(np.std(zs))
        cv_z = calculate_cv(zs)
        
        depth_stats[joint] = {
            'sample_count': len(zs),
            'mean_z_cm': mean_z,
            'std_z_cm': std_z,
            'cv_percent': cv_z,
            'min_z_cm': float(np.min(zs)),
            'max_z_cm': float(np.max(zs)),
        }
    
    return depth_stats


def analyze_joint_angles(data: list, config: ValidationConfig) -> dict:
    """
    分析關節角度合理性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 關節角度分析結果
    """
    joint_angle_defs = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
    }
    
    # 定義特定關節的寬容度 (允許伸直)
    # 膝蓋和手肘允許接近 180 度甚至稍微過伸
    EXTENDED_JOINTS = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
    EXTENDED_MAX_ANGLE = 185.0  # 允許稍微過伸
    
    joint_angles = {name: [] for name in joint_angle_defs.keys()}
    
    for frame in data:
        for joint_name, (j1, j2, j3) in joint_angle_defs.items():
            p1 = get_keypoint_safely(frame, j1)
            p2 = get_keypoint_safely(frame, j2)
            p3 = get_keypoint_safely(frame, j3)
            
            if all(p is not None for p in [p1, p2, p3]):
                angle = calculate_angle(p1, p2, p3)
                if angle is not None:
                    joint_angles[joint_name].append(angle)
    
    angle_stats = {}
    for joint_name, angles in joint_angles.items():
        if not angles:
            continue
        
        arr = np.array(angles, dtype=float)
        
        # 根據關節類型決定閾值
        max_angle = EXTENDED_MAX_ANGLE if joint_name in EXTENDED_JOINTS else config.joint_angle_max
        min_angle = config.joint_angle_min
        
        abnormal_indices = np.where((arr < min_angle) | (arr > max_angle))[0]
        abnormal = len(abnormal_indices)
        
        # 收集異常詳情
        abnormal_details = []
        for idx in abnormal_indices:
            abnormal_details.append({
                "frame": int(idx),
                "angle": float(arr[idx]),
                "type": "too_small" if arr[idx] < min_angle else "too_large"
            })
        
        angle_stats[joint_name] = {
            'sample_count': len(arr),
            'mean_angle_deg': float(np.mean(arr)),
            'std_angle_deg': float(np.std(arr)),
            'min_angle_deg': float(np.min(arr)),
            'max_angle_deg': float(np.max(arr)),
            'abnormal_count': abnormal,
            'abnormal_rate': float(abnormal / len(arr) * 100),
            'abnormal_details': sorted(abnormal_details, key=lambda x: abs(x['angle'] - 90), reverse=True)  # 按異常程度排序
        }
    
    return angle_stats


def analyze_bone_orientation_stability(data: list, config: ValidationConfig) -> dict:
    """
    分析骨向量方向時間穩定性
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 骨向量方向穩定性結果
    """
    orientation_bones = {
        "左上臂": ("left_shoulder", "left_elbow"),
        "右上臂": ("right_shoulder", "right_elbow"),
        "左大腿": ("left_hip", "left_knee"),
        "右大腿": ("right_hip", "right_knee"),
    }
    
    bone_orientations = {name: [] for name in orientation_bones.keys()}
    
    for frame_idx, frame in enumerate(data):
        for bone_name, (j1, j2) in orientation_bones.items():
            p1 = get_keypoint_safely(frame, j1)
            p2 = get_keypoint_safely(frame, j2)
            
            if p1 is not None and p2 is not None:
                v = p2 - p1
                u = calculate_unit_vector(v)
                if u is not None:
                    bone_orientations[bone_name].append((frame_idx, u))
    
    orientation_stats = {}
    for bone_name, vec_list in bone_orientations.items():
        if len(vec_list) < 2:
            continue
        
        angles = []
        for i in range(len(vec_list) - 1):
            _, u1 = vec_list[i]
            _, u2 = vec_list[i + 1]
            cos_a = np.dot(u1, u2)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_a)))
            angles.append(angle)
        
        arr = np.array(angles, dtype=float)
        orientation_stats[bone_name] = {
            'sample_count': len(arr),
            'mean_angle_change_deg': float(np.mean(arr)),
            'max_angle_change_deg': float(np.max(arr)),
            'std_angle_change_deg': float(np.std(arr)),
        }
    
    return orientation_stats


def print_analysis_report(
    bone_analysis: dict,
    symmetry_results: list,
    body_height_stats: dict,
    depth_stats: dict,
    angle_stats: dict,
    orientation_stats: dict,
    config: ValidationConfig
) -> None:
    """列印分析報告"""
    
    bone_stats = bone_analysis["bone_stats"]
    spikes = bone_analysis["spikes"]
    
    print("\n" + "=" * 100)
    print("【1. 骨骼長度一致性驗證】")
    print("=" * 100)
    print(f"{'骨骼':<18} {'樣本數':<8} {'平均(cm)':<12} {'標準差(cm)':<12} {'CV(%)':<10} {'品質':<10} {'常態分布':<12}")
    print("-" * 100)
    
    all_cvs = []
    for bone_name, stats in bone_stats.items():
        zh = BONE_NAMES_ZH.get(bone_name, bone_name)
        if stats is None:
            print(f"{zh:<18} {'無數據':<8}")
            continue
        
        # Shapiro 檢驗結果顯示
        shapiro_result = stats.get('shapiro_test', {})
        is_normal = shapiro_result.get('is_normal')
        normal_text = '[V]常態' if is_normal else '[X]非常態' if is_normal is False else 'N/A'
        
        print(f"{zh:<18} {stats['count']:<8d} {stats['mean']:>10.2f}  {stats['std']:>10.2f}  "
              f"{stats['cv']:>8.2f}  {stats['quality_level']:<10} {normal_text:<12}")
        all_cvs.append(stats['cv'])
    
    print("\n" + "=" * 100)
    print("【2. 左右對稱性檢查】")
    print("=" * 100)
    for result in symmetry_results:
        print(f"{result['pair_name']:<15} 左:{result['left_mean_cm']:>8.2f} 右:{result['right_mean_cm']:>8.2f} "
              f"差異:{result['difference_percent']:>6.2f}% {result['assessment']}")
    
    print("\n" + "=" * 100)
    print("【3. 整體骨長穩定性評估】")
    print("=" * 100)
    if all_cvs:
        avg_cv = float(np.mean(all_cvs))
        quality = config.get_quality_level_cv(avg_cv)
        print(f"平均變異係數: {avg_cv:.2f}%")
        print(f"品質等級: {quality}")
    
    if body_height_stats:
        print("\n" + "=" * 100)
        print("【4. 全身身高分析（Scaling Drift）】")
        print("=" * 100)
        print(f"樣本數: {body_height_stats['sample_count']}")
        print(f"平均身高: {body_height_stats['mean_cm']:.2f} cm")
        print(f"變異係數: {body_height_stats['cv_percent']:.2f}%")
        print(f"評估: {body_height_stats['assessment']}")
    
    if spikes:
        print("\n" + "=" * 100)
        print("【5. 骨長突然跳動（Spike）分析】")
        print("=" * 100)
        print(f"總共偵測到 {len(spikes)} 個 spike")
        print(f"\nTOP 10:")
        for i, spike in enumerate(spikes[:10], 1):
            print(f"{i:2d}. {spike['bone_zh']:<10} Frame {spike['frame_prev']:4d}→{spike['frame_curr']:4d} "
                  f"跳動:{spike['diff']:>6.2f}cm")

    if depth_stats:
        print("\n" + "=" * 100)
        print("【6. 關鍵點深度穩定性】")
        print("=" * 100)
        print(f"{'關鍵點':<12}{'樣本數':>8}{'平均Z(cm)':>14}{'CV(%)':>10}{'範圍(cm)':>12}")
        for joint, stats in depth_stats.items():
            zh_joint = get_keypoint_name_zh(joint)
            print(f"{zh_joint:<12}{stats['sample_count']:>8d}{stats['mean_z_cm']:>14.1f}{stats['cv_percent']:>10.2f}{(stats['max_z_cm']-stats['min_z_cm']):>12.1f}")

    if angle_stats:
        print("\n" + "=" * 100)
        print("【7. 關節角度合理性】")
        print("=" * 100)
        
        # 收集所有異常用於詳細分析
        all_abnormals = []
        for joint, stats in angle_stats.items():
            zh_joint = get_keypoint_name_zh(joint)
            print(f"{zh_joint:<8} 平均 {stats['mean_angle_deg']:.1f}° 範圍 {stats['min_angle_deg']:.1f}°~{stats['max_angle_deg']:.1f}° "
                  f"異常 {stats['abnormal_count']} ({stats['abnormal_rate']:.2f}%)")
            
            if stats['abnormal_count'] > 0:
                for detail in stats['abnormal_details']:
                    all_abnormals.append({
                        'joint': joint,
                        'joint_zh': zh_joint,
                        **detail
                    })
        
        # 詳細異常分析
        if all_abnormals:
            print("\n" + "─" * 100)
            print("【7.1 關節角度異常詳細分析】")
            print("─" * 100)
            
            # 按嚴重程度分級
            severe = [x for x in all_abnormals if abs(x['angle'] - 90) > 60]  # 極端異常
            moderate = [x for x in all_abnormals if 30 < abs(x['angle'] - 90) <= 60]
            mild = [x for x in all_abnormals if abs(x['angle'] - 90) <= 30]
            
            print(f"總異常數: {len(all_abnormals)} 個")
            print(f"  • 嚴重 (角度極端): {len(severe)} 個 ({len(severe)/len(all_abnormals)*100:.1f}%)")
            print(f"  • 中等: {len(moderate)} 個 ({len(moderate)/len(all_abnormals)*100:.1f}%)")
            print(f"  • 輕微: {len(mild)} 個 ({len(mild)/len(all_abnormals)*100:.1f}%)")
            print()
            
            # 嚴重異常 - 全部顯示
            if severe:
                print("[!] 嚴重角度異常 - 全部顯示:")
                for idx, item in enumerate(severe[:20], 1):  # 最多20個
                    print(f" {idx:3d}. {item['joint_zh']:<6} Frame {item['frame']:3d}: {item['angle']:6.1f}° ({item['type']})")
                if len(severe) > 20:
                    print(f" ⋮ 其餘 {len(severe)-20} 個嚴重異常請參閱 JSON")
                print()
            
            # 中等異常 - 顯示前10個
            if moderate:
                display_count = min(10, len(moderate))
                print(f"📊 中等角度異常 - 顯示前 {display_count} 個，共 {len(moderate)} 個:")
                moderate_sorted = sorted(moderate, key=lambda x: abs(x['angle'] - 90), reverse=True)
                for idx, item in enumerate(moderate_sorted[:display_count], 1):
                    print(f" {idx:3d}. {item['joint_zh']:<6} Frame {item['frame']:3d}: {item['angle']:6.1f}° ({item['type']})")
                if len(moderate) > display_count:
                    print(f" ⋮ 其餘 {len(moderate)-display_count} 個中等異常請參閱 JSON")
                print()
            
            # 輕微異常 - 僅統計
            if mild:
                print(f"📋 輕微角度異常 - 共 {len(mild)} 個，詳見 JSON")
                print()
            
            print("💾 完整異常列表已儲存至 JSON: detailed_angle_anomalies")

    if orientation_stats:
        print("\n" + "=" * 100)
        print("【8. 骨向量方向穩定性】")
        print("=" * 100)
        for bone, stats in orientation_stats.items():
            zh_bone = BONE_NAMES_ZH.get(bone, bone)
            print(f"{zh_bone:<15} 平均旋轉 {stats['mean_angle_change_deg']:.2f}°/frame, 最大 {stats['max_angle_change_deg']:.2f}°")


def validate_bone_consistency_analysis(
    json_3d_path: str,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    骨骼一致性驗證分析（主函數）
    
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
    print("\n執行骨骼長度分析...")
    bone_analysis = analyze_bone_lengths(data, config)
    
    print("執行對稱性分析...")
    symmetry_results = analyze_symmetry(bone_analysis["bone_stats"], config)
    
    print("執行身高分析...")
    body_height_stats = analyze_body_height(data, config)
    
    print("執行關節深度分析...")
    depth_stats = analyze_joint_depths(data, config)
    
    print("執行關節角度分析...")
    angle_stats = analyze_joint_angles(data, config)
    
    print("執行骨向量方向分析...")
    orientation_stats = analyze_bone_orientation_stability(data, config)
    
    # 列印報告
    print_analysis_report(
        bone_analysis, symmetry_results, body_height_stats,
        depth_stats, angle_stats, orientation_stats, config
    )
    
    # 整合結果
    all_cvs = [s['cv'] for s in bone_analysis["bone_stats"].values() if s]
    avg_cv = float(np.mean(all_cvs)) if all_cvs else 0.0
    
    results = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "source_file": str(json_3d_path),
            "total_frames": int(len(data)),
            "analysis_type": "Bone Consistency Analysis"
        },
        "overall_summary": {
            "average_cv": float(avg_cv),
            "quality_level": config.get_quality_level_cv(avg_cv),
            "total_bones_analyzed": len([s for s in bone_analysis["bone_stats"].values() if s]),
            "total_spikes_detected": len(bone_analysis["spikes"])
        },
        "overall_quality": {
            "average_cv": avg_cv,
            "quality_level": config.get_quality_level_cv(avg_cv)
        },
        "bone_statistics": {
            name: stats for name, stats in bone_analysis["bone_stats"].items()
        },
        "symmetry_analysis": symmetry_results,
        "body_height_analysis": body_height_stats,
        "joint_depth_stability": depth_stats,
        "joint_angle_analysis": angle_stats,
        "bone_orientation_stability": orientation_stats,
        "spike_detection": {
            "total_spikes": len(bone_analysis["spikes"]),
            "top_10_spikes": bone_analysis["spikes"][:10]
        },
        "detailed_angle_anomalies": {
            joint: {
                "total_abnormals": stats['abnormal_count'],
                "abnormal_rate": stats['abnormal_rate'],
                "abnormal_details": stats.get('abnormal_details', [])
            }
            for joint, stats in angle_stats.items()
            if stats['abnormal_count'] > 0
        }
    }
    
    # 保存結果
    if output_json_path is None:
        # 建立 results 資料夾
        results_dir = os.path.join(os.path.dirname(json_3d_path), 'Verification Result')
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d_path))[0]
        output_json_path = os.path.join(results_dir, f"{base_name}_step2_bone_consistency_results.json")
    
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
        json_3d_path = "trajectory__2/0306_3__2(3D_trajectory_smoothed).json"
        config_path = None
        output_json_path = None
        print("提示: 可使用命令列參數:")
        print("  python step2_bone_consistency_v2.py <json_path> [--config <config>] [--output <output>]")
    
    try:
        results = validate_bone_consistency_analysis(
            json_3d_path,
            output_json_path,
            config_path
        )
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
