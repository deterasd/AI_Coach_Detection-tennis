"""
Step 3: 深度合理性驗證分析
驗證 3D 重建結果的深度（Z 軸）合理性和一致性

功能：
  1. 深度範圍檢查（有效深度區間）
  2. 深度變異係數分析
  3. 深度跳動檢測
  4. 深度對稱性檢查
  5. 深度邏輯檢查（肢體遠近關係）
  6. 深度梯度分析
  7. Z 軸統計特性分析
"""

import numpy as np
from datetime import datetime
import sys
import os
import json

# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入共用模組
try:
    from .utils import (
        get_keypoint_safely,
        load_json_file,
        save_json_results,
        calculate_cv,
        detect_outliers_zscore,
        generate_output_path,
        KEYPOINT_NAMES_EN,
        get_keypoint_name_zh,
    )
except ImportError:
    from utils import (
        get_keypoint_safely,
        load_json_file,
        save_json_results,
        calculate_cv,
        detect_outliers_zscore,
        generate_output_path,
        KEYPOINT_NAMES_EN,
        get_keypoint_name_zh,
    )

from config import load_config, ValidationConfig


# ========================================================
# 輔助函數 (相機變換)
# ========================================================

def rq_decomposition(matrix: np.ndarray) -> tuple:
    """執行 3x3 矩陣的 RQ 分解以取得 K 與 R。"""
    # 使用 Gram-Schmidt 正交化或其他方法
    # 這裡使用 numpy 的簡單實作
    Q, R = np.linalg.qr(np.flipud(matrix).T)
    R = np.flipud(R.T)
    Q = np.flipud(Q.T)
    
    # 修正 R 的對角線符號，確保 K 的對角線為正
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1
            Q[i, :] *= -1
            
    return R, Q

def get_camera_matrices(config_path: str, dataset_name: str) -> dict:
    """從設定檔讀取相機矩陣"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        if dataset_name in configs:
            return configs[dataset_name]
        
        # 嘗試模糊匹配
        for key in configs:
            if key in dataset_name or dataset_name in key:
                return configs[key]
                
        return None
    except Exception as e:
        print(f"讀取相機設定失敗: {e}")
        return None

def transform_to_camera_frame(data: list, P: np.ndarray) -> list:
    """
    將 3D 軌跡變換到相機座標系
    X_cam = K^(-1) * P * X_world
    """
    # 分解 P = K[R|t]
    M = P[:, :3]
    K, R = rq_decomposition(M)
    K_inv = np.linalg.inv(K)
    
    transformed_data = []
    for frame in data:
        new_frame = frame.copy()
        for kp in KEYPOINT_NAMES_EN:
            point = get_keypoint_safely(frame, kp)
            if point is not None:
                # 齊次座標
                X_world = np.append(point, 1.0)
                # 投影到相機座標系: P * X_world = K * X_cam => X_cam = K_inv * P * X_world
                X_cam = K_inv @ (P @ X_world)
                
                # 更新座標
                new_frame[kp] = {
                    'x': float(X_cam[0]),
                    'y': float(X_cam[1]),
                    'z': float(X_cam[2]) # 這是深度
                }
        transformed_data.append(new_frame)
    return transformed_data


# ========================================================
# 核心分析函數
# ========================================================

def safe_percentage(count: float, total: float) -> float:
    """避免零除的比例計算。"""
    return float(count / total * 100) if total else 0.0

def analyze_depth_ranges(data: list, config: ValidationConfig) -> dict:
    """
    分析各關鍵點的深度範圍
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 深度範圍分析結果
    """
    depth_data = {kp: [] for kp in KEYPOINT_NAMES_EN}
    
    for frame in data:
        for kp in KEYPOINT_NAMES_EN:
            point = get_keypoint_safely(frame, kp)
            if point is not None:
                depth_data[kp].append(point[2])  # Z 值
    
    depth_stats = {}
    for kp, depths in depth_data.items():
        if not depths:
            continue
        
        arr = np.array(depths, dtype=float)
        mean_z = float(np.mean(arr))
        std_z = float(np.std(arr))
        cv_z = calculate_cv(arr)
        depth_range = float(np.max(arr) - np.min(arr))
        
        # 深度梯度（新增）
        if len(arr) >= 2:
            gradients = np.abs(np.diff(arr))
            mean_gradient = float(np.mean(gradients))
            max_gradient = float(np.max(gradients))
        else:
            mean_gradient = 0.0
            max_gradient = 0.0
        
        # 異常值檢測
        outlier_indices, outlier_stats = detect_outliers_zscore(arr, config.depth_outlier_sigma)
        
        # 深度合理性檢查
        in_valid_range = np.sum((arr >= config.depth_min_mm) & (arr <= config.depth_max_mm))
        valid_rate = float(in_valid_range / len(arr) * 100)
        
        depth_stats[kp] = {
            'sample_count': len(arr),
            'mean_z_mm': mean_z,
            'std_z_mm': std_z,
            'cv_percent': cv_z,
            'min_z_mm': float(np.min(arr)),
            'max_z_mm': float(np.max(arr)),
            'depth_range_mm': depth_range,
            'mean_gradient_mm': mean_gradient,
            'max_gradient_mm': max_gradient,
            'quality_level': config.get_quality_level_cv(cv_z),
            'valid_range_rate': valid_rate,
            'outlier_count': outlier_stats['outlier_count'],
            'outlier_rate': outlier_stats['outlier_rate']
        }
    
    return depth_stats


def analyze_depth_symmetry(data: list, config: ValidationConfig) -> list:
    """
    分析深度對稱性（左右關節深度差異）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        list: 深度對稱性分析結果
    """
    symmetry_pairs = [
        ('left_shoulder', 'right_shoulder', '肩膀'),
        ('left_hip', 'right_hip', '髖部'),
        ('left_knee', 'right_knee', '膝蓋'),
        ('left_ankle', 'right_ankle', '腳踝'),
        ('left_wrist', 'right_wrist', '手腕'),
    ]
    
    symmetry_results = []
    
    for left_kp, right_kp, zh_name in symmetry_pairs:
        left_depths = []
        right_depths = []
        
        for frame in data:
            left_point = get_keypoint_safely(frame, left_kp)
            right_point = get_keypoint_safely(frame, right_kp)
            
            if left_point is not None and right_point is not None:
                left_depths.append(left_point[2])
                right_depths.append(right_point[2])
        
        if not left_depths:
            continue
        
        left_arr = np.array(left_depths, dtype=float)
        right_arr = np.array(right_depths, dtype=float)
        
        diffs = np.abs(left_arr - right_arr)
        mean_diff = float(np.mean(diffs))
        max_diff = float(np.max(diffs))
        
        # 評估對稱性品質
        # 假設: 深度差異在 50mm 內為優，100mm 內為可接受，超過為差
        if mean_diff <= 50:
            assessment = "Good"
        elif mean_diff <= 100:
            assessment = "Acceptable"
        else:
            assessment = "Poor"

        symmetry_results.append({
            'pair_name': zh_name,
            'sample_count': len(diffs),
            'mean_depth_diff_mm': mean_diff,
            'std_depth_diff_mm': float(np.std(diffs)),
            'max_depth_diff_mm': max_diff,
            'median_depth_diff_mm': float(np.median(diffs)),
            'assessment': assessment
        })
    
    return symmetry_results


def analyze_depth_logic(data: list, config: ValidationConfig) -> dict:
    """
    分析深度邏輯合理性（肢體遠近關係）
    
    參數:
        data: 3D 軌跡數據
        config: 驗證配置
    
    返回:
        dict: 深度邏輯分析結果
    """
    violations = {
        'wrist_behind_shoulder': 0,
        'knee_behind_hip': 0,
    }

    wrist_checks = 0
    knee_checks = 0
    
    # 收集詳細異常信息
    wrist_anomalies = []
    knee_anomalies = []
    
    for frame_idx, frame in enumerate(data):
        # 檢查手腕是否合理地在肩膀前後
        left_shoulder = get_keypoint_safely(frame, "left_shoulder")
        left_wrist = get_keypoint_safely(frame, "left_wrist")
        
        if left_shoulder is not None and left_wrist is not None:
            wrist_checks += 1
            depth_diff = left_shoulder[2] - left_wrist[2]
            # 使用配置的容差值
            if left_wrist[2] < left_shoulder[2] - config.wrist_depth_tolerance:
                violations['wrist_behind_shoulder'] += 1
                wrist_anomalies.append({
                    'frame': frame_idx,
                    'side': 'left',
                    'wrist_z': round(left_wrist[2], 2),
                    'shoulder_z': round(left_shoulder[2], 2),
                    'depth_diff': round(depth_diff, 2),
                    'severity': 'severe' if depth_diff > config.wrist_depth_tolerance * 3 else 
                               'moderate' if depth_diff > config.wrist_depth_tolerance * 2 else 'mild'
                })
        
        right_shoulder = get_keypoint_safely(frame, "right_shoulder")
        right_wrist = get_keypoint_safely(frame, "right_wrist")
        
        if right_shoulder is not None and right_wrist is not None:
            wrist_checks += 1
            depth_diff = right_shoulder[2] - right_wrist[2]
            if right_wrist[2] < right_shoulder[2] - config.wrist_depth_tolerance:
                violations['wrist_behind_shoulder'] += 1
                wrist_anomalies.append({
                    'frame': frame_idx,
                    'side': 'right',
                    'wrist_z': round(right_wrist[2], 2),
                    'shoulder_z': round(right_shoulder[2], 2),
                    'depth_diff': round(depth_diff, 2),
                    'severity': 'severe' if depth_diff > config.wrist_depth_tolerance * 3 else 
                               'moderate' if depth_diff > config.wrist_depth_tolerance * 2 else 'mild'
                })
        
        # 檢查膝蓋是否合理地在髖部前後
        left_hip = get_keypoint_safely(frame, "left_hip")
        left_knee = get_keypoint_safely(frame, "left_knee")
        
        if left_hip is not None and left_knee is not None:
            knee_checks += 1
            depth_diff = left_hip[2] - left_knee[2]
            # 使用配置的容差值
            if left_knee[2] < left_hip[2] - config.knee_depth_tolerance:
                violations['knee_behind_hip'] += 1
                knee_anomalies.append({
                    'frame': frame_idx,
                    'side': 'left',
                    'knee_z': round(left_knee[2], 2),
                    'hip_z': round(left_hip[2], 2),
                    'depth_diff': round(depth_diff, 2),
                    'severity': 'severe' if depth_diff > config.knee_depth_tolerance * 3 else 
                               'moderate' if depth_diff > config.knee_depth_tolerance * 2 else 'mild'
                })
        
        right_hip = get_keypoint_safely(frame, "right_hip")
        right_knee = get_keypoint_safely(frame, "right_knee")
        
        if right_hip is not None and right_knee is not None:
            knee_checks += 1
            depth_diff = right_hip[2] - right_knee[2]
            if right_knee[2] < right_hip[2] - config.knee_depth_tolerance:
                violations['knee_behind_hip'] += 1
                knee_anomalies.append({
                    'frame': frame_idx,
                    'side': 'right',
                    'knee_z': round(right_knee[2], 2),
                    'hip_z': round(right_hip[2], 2),
                    'depth_diff': round(depth_diff, 2),
                    'severity': 'severe' if depth_diff > config.knee_depth_tolerance * 3 else 
                               'moderate' if depth_diff > config.knee_depth_tolerance * 2 else 'mild'
                })
    
    return {
        'total_frames': len(data),
        'wrist_checks': wrist_checks,
        'wrist_behind_shoulder_count': violations['wrist_behind_shoulder'],
        'wrist_behind_shoulder_rate': safe_percentage(violations['wrist_behind_shoulder'], wrist_checks),
        'wrist_anomalies': wrist_anomalies,  # 新增
        'knee_checks': knee_checks,
        'knee_behind_hip_count': violations['knee_behind_hip'],
        'knee_behind_hip_rate': safe_percentage(violations['knee_behind_hip'], knee_checks),
        'knee_anomalies': knee_anomalies  # 新增
    }


def analyze_depth_statistics(depth_stats: dict) -> dict:
    """
    分析 Z 軸統計特性（新增）
    
    參數:
        depth_stats: 深度統計結果
    
    返回:
        dict: Z 軸統計特性分析
    """
    all_means = [s['mean_z_mm'] for s in depth_stats.values()]
    all_cvs = [s['cv_percent'] for s in depth_stats.values()]
    all_ranges = [s['depth_range_mm'] for s in depth_stats.values()]
    
    if not all_means:
        return {}
    
    return {
        'overall_mean_depth_mm': float(np.mean(all_means)),
        'overall_std_depth_mm': float(np.std(all_means)),
        'average_cv_percent': float(np.mean(all_cvs)),
        'average_range_mm': float(np.mean(all_ranges)),
        'max_range_mm': float(np.max(all_ranges)),
        'min_mean_depth_mm': float(np.min(all_means)),
        'max_mean_depth_mm': float(np.max(all_means))
    }


def analyze_temporal_depth_stability(data: list, config: ValidationConfig) -> dict:
    """
    分析時間軸上的深度穩定性 (偵測整體深度跳動)
    """
    frame_diffs = []
    
    for i in range(1, len(data)):
        curr_frame = data[i]
        prev_frame = data[i-1]
        
        diff_sum = 0
        valid_points = 0
        
        for kp in KEYPOINT_NAMES_EN:
            p1 = get_keypoint_safely(prev_frame, kp)
            p2 = get_keypoint_safely(curr_frame, kp)
            
            if p1 is not None and p2 is not None:
                diff_sum += abs(p2[2] - p1[2])
                valid_points += 1
        
        avg_diff = diff_sum / valid_points if valid_points > 0 else 0
        frame_diffs.append({
            'frame_index': i,
            'avg_depth_change_mm': avg_diff
        })
    
    # 統計
    changes = [x['avg_depth_change_mm'] for x in frame_diffs]
    if not changes:
        return {'unstable_frames': [], 'max_change_mm': 0, 'mean_change_mm': 0}
        
    mean_change = float(np.mean(changes))
    std_change = float(np.std(changes))
    threshold = mean_change + 3 * std_change
    
    unstable_frames = [x for x in frame_diffs if x['avg_depth_change_mm'] > threshold]
    
    return {
        'mean_frame_change_mm': mean_change,
        'max_frame_change_mm': float(np.max(changes)),
        'unstable_frames': unstable_frames, # Frames with sudden global depth shift
        'threshold_mm': threshold
    }


def calculate_depth_quality_score(depth_stats: dict, logic_stats: dict, overall_stats: dict) -> dict:
    """計算整體深度品質分數 (0-100)"""
    score = 100.0
    deductions = []
    
    # 1. CV 扣分 (穩定性)
    avg_cv = overall_stats.get('average_cv_percent', 0)
    if avg_cv > 10:
        deduct = (avg_cv - 10) * 2
        score -= deduct
        deductions.append(f"平均 CV 過高 ({avg_cv:.1f}%) -{deduct:.1f}")
    elif avg_cv > 5:
        deduct = (avg_cv - 5) * 1
        score -= deduct
        deductions.append(f"平均 CV 略高 ({avg_cv:.1f}%) -{deduct:.1f}")
        
    # 2. 邏輯異常扣分
    wrist_rate = logic_stats.get('wrist_behind_shoulder_rate', 0)
    knee_rate = logic_stats.get('knee_behind_hip_rate', 0)
    
    if wrist_rate > 0:
        deduct = min(20, wrist_rate * 0.5)
        score -= deduct
        deductions.append(f"手腕深度異常 ({wrist_rate:.1f}%) -{deduct:.1f}")
        
    if knee_rate > 0:
        deduct = min(20, knee_rate * 0.5)
        score -= deduct
        deductions.append(f"膝蓋深度異常 ({knee_rate:.1f}%) -{deduct:.1f}")
        
    # 3. 異常值扣分
    total_outliers = sum(s['outlier_count'] for s in depth_stats.values())
    total_samples = sum(s['sample_count'] for s in depth_stats.values())
    outlier_rate = (total_outliers / total_samples * 100) if total_samples else 0
    
    if outlier_rate > 1:
        deduct = min(20, outlier_rate * 2)
        score -= deduct
        deductions.append(f"深度異常值比例 ({outlier_rate:.1f}%) -{deduct:.1f}")

    return {
        'score': max(0, min(100, score)),
        'deductions': deductions,
        'level': 'Excellent' if score >= 90 else 'Good' if score >= 80 else 'Acceptable' if score >= 60 else 'Poor'
    }


def print_analysis_report(
    depth_stats: dict,
    symmetry_results: list,
    logic_stats: dict,
    overall_stats: dict,
    config: ValidationConfig
) -> None:
    """列印分析報告"""
    
    print("\n" + "=" * 100)
    print("【1. 深度範圍與穩定性分析】")
    print("=" * 100)
    print(f"{'關鍵點':<15} {'樣本數':<8} {'平均Z(mm)':<12} {'CV(%)':<10} {'範圍(mm)':<12} {'品質':<10} {'合法%':>8}")
    print("-" * 100)
    
    for kp, stats in depth_stats.items():
        zh_name = get_keypoint_name_zh(kp)
        print(f"{zh_name:<15} {stats['sample_count']:<8d} {stats['mean_z_mm']:>10.2f}  "
            f"{stats['cv_percent']:>8.2f}  {stats['depth_range_mm']:>10.2f}  {stats['quality_level']:<10} "
            f"{stats['valid_range_rate']:>7.1f}%")
    
    if symmetry_results:
        print("\n" + "=" * 100)
        print("【2. 深度對稱性檢查】")
        print("=" * 100)
        for result in symmetry_results:
            print(f"{result['pair_name']:<10} 平均深度差:{result['mean_depth_diff_mm']:>8.2f}mm "
                  f"最大:{result['max_depth_diff_mm']:>8.2f}mm")
    
    if logic_stats:
        print("\n" + "=" * 100)
        print("【3. 深度邏輯合理性檢查】")
        print("=" * 100)
        wrist_checks = logic_stats.get('wrist_checks', 0)
        knee_checks = logic_stats.get('knee_checks', 0)
        print(f"手腕異常後置: {logic_stats['wrist_behind_shoulder_count']}/{wrist_checks} "
            f"({logic_stats['wrist_behind_shoulder_rate']:.1f}%)")
        print(f"膝蓋異常後置: {logic_stats['knee_behind_hip_count']}/{knee_checks} "
            f"({logic_stats['knee_behind_hip_rate']:.1f}%)")
        
        # 智能顯示手腕異常詳情
        wrist_anomalies = logic_stats.get('wrist_anomalies', [])
        if wrist_anomalies:
            # 按嚴重度分類
            severe = [a for a in wrist_anomalies if a['severity'] == 'severe']
            moderate = [a for a in wrist_anomalies if a['severity'] == 'moderate']
            mild = [a for a in wrist_anomalies if a['severity'] == 'mild']
            
            tolerance = config.wrist_depth_tolerance
            print(f"\n  手腕異常分類:")
            print(f"  • 嚴重 (深度差>{tolerance*3:.0f}mm): {len(severe):3d} 個 ({len(severe)/len(wrist_anomalies)*100:5.1f}%)")
            print(f"  • 中度 ({tolerance*2:.0f}-{tolerance*3:.0f}mm): {len(moderate):3d} 個 ({len(moderate)/len(wrist_anomalies)*100:5.1f}%)")
            print(f"  • 輕度 ({tolerance:.0f}-{tolerance*2:.0f}mm):   {len(mild):3d} 個 ({len(mild)/len(wrist_anomalies)*100:5.1f}%)")
            
            # 嚴重異常：全部顯示
            if severe:
                print(f"\n[!] 嚴重手腕異常 (深度差>{tolerance*3:.0f}mm) - 全部 {len(severe)} 個:")
                for a in severe:
                    print(f"      Frame {a['frame']:3d} | {a['side']:5s} | 手腕Z={a['wrist_z']:7.1f}mm, 肩膀Z={a['shoulder_z']:7.1f}mm, 深度差={a['depth_diff']:6.1f}mm")
            
            # 中度異常：顯示前10個
            if moderate:
                show_count = min(10, len(moderate))
                print(f"\n  中度手腕異常 ({tolerance*2:.0f}-{tolerance*3:.0f}mm) - 顯示前 {show_count}/{len(moderate)} 個:")
                for a in moderate[:show_count]:
                    print(f"      Frame {a['frame']:3d} | {a['side']:5s} | 手腕Z={a['wrist_z']:7.1f}mm, 肩膀Z={a['shoulder_z']:7.1f}mm, 深度差={a['depth_diff']:6.1f}mm")
                if len(moderate) > 10:
                    print(f"      ... 還有 {len(moderate)-10} 個中度異常")
            
            # 輕度異常：僅統計
            if mild:
                print(f"\n  輕度手腕異常 ({tolerance:.0f}-{tolerance*2:.0f}mm): {len(mild)} 個（詳情見 JSON）")
        
        # 智能顯示膝蓋異常詳情
        knee_anomalies = logic_stats.get('knee_anomalies', [])
        if knee_anomalies:
            # 按嚴重度分類
            severe = [a for a in knee_anomalies if a['severity'] == 'severe']
            moderate = [a for a in knee_anomalies if a['severity'] == 'moderate']
            mild = [a for a in knee_anomalies if a['severity'] == 'mild']
            
            tolerance = config.knee_depth_tolerance
            print(f"\n  膝蓋異常分類:")
            print(f"  • 嚴重 (深度差>{tolerance*3:.0f}mm): {len(severe):3d} 個 ({len(severe)/len(knee_anomalies)*100:5.1f}%)")
            print(f"  • 中度 ({tolerance*2:.0f}-{tolerance*3:.0f}mm): {len(moderate):3d} 個 ({len(moderate)/len(knee_anomalies)*100:5.1f}%)")
            print(f"  • 輕度 ({tolerance:.0f}-{tolerance*2:.0f}mm):   {len(mild):3d} 個 ({len(mild)/len(knee_anomalies)*100:5.1f}%)")
            
            # 嚴重異常：全部顯示
            if severe:
                print(f"\n[!] 嚴重膝蓋異常 (深度差>{tolerance*3:.0f}mm) - 全部 {len(severe)} 個:")
                for a in severe:
                    print(f"      Frame {a['frame']:3d} | {a['side']:5s} | 膝蓋Z={a['knee_z']:7.1f}mm, 髖部Z={a['hip_z']:7.1f}mm, 深度差={a['depth_diff']:6.1f}mm")
            
            # 中度異常：顯示前10個
            if moderate:
                show_count = min(10, len(moderate))
                print(f"\n  中度膝蓋異常 ({tolerance*2:.0f}-{tolerance*3:.0f}mm) - 顯示前 {show_count}/{len(moderate)} 個:")
                for a in moderate[:show_count]:
                    print(f"      Frame {a['frame']:3d} | {a['side']:5s} | 膝蓋Z={a['knee_z']:7.1f}mm, 髖部Z={a['hip_z']:7.1f}mm, 深度差={a['depth_diff']:6.1f}mm")
                if len(moderate) > 10:
                    print(f"      ... 還有 {len(moderate)-10} 個中度異常")
            
            # 輕度異常：僅統計
            if mild:
                print(f"\n  輕度膝蓋異常 ({tolerance:.0f}-{tolerance*2:.0f}mm): {len(mild)} 個（詳情見 JSON）")
        
        # JSON 儲存提示
        if wrist_anomalies or knee_anomalies:
            print(f"\n  [V] 完整異常詳情已儲存至 JSON:")
            if wrist_anomalies:
                print(f"     • logic_results.wrist_anomalies - 所有 {len(wrist_anomalies)} 個手腕異常")
            if knee_anomalies:
                print(f"     • logic_results.knee_anomalies - 所有 {len(knee_anomalies)} 個膝蓋異常")
    
    if overall_stats:
        print("\n" + "=" * 100)
        print("【4. 整體深度統計特性】")
        print("=" * 100)
        print(f"平均深度: {overall_stats['overall_mean_depth_mm']:.2f} mm")
        print(f"平均 CV: {overall_stats['average_cv_percent']:.2f}%")
        print(f"平均深度範圍: {overall_stats['average_range_mm']:.2f} mm")


def validate_depth_reasonableness_analysis(
    json_3d_path: str,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    深度合理性驗證分析（主函數）
    支援多視角分析 (Global, 45, Side)
    
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
    
    # 嘗試載入相機參數以進行多視角分析
    views = {'Global': data}
    
    # 尋找 camera_configs.json
    cam_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(json_3d_path))), 'camera_configs.json')
    if not os.path.exists(cam_config_path):
        # 嘗試往上找
        current_dir = os.path.dirname(os.path.abspath(json_3d_path))
        while len(current_dir) > 3:
            p = os.path.join(current_dir, 'camera_configs.json')
            if os.path.exists(p):
                cam_config_path = p
                break
            current_dir = os.path.dirname(current_dir)
            
    if os.path.exists(cam_config_path):
        # 推斷 dataset_name (使用父資料夾名稱)
        dataset_name = os.path.basename(os.path.dirname(json_3d_path))
        print(f"嘗試載入相機設定 (Dataset: {dataset_name})...")
        
        cam_matrices = get_camera_matrices(cam_config_path, dataset_name)
        if cam_matrices:
            print("成功載入相機矩陣，將進行多視角分析。")
            if 'p1' in cam_matrices:
                P1 = np.array(cam_matrices['p1'])
                views['45'] = transform_to_camera_frame(data, P1)
            if 'p2' in cam_matrices:
                P2 = np.array(cam_matrices['p2'])
                views['Side'] = transform_to_camera_frame(data, P2)
        else:
            print(f"未找到 {dataset_name} 的相機設定，僅執行 Global 分析。")
    
    full_results = {}
    
    for view_name, view_data in views.items():
        print(f"\n[{view_name} View Analysis]")
        
        # 執行各項分析
        depth_stats = analyze_depth_ranges(view_data, config)
        symmetry_results = analyze_depth_symmetry(view_data, config)
        logic_stats = analyze_depth_logic(view_data, config)
        overall_stats = analyze_depth_statistics(depth_stats)
        temporal_stats = analyze_temporal_depth_stability(view_data, config)
        quality_score = calculate_depth_quality_score(depth_stats, logic_stats, overall_stats)
        
        # 僅對 Global 視角列印詳細報告，避免洗版
        if view_name == 'Global':
            print_analysis_report(
                depth_stats, symmetry_results,
                logic_stats, overall_stats, config
            )
        
        # 整合該視角的結果
        full_results[view_name] = {
            "metadata": {
                "analysis_time": datetime.now().isoformat(),
                "source_file": str(json_3d_path),
                "total_frames": int(len(view_data)),
                "analysis_type": "Depth Reasonableness Analysis",
                "view": view_name
            },
            "overall_summary": {
                "average_cv": overall_stats.get('average_cv_percent', 0.0),
                "quality_level": config.get_quality_level_cv(overall_stats.get('average_cv_percent', 0.0)),
                "total_keypoints_analyzed": len(depth_stats),
                "total_logic_violations": (logic_stats.get('wrist_behind_shoulder_count', 0) + 
                                          logic_stats.get('knee_behind_hip_count', 0)),
                "mean_depth_mm": overall_stats.get('overall_mean_depth_mm', 0.0),
                "quality_score": quality_score
            },
            "depth_range_analysis": depth_stats,
            "depth_symmetry": symmetry_results,
            "depth_logic_check": logic_stats,
            "overall_statistics": overall_stats,
            "temporal_stability": temporal_stats
        }

    # 保存結果
    if output_json_path is None:
        # 建立 results 資料夾
        results_dir = os.path.join(os.path.dirname(json_3d_path), 'Verification Result')
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d_path))[0]
        output_json_path = os.path.join(results_dir, f"{base_name}_step3_depth_reasonableness_results.json")
    
    save_json_results(full_results, output_json_path)
    print(f"\n[OK] 結果已儲存至: {output_json_path}")
    
    return full_results


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
        print("  python step3_depth_reasonableness_v2.py <json_path> [--config <config>] [--output <output>]")
    
    try:
        results = validate_depth_reasonableness_analysis(
            json_3d_path,
            output_json_path,
            config_path
        )
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
