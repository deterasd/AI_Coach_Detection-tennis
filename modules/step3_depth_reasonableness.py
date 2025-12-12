"""
Step 3: 深度合理性驗證分析
驗證 3D 重建結果的深度（Z 軸）合理性和一致性

功能：
  1. 深度範圍檢查（有效深度區間）
  2. 深度變異係數分析
  3. 深度跳動檢測
  4. 深度邏輯檢查（肢體遠近關係）
  5. 深度梯度分析
  6. Z 軸統計特性分析
"""

import numpy as np
from datetime import datetime
import sys
import os
import json

# 修正路徑問題，確保能夠正確引入模組
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入共用模組
from modules.utils import (
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


def calculate_bone_length(p1, p2):
    """計算兩點間的歐幾里得距離"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_vector_angle(v1, v2):
    """計算兩個向量的夾角 (度)"""
    # v1 dot v2 = |v1| |v2| cos(theta)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    # 限制範圍避免數值誤差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def analyze_depth_logic(data: list, config: ValidationConfig) -> dict:
    """
    分析深度邏輯合理性 (改進版)
    重點檢查：
    1. 骨骼長度異常 (Bone Length) - 檢測深度估計導致的肢體伸縮
    2. 關節反向彎曲 (Joint Hyperextension) - 檢測手肘/膝蓋反折
    """
    # 初始化結果結構
    checks = {
        'elbow_hyperextension': {'total': 0, 'left': 0, 'right': 0},
        'knee_hyperextension': {'total': 0, 'left': 0, 'right': 0}
    }
    violations = {
        'elbow_hyperextension': {'total': 0, 'left': 0, 'right': 0},
        'knee_hyperextension': {'total': 0, 'left': 0, 'right': 0}
    }
    anomalies = {
        'elbow_hyperextension': [],
        'knee_hyperextension': []
    }

    # 2. 逐幀檢查
    for frame_idx, frame in enumerate(data):
        
        # --- B. 手肘反向彎曲 (Hyperextension) ---
        # 這裡簡化判斷：如果手肘深度異常地比肩膀和手腕連線還深很多 (反折)
        for side in ['left', 'right']:
            shoulder = get_keypoint_safely(frame, f"{side}_shoulder")
            elbow = get_keypoint_safely(frame, f"{side}_elbow")
            wrist = get_keypoint_safely(frame, f"{side}_wrist")
            
            if shoulder is not None and elbow is not None and wrist is not None:
                checks['elbow_hyperextension']['total'] += 1
                checks['elbow_hyperextension'][side] += 1
                
                # 簡單幾何：如果手肘 Z 值比 肩膀和手腕連線都大很多 (凹陷進去)
                # 假設相機在正前方，Z越大越遠
                avg_z = (shoulder[2] + wrist[2]) / 2
                # 容忍度 200mm
                if elbow[2] > avg_z + 200: 
                    violations['elbow_hyperextension']['total'] += 1
                    violations['elbow_hyperextension'][side] += 1
                    anomalies['elbow_hyperextension'].append({
                        'frame': frame_idx,
                        'side': side,
                        'val_z': round(elbow[2], 1),
                        'ref_z': round(avg_z, 1),
                        'severity': 'moderate'
                    })

        # --- C. 膝蓋反向彎曲 ---
        # 站立時，膝蓋不應該比「臀部與腳踝連線」更靠後太多 (反膝)
        for side in ['left', 'right']:
            hip = get_keypoint_safely(frame, f"{side}_hip")
            knee = get_keypoint_safely(frame, f"{side}_knee")
            ankle = get_keypoint_safely(frame, f"{side}_ankle")
            
            if hip is not None and knee is not None and ankle is not None:
                checks['knee_hyperextension']['total'] += 1
                checks['knee_hyperextension'][side] += 1
                
                # 簡單幾何：如果膝蓋 Z 值比 臀部和腳踝都大很多 (凹陷進去)
                avg_z = (hip[2] + ankle[2]) / 2
                if knee[2] > avg_z + 200: # 膝蓋比連線深 20公分 (反折)
                    violations['knee_hyperextension']['total'] += 1
                    violations['knee_hyperextension'][side] += 1
                    anomalies['knee_hyperextension'].append({
                        'frame': frame_idx,
                        'side': side,
                        'val_z': round(knee[2], 1),
                        'ref_z': round(avg_z, 1),
                        'severity': 'moderate'
                    })

    # 整理結果
    result = {}
    for key in checks.keys():
        result[key] = {
            'checks': checks[key],
            'violations': violations[key],
            'rate': safe_percentage(violations[key]['total'], checks[key]['total']),
            'anomalies': anomalies[key]
        }
        
    return result


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
    changes = []
    
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
        changes.append(avg_diff)
        frame_diffs.append({
            'frame_index': i,
            'avg_depth_change_mm': avg_diff
        })
    
    if not changes:
        return {
            'unstable_frames': [], 
            'max_change_mm': 0, 
            'mean_change_mm': 0,
            'series_data': []
        }
        
    mean_change = float(np.mean(changes))
    std_change = float(np.std(changes))
    # 動態閾值：平均值 + 3個標準差，且至少要大於 50mm 才會被視為異常跳動
    threshold = max(mean_change + 3 * std_change, 50.0)
    
    unstable_frames = [x for x in frame_diffs if x['avg_depth_change_mm'] > threshold]
    
    return {
        'mean_frame_change_mm': mean_change,
        'max_frame_change_mm': float(np.max(changes)),
        'unstable_frames': unstable_frames,
        'unstable_frame_count': len(unstable_frames),
        'unstable_rate': safe_percentage(len(unstable_frames), len(data)),
        'threshold_mm': threshold,
        'series_data': changes  # 用於前端繪製折線圖
    }


def calculate_depth_quality_score(
    depth_stats: dict, 
    logic_stats: dict, 
    overall_stats: dict,
    temporal_stats: dict
) -> dict:
    """
    計算整體深度品質分數 (0-100) - 全面性評分版
    權重分配：
    1. 穩定性 (CV) - 30%
    2. 邏輯合理性 (Logic) - 30%
    3. 時間連續性 (Temporal) - 20%
    4. 物理限制 (Velocity/Outliers) - 20%
    """
    score = 100.0
    deductions = []
    
    # 1. 穩定性扣分 (CV) - 反映數據的雜訊程度
    avg_cv = overall_stats.get('average_cv_percent', 0)
    if avg_cv > 15:
        deduct = 30
        deductions.append({"reason": f"深度極度不穩定 (CV {avg_cv:.1f}%)", "points": deduct})
    elif avg_cv > 10:
        deduct = 20 + (avg_cv - 10) * 2
        deductions.append({"reason": f"深度不穩定 (CV {avg_cv:.1f}%)", "points": deduct})
    elif avg_cv > 5:
        deduct = (avg_cv - 5) * 2
        deductions.append({"reason": f"深度略有雜訊 (CV {avg_cv:.1f}%)", "points": deduct})
    score -= min(30, sum(d['points'] for d in deductions if 'CV' in d['reason']))

    # 2. 邏輯異常扣分 (Logic) - 反映重建結構錯誤
    logic_deduct = 0
    
    # 遍歷所有邏輯檢查項目
    for check_name, result in logic_stats.items():
        if not isinstance(result, dict) or 'rate' not in result:
            continue
            
        rate = result['rate']
        if rate > 0:
            # 根據不同項目給予不同權重
            weight = 1.0
            label = check_name
            if check_name == 'elbow_hyperextension':
                label = "手肘反向彎曲"
            elif check_name == 'knee_hyperextension':
                label = "膝蓋反向彎曲"
                
            d = min(15, rate * weight) 
            logic_deduct += d
            deductions.append({"reason": f"{label} ({rate:.1f}%)", "points": d})
    
    score -= min(30, logic_deduct)

    # 3. 時間連續性扣分 (Temporal) - 反映畫面閃爍/跳動
    unstable_rate = temporal_stats.get('unstable_rate', 0)
    max_jump = temporal_stats.get('max_frame_change_mm', 0)
    
    temp_deduct = 0
    if unstable_rate > 0:
        d = min(20, unstable_rate * 5.0) # 1% 跳動幀扣 5 分 (跳動很嚴重)
        temp_deduct += d
        deductions.append({"reason": f"深度瞬間跳動 ({unstable_rate:.1f}% 幀)", "points": d})
    
    if max_jump > 200: # 瞬間移動超過 20公分
        d = 10
        temp_deduct += d
        deductions.append({"reason": f"存在劇烈深度突變 (最大 {max_jump:.0f}mm)", "points": d})
        
    score -= min(20, temp_deduct)

    # 4. 物理限制扣分 (Velocity/Outliers)
    # 檢查是否有關鍵點移動速度過快 (Z-Velocity)
    high_velocity_kps = 0
    for kp, stat in depth_stats.items():
        # 假設 100mm/frame 是非常快的 Z 軸移動 (約 3m/s @ 30fps)
        if stat.get('mean_gradient_mm', 0) > 100:
            high_velocity_kps += 1
            
    phy_deduct = 0
    if high_velocity_kps > 0:
        d = min(15, high_velocity_kps * 3)
        phy_deduct += d
        deductions.append({"reason": f"{high_velocity_kps} 個部位 Z 軸移動過快", "points": d})

    # 異常值比例
    total_outliers = sum(s['outlier_count'] for s in depth_stats.values())
    total_samples = sum(s['sample_count'] for s in depth_stats.values())
    outlier_rate = (total_outliers / total_samples * 100) if total_samples else 0
    
    if outlier_rate > 2:
        d = min(10, (outlier_rate - 2) * 2)
        phy_deduct += d
        deductions.append({"reason": f"深度離群值過多 ({outlier_rate:.1f}%)", "points": d})
        
    score -= min(20, phy_deduct)

    # 確保分數範圍
    final_score = max(0.0, min(100.0, score))
    
    return {
        'score': final_score,
        'deductions': deductions,
        'level': 'Excellent' if final_score >= 90 else 'Good' if final_score >= 80 else 'Acceptable' if final_score >= 60 else 'Poor'
    }


def print_analysis_report(
    depth_stats: dict,
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
    
    if logic_stats:
        print("\n" + "=" * 100)
        print("【2. 深度邏輯合理性檢查】")
        print("=" * 100)
        
        for check_name, result in logic_stats.items():
            if not isinstance(result, dict): continue
            
            label = check_name
            if check_name == 'elbow_hyperextension': label = "手肘反向彎曲"
            elif check_name == 'knee_hyperextension': label = "膝蓋反向彎曲"
            
            total_checks = result['checks']['total']
            total_violations = result['violations']['total']
            rate = result['rate']
            
            print(f"{label:<15}: {total_violations}/{total_checks} ({rate:.1f}%) "
                  f"[L: {result['violations']['left']} | R: {result['violations']['right']}]")
            
            # 顯示嚴重異常
            anomalies = result.get('anomalies', [])
            severe = [a for a in anomalies if a['severity'] == 'severe']
            if severe:
                print(f"  [!] 發現 {len(severe)} 個嚴重異常")
    
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
        logic_stats = analyze_depth_logic(view_data, config)
        overall_stats = analyze_depth_statistics(depth_stats)
        temporal_stats = analyze_temporal_depth_stability(view_data, config)
        quality_score = calculate_depth_quality_score(depth_stats, logic_stats, overall_stats, temporal_stats)
        
        # 僅對 Global 視角列印詳細報告，避免洗版
        if view_name == 'Global':
            print_analysis_report(
                depth_stats,
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
                "total_logic_violations": sum(v['violations']['total'] for v in logic_stats.values() if isinstance(v, dict) and 'violations' in v),
                "mean_depth_mm": overall_stats.get('overall_mean_depth_mm', 0.0),
                "quality_score": quality_score
            },
            "depth_range_analysis": depth_stats,
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
