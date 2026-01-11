"""
Step 1: 重投影誤差驗證分析
驗證 3D 重建結果的重投影精度，包含畸變參數檢查和相機內參驗證

功能：
  1. 2D/3D 重投影誤差計算
  2. 各關節點誤差統計
  3. 異常值檢測
  4. 逐幀誤差分析
  5. 誤差 vs 深度關係分析
  6. 時間穩定性分析
  7. 相機偏差分析
  8. 畸變參數驗證
  9. 相機內參合理性檢查
  10. 2D Confidence 數值提取 (新增)
"""

import numpy as np
from datetime import datetime
import sys
import os
import json  # 確保有這行
from typing import Optional, Tuple

# 修正路徑問題，確保能夠正確引入模組
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 引入共用模組
from modules.utils import (
    get_keypoint_safely,
    get_keypoint_2d,
    is_valid_keypoint,
    load_json_file,
    validate_frame_structure,
    convert_to_serializable,
    save_json_results,
    generate_output_path,
    KEYPOINT_NAMES_EN,
)
from config import load_config, ValidationConfig


# ========================================================
# 新增: Confidence 提取函數
# ========================================================
def extract_confidence_series(json_path):
    """
    讀取原始 2D JSON，提取每個部位的 confidence 序列
    - 人體 17 點（同層）
    - tennis_ball（同層）
    - paddle_*（巢狀）
    """
    try:
        if not json_path or not isinstance(json_path, str):
            return {"frames": [], "series": {}}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            return {"frames": [], "series": {}}

        # 1) 建立 key 清單：人體點 + tennis_ball + paddle_*
        all_keys = set()
        ignore_keys = ['frame', 'ball', 'court', 'person']  # ✅ 不再忽略 tennis_ball / paddle

        check_frames = data[:30] if len(data) > 30 else data
        for frame in check_frames:
            if not isinstance(frame, dict):
                continue

            # (A) 同層 key：人體17點 + tennis_ball
            for k in frame.keys():
                if k not in ignore_keys and k != "paddle":
                    all_keys.add(k)

            # (B) paddle 巢狀 key：top/right/bottom/left/center
            paddle = frame.get("paddle")
            if isinstance(paddle, dict):
                for part in paddle.keys():
                    all_keys.add(f"paddle_{part}")

        # 初始化
        series = {k: [] for k in sorted(all_keys)}
        frames = []

        # 2) 遍歷每一幀收集 conf
        for frame in data:
            if not isinstance(frame, dict):
                continue

            frames.append(frame.get("frame", len(frames)))

            # (A) 同層：人體17點 + tennis_ball
            for k in series.keys():
                conf = None

                if k.startswith("paddle_"):
                    # (B) paddle 巢狀：paddle_top/right/...
                    part = k.replace("paddle_", "", 1)
                    paddle = frame.get("paddle", {})
                    pv = paddle.get(part, {}) if isinstance(paddle, dict) else {}
                    if isinstance(pv, dict):
                        val = pv.get("conf")
                        if val is not None:
                            conf = round(float(val), 4)
                else:
                    # 同層點（含 tennis_ball）
                    item = frame.get(k, {})
                    if isinstance(item, dict):
                        val = item.get("conf")
                        if val is not None:
                            conf = round(float(val), 4)

                series[k].append(conf)

        return {"frames": frames, "series": series}

    except Exception as e:
        print(f"Error extracting confidence from {json_path}: {e}")
        return {"frames": [], "series": {}}

# ========================================================
# 投影相關函數
# ========================================================
def project_point(P: np.ndarray, X: np.ndarray, distortion: np.ndarray = None) -> np.ndarray:
    """使用投影矩陣 P 將 3D 齊次座標投影到 2D"""
    x = P @ X
    if abs(x[2]) < 1e-6:
        return np.array([np.nan, np.nan])
    
    # 歸一化投影座標
    x_norm = x[:2] / x[2]
    
    # 若有畸變參數，進行畸變校正
    if distortion is not None:
        x_norm = apply_distortion(x_norm, distortion)
    
    return x_norm


def apply_distortion(point: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """
    應用鏡頭畸變模型
    
    參數:
        point: 歸一化座標 [x, y]
        distortion: 畸變參數 [k1, k2, p1, p2, k3]
    
    返回:
        np.ndarray: 畸變後的座標
    """
    if len(distortion) < 5:
        return point
    
    k1, k2, p1, p2, k3 = distortion[:5]
    x, y = point[0], point[1]
    r2 = x**2 + y**2
    
    # 徑向畸變
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    
    # 切向畸變
    x_distorted = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_distorted = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y
    
    return np.array([x_distorted, y_distorted])


def rq_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """執行 3x3 矩陣的 RQ 分解以取得 K 與 R。"""
    if matrix.shape != (3, 3):
        raise ValueError("RQ 分解僅支援 3x3 矩陣")

    m = matrix.astype(float)
    # 利用 QR 分解推導 RQ，避免依賴額外套件
    q, r = np.linalg.qr(np.flipud(m).T)
    r = np.flipud(r.T)
    q = q.T
    r = np.fliplr(r)
    q = np.flipud(q)

    diag = np.sign(np.diag(r))
    diag[diag == 0] = 1
    d = np.diag(diag)
    r = r @ d
    q = d @ q

    return r, q


def extract_intrinsics(P: np.ndarray) -> Optional[np.ndarray]:
    """從投影矩陣中提取內參 K，若失敗則返回 None。"""
    try:
        M = P[:, :3]
        if np.linalg.matrix_rank(M) < 3:
            return None
        K, _ = rq_decomposition(M)
        if abs(K[2, 2]) > 1e-8:
            K = K / K[2, 2]
        return K
    except Exception:
        return None


def safe_ratio(numerator: float, denominator: float) -> float:
    """避免零除的比例計算。"""
    return float(numerator / denominator * 100) if denominator > 0 else 0.0


def analyze_error_depth_relationship(error_records: list) -> Optional[dict]:
    """根據誤差紀錄分析誤差與深度的關聯。"""
    if not error_records:
        return None

    depths = np.array([rec[4] for rec in error_records], dtype=float)
    errors = np.array([rec[3] for rec in error_records], dtype=float)

    if len(depths) < 3 or np.std(depths) < 1e-6:
        return None

    corr = float(np.corrcoef(depths, errors)[0, 1])
    slope, intercept = np.polyfit(depths, errors, 1)

    return {
        "corr": corr,
        "slope": float(slope),
        "intercept": float(intercept),
        "depth_min": float(depths.min()),
        "depth_max": float(depths.max())
    }


def analyze_temporal_stability(per_frame_side: np.ndarray, per_frame_45: np.ndarray, config: ValidationConfig) -> dict:
    """分析逐幀平均誤差的穩定性。"""
    def _analyze(series: np.ndarray) -> dict:
        valid = series[~np.isnan(series)]
        if valid.size == 0:
            return {"samples": 0}

        diffs = np.abs(np.diff(valid))
        spike_threshold = max(config.reprojection_good * 0.5, 1.0)
        spikes = int(np.sum(diffs > spike_threshold))

        return {
            "samples": int(valid.size),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "cv": float(np.std(valid) / np.mean(valid) * 100) if np.mean(valid) > 1e-6 else 0.0,
            "spike_count": spikes,
            "max_jump": float(np.max(diffs)) if diffs.size else 0.0
        }

    return {
        "side": _analyze(per_frame_side),
        "camera_45": _analyze(per_frame_45)
    }


def analyze_detailed_anomalies(error_records: list, config: ValidationConfig) -> dict:
    """
    詳細分析異常誤差的分級、分布和模式
    
    參數:
        error_records: 誤差記錄列表 [(camera, frame, keypoint, error, depth), ...]
        config: 驗證配置
    
    返回:
        dict: 詳細異常分析結果
    """
    if not error_records:
        return {}
    
    # 篩選異常值
    threshold = config.reprojection_outlier_threshold
    anomalies = [rec for rec in error_records if rec[3] > threshold]
    
    if not anomalies:
        return {
            "total_count": 0,
            "by_severity": {"severe": [], "moderate": [], "mild": []},
            "by_keypoint": {},
            "continuous_segments": []
        }
    
    # 嚴重程度分級
    severe_threshold = threshold * 2  # >40px
    moderate_threshold = threshold * 1.5  # 30-40px
    
    severe = [a for a in anomalies if a[3] > severe_threshold]
    moderate = [a for a in anomalies if moderate_threshold < a[3] <= severe_threshold]
    mild = [a for a in anomalies if threshold < a[3] <= moderate_threshold]
    
    # 按關節點統計
    keypoint_stats = {}
    for cam, frame, kp, err, depth in anomalies:
        if kp not in keypoint_stats:
            keypoint_stats[kp] = {"side": 0, "45": 0, "errors": [], "frames": []}
        keypoint_stats[kp]["side" if cam == "side" else "45"] += 1
        keypoint_stats[kp]["errors"].append(float(err))
        keypoint_stats[kp]["frames"].append(int(frame))
    
    # 計算每個關節點的統計
    for kp, stats in keypoint_stats.items():
        stats["count"] = len(stats["errors"])
        stats["mean_error"] = float(np.mean(stats["errors"]))
        stats["max_error"] = float(np.max(stats["errors"]))
        stats["side_ratio"] = stats["side"] / stats["count"] if stats["count"] > 0 else 0
    
    # 檢測連續異常區段
    continuous_segments = []
    anomalies_sorted = sorted(anomalies, key=lambda x: (x[2], x[0], x[1]))  # 按 keypoint, camera, frame
    
    current_segment = None
    for cam, frame, kp, err, depth in anomalies_sorted:
        if current_segment is None:
            current_segment = {
                "keypoint": kp,
                "camera": cam,
                "start_frame": frame,
                "end_frame": frame,
                "errors": [err]
            }
        elif (current_segment["keypoint"] == kp and 
              current_segment["camera"] == cam and 
              frame - current_segment["end_frame"] <= 2):  # 允許 1-2 幀間隔
            current_segment["end_frame"] = frame
            current_segment["errors"].append(err)
        else:
            if len(current_segment["errors"]) >= 5:  # 至少連續 5 幀
                current_segment["duration"] = current_segment["end_frame"] - current_segment["start_frame"] + 1
                current_segment["mean_error"] = float(np.mean(current_segment["errors"]))
                current_segment["max_error"] = float(np.max(current_segment["errors"]))
                continuous_segments.append(current_segment)
            current_segment = {
                "keypoint": kp,
                "camera": cam,
                "start_frame": frame,
                "end_frame": frame,
                "errors": [err]
            }
    
    # 檢查最後一個區段
    if current_segment and len(current_segment["errors"]) >= 5:
        current_segment["duration"] = current_segment["end_frame"] - current_segment["start_frame"] + 1
        current_segment["mean_error"] = float(np.mean(current_segment["errors"]))
        current_segment["max_error"] = float(np.max(current_segment["errors"]))
        continuous_segments.append(current_segment)
    
    # 移除 errors 列表（太大）
    for seg in continuous_segments:
        del seg["errors"]
    
    return {
        "total_count": len(anomalies),
        "by_severity": {
            "severe": severe,
            "moderate": moderate,
            "mild": mild
        },
        "by_keypoint": keypoint_stats,
        "continuous_segments": sorted(continuous_segments, key=lambda x: x["duration"], reverse=True)
    }


def validate_camera_intrinsics(P: np.ndarray, camera_name: str = "Camera") -> dict:
    """
    驗證相機內參的合理性
    
    參數:
        P: 投影矩陣 (3x4)
        camera_name: 相機名稱
    
    返回:
        dict: 驗證結果
    """
    results = {
        "camera_name": camera_name,
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    # 提取內參矩陣 K
    K = extract_intrinsics(P)
    if K is None:
        results["errors"].append("投影矩陣無法分解出有效內參，請檢查輸入 P")
        results["is_valid"] = False
        return results
    
    # 檢查焦距 (fx, fy)
    fx = K[0, 0]
    fy = K[1, 1]
    
    if fx <= 0 or fy <= 0:
        results["errors"].append(f"焦距異常: fx={fx:.2f}, fy={fy:.2f}")
        results["is_valid"] = False
    
    # 檢查焦距比例（通常接近 1）
    if abs(fx / fy - 1.0) > 0.1:
        results["warnings"].append(f"焦距比例異常: fx/fy={fx/fy:.3f}")
    
    # 檢查主點 (cx, cy)
    cx = K[0, 2]
    cy = K[1, 2]
    
    if cx < 0 or cy < 0:
        results["warnings"].append(f"主點座標異常: cx={cx:.2f}, cy={cy:.2f}")
    
    # 檢查非對角元素（傾斜參數，通常接近 0）
    skew = K[0, 1]
    if abs(skew) > 10:
        results["warnings"].append(f"傾斜參數異常: skew={skew:.2f}")
    
    results["intrinsics"] = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "skew": float(skew),
        "aspect_ratio": float(fx / fy) if fy > 0 else 0.0
    }
    
    return results


# ========================================================
# 主要分析函數
# ========================================================

def calculate_reprojection_errors(
    data_3d: list,
    data_2d_side: list,
    data_2d_45: list,
    P1: np.ndarray,
    P2: np.ndarray,
    config: ValidationConfig,
    distortion_side: Optional[np.ndarray] = None,
    distortion_45: Optional[np.ndarray] = None
) -> dict:
    """計算重投影誤差並收集統計資訊。"""

    num_frames = min(len(data_3d), len(data_2d_side), len(data_2d_45))
    keypoints = KEYPOINT_NAMES_EN

    errors_side = {kp: [] for kp in keypoints}
    errors_45 = {kp: [] for kp in keypoints}

    per_frame_error_side = []
    per_frame_error_45 = []
    Z_values = []
    error_records = []

    distortion_side_arr = (
        np.array(distortion_side, dtype=float)
        if distortion_side is not None and len(distortion_side)
        else None
    )
    distortion_45_arr = (
        np.array(distortion_45, dtype=float)
        if distortion_45 is not None and len(distortion_45)
        else None
    )

    for frame_idx in range(num_frames):
        f3d = data_3d[frame_idx]
        f2d_side = data_2d_side[frame_idx]
        f2d_45 = data_2d_45[frame_idx]

        frame_err_side = []
        frame_err_45 = []

        for kp in keypoints:
            point_3d = get_keypoint_safely(f3d, kp)
            if point_3d is None:
                continue

            # 修正：因為輸入的 3D 檔案 Y, Z 軸已被反轉，需轉回原始座標系以配合 P 矩陣
            X_h = np.array([point_3d[0], -point_3d[1], point_3d[2], 1.0], dtype=float)

            if kp == "nose":
                Z_values.append(point_3d[2])

            point_2d_side = get_keypoint_2d(f2d_side, kp)
            if point_2d_side is not None:
                obs = point_2d_side[:2]
                proj = project_point(P1, X_h, distortion_side_arr)
                if not np.any(np.isnan(proj)):
                    err = float(np.linalg.norm(obs - proj))
                    errors_side[kp].append(err)
                    frame_err_side.append(err)
                    error_records.append(("side", frame_idx, kp, err, point_3d[2]))

            point_2d_45 = get_keypoint_2d(f2d_45, kp)
            if point_2d_45 is not None:
                obs = point_2d_45[:2]
                proj = project_point(P2, X_h, distortion_45_arr)
                if not np.any(np.isnan(proj)):
                    err = float(np.linalg.norm(obs - proj))
                    errors_45[kp].append(err)
                    frame_err_45.append(err)
                    error_records.append(("45", frame_idx, kp, err, point_3d[2]))

        per_frame_error_side.append(np.mean(frame_err_side) if frame_err_side else np.nan)
        per_frame_error_45.append(np.mean(frame_err_45) if frame_err_45 else np.nan)

    return {
        'errors_side': errors_side,
        'errors_45': errors_45,
        'per_frame_error_side': np.array(per_frame_error_side, dtype=float),
        'per_frame_error_45': np.array(per_frame_error_45, dtype=float),
        'Z_values': np.array(Z_values, dtype=float),
        'error_records': error_records,
        'num_frames': num_frames,
        'keypoints': keypoints
    }


def analyze_reprojection_results(error_data: dict, config: ValidationConfig) -> dict:
    """
    分析重投影誤差結果
    
    參數:
        error_data: 誤差數據
        config: 驗證配置
    
    返回:
        dict: 分析結果
    """
    errors_side = error_data['errors_side']
    errors_45 = error_data['errors_45']
    keypoints = error_data['keypoints']
    error_records = error_data['error_records']
    
    global_stat = {}
    all_side = []
    all_45 = []

    # 計算各關節點統計
    for kp in keypoints:
        s = np.array(errors_side[kp])
        d = np.array(errors_45[kp])

        mean_s = float(s.mean()) if s.size else 0
        std_s = float(s.std()) if s.size else 0
        mean_d = float(d.mean()) if d.size else 0
        std_d = float(d.std()) if d.size else 0

        if s.size:
            all_side.extend(s)
        if d.size:
            all_45.extend(d)

        global_stat[kp] = {
            "side_mean": mean_s,
            "side_std": std_s,
            "45_mean": mean_d,
            "45_std": std_d,
            "avg_error": (mean_s + mean_d) / 2,
        }

    all_side = np.array(all_side, dtype=float)
    all_45 = np.array(all_45, dtype=float)

    # 全域統計
    global_mean_side = float(all_side.mean()) if all_side.size else 0.0
    global_mean_45 = float(all_45.mean()) if all_45.size else 0.0
    
    # 異常值檢測
    out_s = int(np.sum(all_side > config.reprojection_outlier_threshold))
    out_45 = int(np.sum(all_45 > config.reprojection_outlier_threshold))
    
    # TOP 10 最大誤差
    error_sorted = sorted(error_records, key=lambda r: r[3], reverse=True)
    
    # 品質評估
    quality_level = config.get_quality_level_reprojection(
        (global_mean_side + global_mean_45) / 2
    )
    
    return {
        'global_stat': global_stat,
        'all_side': all_side,
        'all_45': all_45,
        'global_mean_side': global_mean_side,
        'global_mean_45': global_mean_45,
        'outlier_count_side': out_s,
        'outlier_count_45': out_45,
        'error_sorted': error_sorted,
        'quality_level': quality_level
    }


def print_analysis_report(
    error_data: dict,
    analysis: dict,
    config: ValidationConfig,
    depth_relationship: Optional[dict] = None,
    temporal_stability: Optional[dict] = None
) -> None:
    """
    列印分析報告
    
    參數:
        error_data: 誤差數據
        analysis: 分析結果
        config: 驗證配置
    """
    print("\n" + "=" * 100)
    print("【1. 重投影誤差統計 - 各關節點分析】")
    print("=" * 100)
    
    for kp in error_data['keypoints']:
        stat = analysis['global_stat'][kp]
        print(f"{kp:<15s} Side={stat['side_mean']:6.2f}±{stat['side_std']:5.2f}   "
              f"45°={stat['45_mean']:6.2f}±{stat['45_std']:5.2f}")
    
    print("\n" + "=" * 100)
    print("【2. 全域誤差總覽】")
    print("=" * 100)
    print(f"Side 相機平均誤差: {analysis['global_mean_side']:.2f} px")
    print(f"45° 相機平均誤差:  {analysis['global_mean_45']:.2f} px")
    print(f"整體平均誤差:      {(analysis['global_mean_side'] + analysis['global_mean_45'])/2:.2f} px")
    print(f"品質等級:          {analysis['quality_level']}")
    
    all_side = analysis['all_side']
    all_45 = analysis['all_45']
    
    print("\n" + "=" * 100)
    print("【3. 誤差分佈統計（中位數/最大值/百分位）】")
    print("=" * 100)
    if all_side.size:
        print(f"Side median={np.median(all_side):.2f}, max={np.max(all_side):.2f}, "
              f"95th={np.percentile(all_side, 95):.2f}")
    else:
        print("Side 無有效誤差樣本")
    if all_45.size:
        print(f"45° median={np.median(all_45):.2f}, max={np.max(all_45):.2f}, "
              f"95th={np.percentile(all_45, 95):.2f}")
    else:
        print("45° 無有效誤差樣本")
    
    print("\n" + "=" * 100)
    print(f"【4. 異常值檢測（誤差 > {config.reprojection_outlier_threshold} px）】")
    print("=" * 100)
    side_outlier_rate = safe_ratio(analysis['outlier_count_side'], len(all_side))
    cam45_outlier_rate = safe_ratio(analysis['outlier_count_45'], len(all_45))
    print(f"Side 相機異常值: {analysis['outlier_count_side']} 個 ({side_outlier_rate:.2f}%)")
    print(f"45° 相機異常值:  {analysis['outlier_count_45']} 個 ({cam45_outlier_rate:.2f}%)")
    
    print("\n" + "=" * 100)
    print(f"【5. 異常誤差詳細分析】（誤差 > {config.reprojection_outlier_threshold:.1f} px）")
    print("=" * 100)
    
    if 'detailed_anomalies' in analysis and analysis['detailed_anomalies'].get('total_count', 0) > 0:
        detail = analysis['detailed_anomalies']
        total = detail['total_count']
        severe = detail['by_severity']['severe']
        moderate = detail['by_severity']['moderate']
        mild = detail['by_severity']['mild']
        
        print(f"總異常數: {total} 個（Side: {analysis['outlier_count_side']}, 45°: {analysis['outlier_count_45']}）")
        print()
        
        # 嚴重程度分級
        print("▸ 按嚴重程度分級:")
        severe_threshold = config.reprojection_outlier_threshold * 2
        moderate_threshold = config.reprojection_outlier_threshold * 1.5
        print(f"  • 嚴重 (>{severe_threshold:.0f}px):  {len(severe):3d} 個 ({len(severe)/total*100:5.1f}%) - [!] 建議優先檢查")
        print(f"  • 中等 ({moderate_threshold:.0f}-{severe_threshold:.0f}px): {len(moderate):3d} 個 ({len(moderate)/total*100:5.1f}%)")
        print(f"  • 輕微 ({config.reprojection_outlier_threshold:.0f}-{moderate_threshold:.0f}px): {len(mild):3d} 個 ({len(mild)/total*100:5.1f}%)")
        print()
        
        # 按關節點統計 TOP 5
        kp_stats = detail['by_keypoint']
        sorted_kps = sorted(kp_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        if sorted_kps:
            print("▸ 按關節點統計（TOP 5）:")
            for idx, (kp, stats) in enumerate(sorted_kps, 1):
                side_pct = stats['side_ratio'] * 100
                cam_info = f"Side 相機佔 {side_pct:.0f}%" if side_pct > 60 else f"45° 相機佔 {100-side_pct:.0f}%" if side_pct < 40 else "兩相機分布均勻"
                print(f"  {idx}. {kp:<15s}: {stats['count']:3d} 個 ({stats['count']/total*100:5.1f}%) - {cam_info}")
            print()
        
        # 連續異常區段
        segments = detail['continuous_segments']
        if segments:
            print("▸ 連續異常區段（≥5 幀）:")
            for seg in segments[:5]:  # 最多顯示 5 個
                cam_label = "Side" if seg['camera'] == "side" else "45°"
                print(f"  • Frame {seg['start_frame']:3d}-{seg['end_frame']:3d} ({seg['duration']:2d} 幀): "
                      f"{seg['keypoint']:<12s} @ {cam_label:<4s}, 平均 {seg['mean_error']:5.1f}px, "
                      f"最大 {seg['max_error']:5.1f}px")
            print()
        
        # 嚴重異常 - 全部顯示
        if severe:
            print("─" * 100)
            print(f"[!] 嚴重異常 (>{severe_threshold:.0f}px) - 全部 {len(severe)} 個:")
            print("─" * 100)
            for idx, (cam, frame, kp, err, z) in enumerate(severe, 1):
                cam_label = "Side" if cam == "side" else "45° "
                print(f" {idx:3d}. Frame {frame:3d}: {kp:<15s} @ {cam_label} - {err:6.2f}px (z={z:7.1f}mm)")
            print()
        
        # 中等異常 - 顯示前 10 個
        if moderate:
            print("─" * 100)
            display_count = min(10, len(moderate))
            print(f"📊 中等異常 ({moderate_threshold:.0f}-{severe_threshold:.0f}px) - 顯示前 {display_count} 個，共 {len(moderate)} 個:")
            print("─" * 100)
            moderate_sorted = sorted(moderate, key=lambda x: x[3], reverse=True)
            for idx, (cam, frame, kp, err, z) in enumerate(moderate_sorted[:display_count], 1):
                cam_label = "Side" if cam == "side" else "45° "
                print(f" {idx:3d}. Frame {frame:3d}: {kp:<15s} @ {cam_label} - {err:6.2f}px (z={z:7.1f}mm)")
            if len(moderate) > display_count:
                print(f"\n ⋮ 其餘 {len(moderate) - display_count} 個中等異常請參閱 JSON 輸出")
            print()
        
        # 輕微異常 - 只顯示統計
        if mild:
            print("─" * 100)
            print(f"📋 輕微異常 ({config.reprojection_outlier_threshold:.0f}-{moderate_threshold:.0f}px) - 共 {len(mild)} 個，詳見 JSON")
            print("─" * 100)
            print()
        
        print("💾 完整異常列表已儲存至 JSON:")
        print("   ✓ detailed_anomalies.all_anomalies     - 所有異常（按誤差排序）")
        print("   ✓ detailed_anomalies.by_keypoint       - 按關節點分組")
        print("   ✓ detailed_anomalies.continuous_segments - 連續異常區段")
    else:
        print("[OK] 未檢測到異常誤差")
    
    # 相機偏差分析
    diff = analysis['global_mean_side'] - analysis['global_mean_45']
    print("\n" + "=" * 100)
    print("【6. 相機偏差分析（Side vs 45°）】")
    print("=" * 100)
    print(f"Side - 45° 誤差差異: {diff:.2f} px")
    if diff > 0:
        print("[OK] 結論: Side 相機誤差較大")
    else:
        print("[OK] 結論: 45° 相機誤差較大")

    if depth_relationship:
        print("\n" + "=" * 100)
        print("【7. 誤差 vs 深度 關係】")
        print("=" * 100)
        print(f"相關係數: {depth_relationship['corr']:.3f}")
        print(f"趨勢: error = {depth_relationship['slope']:.4f} * depth + {depth_relationship['intercept']:.2f}")
        print(f"深度範圍: {depth_relationship['depth_min']:.1f} ~ {depth_relationship['depth_max']:.1f} mm")

    if temporal_stability:
        print("\n" + "=" * 100)
        print("【8. 誤差時間穩定性】")
        print("=" * 100)
        label_map = {"side": "Side", "camera_45": "45°"}
        for cam_label, stats in temporal_stability.items():
            pretty_label = label_map.get(cam_label, cam_label)
            if stats.get("samples", 0) == 0:
                print(f"{pretty_label}: 無有效樣本")
                continue
            print(f"{pretty_label}: 平均 {stats['mean']:.2f}px, CV={stats['cv']:.2f}%, 突波 {stats['spike_count']} 次, 最大跳變 {stats['max_jump']:.2f}px")


def validate_reprojection_analysis(
    json_3d_path: str,
    json_2d_side_path: str,
    json_2d_45_path: str,
    P1: np.ndarray,
    P2: np.ndarray,
    output_json_path: str = None,
    config_path: str = None
) -> dict:
    """
    重投影誤差驗證分析（主函數）
    
    參數:
        json_3d_path: 3D 軌跡 JSON 檔案路徑
        json_2d_side_path: Side 相機 2D 軌跡 JSON 檔案路徑
        json_2d_45_path: 45° 相機 2D 軌跡 JSON 檔案路徑
        P1: Side 相機投影矩陣 (3x4)
        P2: 45° 相機投影矩陣 (3x4)
        output_json_path: 輸出結果 JSON 路徑（可選）
        config_path: 配置檔案路徑（可選）
    
    返回:
        dict: 完整分析結果
    
    範例:
        >>> P1 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
        >>> P2 = np.array([[...], [...], [...]])
        >>> results = validate_reprojection_analysis(
        ...     '3d_data.json', '2d_side.json', '2d_45.json', P1, P2
        ... )
    """
    # 載入配置
    config = load_config(config_path)
    
    # 載入數據
    print(f"\n載入數據...")
    data_3d = load_json_file(json_3d_path)
    data_2d_side = load_json_file(json_2d_side_path)
    data_2d_45 = load_json_file(json_2d_45_path)
    
    print(f"3D 軌跡: {len(data_3d)} 幀")
    print(f"2D Side: {len(data_2d_side)} 幀")
    print(f"2D 45°:  {len(data_2d_45)} 幀")
    
    # 驗證相機內參
    print(f"\n驗證相機內參...")
    intrinsics_side = validate_camera_intrinsics(P1, "Side Camera")
    intrinsics_45 = validate_camera_intrinsics(P2, "45° Camera")
    
    if not intrinsics_side["is_valid"]:
        print(f"[!] Side 相機內參驗證失敗:")
        for err in intrinsics_side["errors"]:
            print(f"  - {err}")
    
    if not intrinsics_45["is_valid"]:
        print(f"[!] 45° 相機內參驗證失敗:")
        for err in intrinsics_45["errors"]:
            print(f"  - {err}")
    
    # 計算重投影誤差
    print(f"\n計算重投影誤差...")
    error_data = calculate_reprojection_errors(
        data_3d,
        data_2d_side,
        data_2d_45,
        P1,
        P2,
        config,
        distortion_side=config.side_camera_distortion,
        distortion_45=config.camera_45_distortion,
    )
    
    # 分析結果
    print(f"\n分析誤差數據...")
    analysis = analyze_reprojection_results(error_data, config)
    depth_relationship = analyze_error_depth_relationship(error_data['error_records'])
    temporal_stability = analyze_temporal_stability(
        error_data['per_frame_error_side'],
        error_data['per_frame_error_45'],
        config
    )
    
    # 詳細異常分析
    print(f"\n執行詳細異常分析...")
    detailed_anomalies = analyze_detailed_anomalies(error_data['error_records'], config)
    analysis['detailed_anomalies'] = detailed_anomalies

    # 整理完整誤差數據供前端繪圖
    full_keypoint_details = {kp: [] for kp in error_data['keypoints']}
    for cam, frame, kp, err, z in error_data['error_records']:
        full_keypoint_details[kp].append({
            "frame": int(frame),
            "camera": cam,
            "error": float(err),
            "depth_z": float(z)
        })
    
    # [新增] 提取信心值數據
    print(f"\n提取信心值數據 (Confidence Series)...")
    conf_data_side = extract_confidence_series(json_2d_side_path)
    conf_data_45 = extract_confidence_series(json_2d_45_path)
     
     # 列印報告
    print_analysis_report(
        error_data,
        analysis,
        config,
        depth_relationship=depth_relationship,
        temporal_stability=temporal_stability
    )
    # 整合結果（確保所有數值都經過序列化處理）
    results = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "source_file": str(json_3d_path),
            "total_frames": int(error_data['num_frames']),
            "total_keypoints": int(len(error_data['keypoints'])),
            "analysis_type": "Reprojection Error Analysis"
        },
        "camera_intrinsics_validation": {
            "side_camera": intrinsics_side,
            "45_camera": intrinsics_45
        },
        "global_stats": {
            "overall_mean": (analysis['global_mean_side'] + analysis['global_mean_45']) / 2,
            "side_mean": analysis['global_mean_side'],
            "45_mean": analysis['global_mean_45'],
            "side_median": float(np.median(analysis['all_side'])) if analysis['all_side'].size else 0.0,
            "45_median": float(np.median(analysis['all_45'])) if analysis['all_45'].size else 0.0,
            "side_max": float(np.max(analysis['all_side'])) if analysis['all_side'].size else 0.0,
            "45_max": float(np.max(analysis['all_45'])) if analysis['all_45'].size else 0.0,
            "side_95th": float(np.percentile(analysis['all_side'], 95)) if analysis['all_side'].size else 0.0,
            "45_95th": float(np.percentile(analysis['all_45'], 95)) if analysis['all_45'].size else 0.0,
            "outlier_count_side": analysis['outlier_count_side'],
            "outlier_count_45": analysis['outlier_count_45'],
            "outlier_rate_side": safe_ratio(analysis['outlier_count_side'], len(analysis['all_side'])),
            "outlier_rate_45": safe_ratio(analysis['outlier_count_45'], len(analysis['all_45'])),
            "camera_bias": analysis['global_mean_side'] - analysis['global_mean_45'],
            "quality_level": analysis['quality_level']
        },
        "keypoint_errors": [
            {
                "name": kp,
                **analysis['global_stat'][kp]
            }
            for kp in error_data['keypoints']
        ],
        "per_frame_errors": {
            "frames": list(range(error_data['num_frames'])),
            "side": [None if np.isnan(v) else float(v) for v in error_data['per_frame_error_side']],
            "45": [None if np.isnan(v) else float(v) for v in error_data['per_frame_error_45']]
        },
        "top10_worst_errors": [
            {
                "rank": i + 1,
                "camera": cam,
                "frame": int(frame),
                "keypoint": kp,
                "error": float(err),
                "depth_z": float(z)
            }
            for i, (cam, frame, kp, err, z) in enumerate(analysis['error_sorted'][:10])
        ],
        "error_depth_relationship": depth_relationship,
        "temporal_stability": temporal_stability,
        "full_keypoint_details": full_keypoint_details,
        "detailed_anomalies": {
            "summary": {
                "total_count": detailed_anomalies.get('total_count', 0),
                "severe_count": len(detailed_anomalies.get('by_severity', {}).get('severe', [])),
                "moderate_count": len(detailed_anomalies.get('by_severity', {}).get('moderate', [])),
                "mild_count": len(detailed_anomalies.get('by_severity', {}).get('mild', []))
            },
            "by_keypoint": detailed_anomalies.get('by_keypoint', {}),
            "continuous_segments": detailed_anomalies.get('continuous_segments', []),
            "all_anomalies": [
                {
                    "frame": int(frame),
                    "camera": cam,
                    "keypoint": kp,
                    "error": float(err),
                    "depth_z": float(z),
                    "severity": "severe" if err > config.reprojection_outlier_threshold * 2 else 
                               "moderate" if err > config.reprojection_outlier_threshold * 1.5 else "mild"
                }
                for cam, frame, kp, err, z in (detailed_anomalies.get('by_severity', {}).get('severe', []) +
                                               detailed_anomalies.get('by_severity', {}).get('moderate', []) +
                                               detailed_anomalies.get('by_severity', {}).get('mild', []))
            ]
        } if detailed_anomalies.get('total_count', 0) > 0 else {},
        
        # [新增] 將信心值數據加入輸出 JSON
        "confidence_trends": {
            "frames": conf_data_side.get("frames", []),
            "side": conf_data_side.get("series", {}),
            "camera_45": conf_data_45.get("series", {})
        }
    }
    
    # 保存結果
    if output_json_path is None:
        # 建立 results 資料夾
        results_dir = os.path.join(os.path.dirname(json_3d_path), 'Verification Result')
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(json_3d_path))[0]
        output_json_path = os.path.join(results_dir, f"{base_name}_step1_reprojection_error_results.json")
    
    save_json_results(results, output_json_path)
    print(f"\n[OK] 結果已儲存至: {output_json_path}")
    
    return results


# ========================================================
# 主程式
# ========================================================

if __name__ == "__main__":
    # 預設投影矩陣（請依實際標定結果修改）
    P1 = np.array([
        [917.153880, 0.000000, 994.529968, 0.000000],
        [0.000000, 920.803487, 531.057076, 0.000000],
        [0.000000, 0.000000, 1.000000, 0.000000],
    ])

    P2 = np.array([
        [286.476533, 43.805594, 1301.943509, -765436.820164],
        [-309.560886, 957.641377, 401.534167, 365723.173062],
        [-0.553187, 0.008475, 0.833014, 660.964347],
    ])

    # 命令列參數支援
    if len(sys.argv) >= 4:
        json_3d_path = sys.argv[1]
        json_2d_side_path = sys.argv[2]
        json_2d_45_path = sys.argv[3]
        output_json_path = sys.argv[4] if len(sys.argv) > 4 else None
        config_path = sys.argv[5] if len(sys.argv) > 5 else None
    else:
        # 預設測試路徑
        json_3d_path = "data/trajectory_v11x(new)/test1-v11x(new)__1_segment(3D_trajectory).json"
        json_2d_45_path = "data/trajectory_v11x(new)/test1-v11x(new)__1_45_segment(2D_trajectory).json"
        json_2d_side_path = "data/trajectory_v11x(new)/test1-v11x(new)__1_side_segment(2D_trajectory).json"
        output_json_path = None
        config_path = None
        print("提示: 可使用命令列參數:")
        print("  python step1_Reprojection_Error.py <3d_json> <2d_side_json> <2d_45_json> [output_json] [config_json]")

    # 執行分析
    try:
        results = validate_reprojection_analysis(
            json_3d_path,
            json_2d_side_path,
            json_2d_45_path,
            P1, P2,
            output_json_path,
            config_path
        )
    except Exception as e:
        print(f"\n[ERROR] 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)