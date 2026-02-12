"""
擊球出拍/轉身分析模組（Hitball-swing）
--------------------------------------------------------
分析時間範圍：擊球幀（tennis_ball_hit = true）

兩個分析點：
A. 轉身動作：肩線和腰線在 XZ 平面的投影距離，與專家範圍比較
B. 右手腕位置：高度（膝蓋~肩膀）+ 水平位置（身體朝向與手腕方向的夾角）
--------------------------------------------------------
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


# ========== 工具函式區 ==========
def load_json(file_path: str) -> Dict:
    """載入JSON檔案"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _get_point(frame: Dict, name: str) -> Optional[np.ndarray]:
    """從幀中提取關鍵點座標"""
    p = frame.get(name)
    if not p:
        return None
    x, y, z = p.get("x"), p.get("y"), p.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def _find_impact_frame(frames: List[Dict]) -> Optional[int]:
    """找出擊球幀的索引"""
    for i, frame in enumerate(frames):
        if frame.get("tennis_ball_hit"):
            return i
    return None


def _calculate_xz_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點在 XZ 平面的投影距離"""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    return float(np.sqrt(dx**2 + dz**2))


def _calculate_3d_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點的 3D 距離（用於計算骨盆寬度）"""
    return float(np.linalg.norm(p2 - p1))


def _calculate_xz_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """計算兩個 2D 向量（XZ 平面）的夾角（度）"""
    # 正規化
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    v1_n = v1 / norm1
    v2_n = v2 / norm2
    
    # 計算夾角
    dot = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    return float(angle)


# ========== 專家數據統計 ==========
def _calculate_pro_ranges(knn_dataset: List[Dict]) -> Dict:
    """
    從 knn_dataset_new.json 計算 pro 專家的統計範圍
    返回 A（肩腰線距離，已正規化）和 B（手腕夾角）的 P10/P50/P90
    
    正規化方式：肩線和腰線距離（XZ 平面投影）除以骨盆寬度（XZ 平面投影）
    這樣可以消除不同體型造成的影響，更公平地比較動作模式
    """
    pro_data = [d for d in knn_dataset if d.get("level") == "pro"]
    
    shoulder_dists = []
    hip_dists = []
    wrist_angles = []
    wrist_heights_relative = []  # 相對於膝蓋~肩膀的比例
    
    for expert in pro_data:
        frames = expert.get("data", [])
        if not frames:
            continue
        
        # 找擊球幀
        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            continue
        
        frame = frames[impact_idx]
        
        # 取得點位
        ls = _get_point(frame, "left_shoulder")
        rs = _get_point(frame, "right_shoulder")
        lh = _get_point(frame, "left_hip")
        rh = _get_point(frame, "right_hip")
        rw = _get_point(frame, "right_wrist")
        rk = _get_point(frame, "right_knee")
        
        # A: 肩腰線距離（正規化）
        if all(p is not None for p in [ls, rs, lh, rh]):
            # 計算骨盆寬度作為標準化基準（XZ 平面投影，與肩腰線距離一致）
            pelvis_width_xz = _calculate_xz_distance(lh, rh)
            if pelvis_width_xz > 1e-6:  # 避免除以零
                shoulder_dist = _calculate_xz_distance(ls, rs)
                hip_dist = _calculate_xz_distance(lh, rh)
                # 正規化：除以骨盆寬度（XZ 平面投影）
                shoulder_dist_norm = shoulder_dist / pelvis_width_xz
                hip_dist_norm = hip_dist / pelvis_width_xz
                shoulder_dists.append(shoulder_dist_norm)
                hip_dists.append(hip_dist_norm)
        
        # B: 手腕夾角和高度
        if all(p is not None for p in [ls, rs, rw]):
            # 身體中心
            body_center = (ls + rs) / 2
            
            # 身體朝向（XZ 平面）
            body_dir_xz = np.array([ls[0] - rs[0], ls[2] - rs[2]])
            
            # 手腕方向（XZ 平面）
            wrist_dir_xz = np.array([rw[0] - body_center[0], rw[2] - body_center[2]])
            
            # 計算夾角
            angle = _calculate_xz_angle(body_dir_xz, wrist_dir_xz)
            wrist_angles.append(angle)
        
        # 手腕高度相對位置
        if all(p is not None for p in [rw, rk, rs]):
            knee_y = rk[1]
            shoulder_y = rs[1]
            wrist_y = rw[1]
            
            # 計算相對位置（0=膝蓋高度, 1=肩膀高度）
            if abs(shoulder_y - knee_y) > 1e-6:
                relative_height = (wrist_y - knee_y) / (shoulder_y - knee_y)
                wrist_heights_relative.append(relative_height)
    
    # 計算分位數
    result = {
        "shoulder_dist": {},
        "hip_dist": {},
        "wrist_angle": {},
        "sample_count": len(pro_data)
    }
    
    if shoulder_dists:
        result["shoulder_dist"] = {
            "p10": float(np.percentile(shoulder_dists, 10)),
            "p50": float(np.percentile(shoulder_dists, 50)),
            "p90": float(np.percentile(shoulder_dists, 90))
        }
    
    if hip_dists:
        result["hip_dist"] = {
            "p10": float(np.percentile(hip_dists, 10)),
            "p50": float(np.percentile(hip_dists, 50)),
            "p90": float(np.percentile(hip_dists, 90))
        }
    
    if wrist_angles:
        result["wrist_angle"] = {
            "p10": float(np.percentile(wrist_angles, 10)),
            "p50": float(np.percentile(wrist_angles, 50)),
            "p90": float(np.percentile(wrist_angles, 90))
        }
    
    return result


# ========== A. 轉身動作分析 ==========
def _analyze_body_rotation(frame: Dict, pro_ranges: Dict) -> Dict:
    """
    分析A：轉身動作
    計算肩線和腰線在 XZ 平面的投影距離（正規化後），與專家範圍比較
    
    正規化：距離（XZ 平面投影）除以骨盆寬度（XZ 平面投影）
    這樣可以消除不同體型造成的影響，更公平地比較動作模式
    """
    result = {
        "is_valid": False,
        "shoulder_dist": None,
        "hip_dist": None,
        "shoulder_in_range": None,
        "hip_in_range": None,
        "level": None,
        "advice": None
    }
    
    ls = _get_point(frame, "left_shoulder")
    rs = _get_point(frame, "right_shoulder")
    lh = _get_point(frame, "left_hip")
    rh = _get_point(frame, "right_hip")
    
    if any(p is None for p in [ls, rs, lh, rh]):
        return result
    
    # 計算骨盆寬度作為標準化基準（XZ 平面投影，與肩腰線距離一致）
    pelvis_width_xz = _calculate_xz_distance(lh, rh)
    if pelvis_width_xz <= 1e-6:
        result.update({
            "is_valid": False,
            "advice": "無法計算骨盆寬度，數據不足"
        })
        return result
    
    # 計算 XZ 平面距離並正規化
    shoulder_dist_raw = _calculate_xz_distance(ls, rs)
    hip_dist_raw = _calculate_xz_distance(lh, rh)
    shoulder_dist = shoulder_dist_raw / pelvis_width_xz  # 正規化
    hip_dist = hip_dist_raw / pelvis_width_xz  # 正規化
    
    # 取得專家範圍（已正規化）
    shoulder_range = pro_ranges.get("shoulder_dist", {})
    hip_range = pro_ranges.get("hip_dist", {})
    
    if not shoulder_range or not hip_range:
        result.update({
            "is_valid": True,
            "shoulder_dist": shoulder_dist,
            "hip_dist": hip_dist,
            "shoulder_dist_raw": shoulder_dist_raw,
            "hip_dist_raw": hip_dist_raw,
            "pelvis_width_xz": pelvis_width_xz,
            "level": 0,
            "advice": "專家數據不足，無法比較"
        })
        return result
    
    # 判斷是否在範圍內（使用正規化後的值）
    shoulder_in_range = shoulder_range["p10"] <= shoulder_dist <= shoulder_range["p90"]
    hip_in_range = hip_range["p10"] <= hip_dist <= hip_range["p90"]
    sd, hd = round(shoulder_dist, 2), round(hip_dist, 2)
    s_p50 = shoulder_range.get("p50", (shoulder_range["p10"] + shoulder_range["p90"]) / 2)
    h_p50 = hip_range.get("p50", (hip_range["p10"] + hip_range["p90"]) / 2)
    
    # 判斷等級和建議
    if shoulder_in_range and hip_in_range:
        level = 1
        advice = "擊球時轉身動作得宜"
    elif shoulder_dist < shoulder_range["p10"] or hip_dist < hip_range["p10"]:
        diff_s = round(shoulder_dist - s_p50, 2) if not shoulder_in_range else 0
        diff_h = round(hip_dist - h_p50, 2) if not hip_in_range else 0
        diff_str = []
        if diff_s != 0:
            diff_str.append(f"肩線與專家中位數差 {diff_s}（正規化比例）")
        if diff_h != 0:
            diff_str.append(f"腰線與專家中位數差 {diff_h}（正規化比例）")
        ext = f"（{', '.join(diff_str)}）" if diff_str else ""
        level = 2
        advice = f"擊球時轉身動作不足{ext}。建議肩腰旋轉再增加，擊球時記得轉動腰部和肩膀。"
    elif shoulder_dist > shoulder_range["p90"] or hip_dist > hip_range["p90"]:
        diff_s = round(shoulder_dist - s_p50, 2) if shoulder_dist > shoulder_range["p90"] else 0
        diff_h = round(hip_dist - h_p50, 2) if hip_dist > hip_range["p90"] else 0
        diff_str = []
        if diff_s != 0:
            diff_str.append(f"肩線與專家中位數差 {diff_s}（正規化比例）")
        if diff_h != 0:
            diff_str.append(f"腰線與專家中位數差 {diff_h}（正規化比例）")
        ext = f"（{', '.join(diff_str)}）" if diff_str else ""
        level = 3
        advice = f"擊球時轉身動作過頭{ext}。建議肩膀轉動不要過大，身體正面朝向擊球方向後就要變慢停止。"
    else:
        # 其他超出範圍情況（混合或邊界），仍加上與 P50 的差值
        level = 2
        diff_s = round(shoulder_dist - s_p50, 2)
        diff_h = round(hip_dist - h_p50, 2)
        diff_str = []
        if diff_s != 0:
            diff_str.append(f"肩線與專家中位數差 {diff_s}（正規化比例）")
        if diff_h != 0:
            diff_str.append(f"腰線與專家中位數差 {diff_h}（正規化比例）")
        ext = f"（{', '.join(diff_str)}）" if diff_str else ""
        advice = f"擊球時轉身動作不足{ext}。建議擊球時記得轉動腰部和肩膀。"
    
    result.update({
        "is_valid": True,
        "shoulder_dist": shoulder_dist,  # 正規化後的值
        "hip_dist": hip_dist,  # 正規化後的值
        "shoulder_dist_raw": shoulder_dist_raw,  # 原始值（用於參考）
        "hip_dist_raw": hip_dist_raw,  # 原始值（用於參考）
        "pelvis_width_xz": pelvis_width_xz,  # 骨盆寬度（XZ 平面投影，用於參考）
        "shoulder_in_range": shoulder_in_range,
        "hip_in_range": hip_in_range,
        "pro_shoulder_range": shoulder_range,
        "pro_hip_range": hip_range,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== B. 右手腕位置分析 ==========
def _analyze_wrist_position(frame: Dict, pro_ranges: Dict) -> Dict:
    """
    分析B：右手腕位置
    1. 高度：右手腕 Y 在「右膝 Y ~ 右肩 Y」之間
    2. 水平位置：身體朝向與手腕方向的夾角，與專家範圍比較
    """
    result = {
        "is_valid": False,
        "wrist_y": None,
        "knee_y": None,
        "shoulder_y": None,
        "height_ok": None,
        "wrist_angle": None,
        "angle_in_range": None,
        "level": None,
        "advice": None
    }
    
    ls = _get_point(frame, "left_shoulder")
    rs = _get_point(frame, "right_shoulder")
    rw = _get_point(frame, "right_wrist")
    rk = _get_point(frame, "right_knee")
    
    if any(p is None for p in [ls, rs, rw, rk]):
        return result
    
    # 高度判斷（Y 越大 = 越低）
    wrist_y = rw[1]
    knee_y = rk[1]
    shoulder_y = rs[1]
    
    # 注意：Y 越大越低，所以 shoulder_y < knee_y
    # 手腕要在膝蓋和肩膀之間
    y_min = min(shoulder_y, knee_y)
    y_max = max(shoulder_y, knee_y)
    height_ok = y_min <= wrist_y <= y_max
    
    # 水平位置判斷
    body_center = (ls + rs) / 2
    body_dir_xz = np.array([ls[0] - rs[0], ls[2] - rs[2]])
    wrist_dir_xz = np.array([rw[0] - body_center[0], rw[2] - body_center[2]])
    wrist_angle = _calculate_xz_angle(body_dir_xz, wrist_dir_xz)
    
    # 取得專家範圍
    angle_range = pro_ranges.get("wrist_angle", {})
    
    if not angle_range:
        angle_in_range = None
        angle_advice = "專家數據不足"
    else:
        angle_in_range = angle_range["p10"] <= wrist_angle <= angle_range["p90"]
        
        if angle_in_range:
            angle_advice = "位置得宜"
        elif wrist_angle < angle_range["p10"]:
            angle_advice = "位置太前面"
        else:
            angle_advice = "位置太後方"
    
    # 綜合判斷（在範圍內只顯示使用者值；超出範圍則加 P50 差值）
    wa = round(wrist_angle, 1)
    has_angle_range = angle_range and "p10" in angle_range and "p90" in angle_range
    a_p50 = angle_range.get("p50", (angle_range["p10"] + angle_range["p90"]) / 2) if has_angle_range else None
    if has_angle_range and (angle_in_range is None or angle_in_range):
        angle_str = ""
    elif has_angle_range and angle_in_range is False:
        diff_a = round(wrist_angle - a_p50, 1)
        angle_str = f"（與專家中位數差 {diff_a}°）"
    else:
        angle_str = ""
    
    if height_ok and (angle_in_range is None or angle_in_range):
        level = 1
        advice = f"擊球時右手腕位置得宜{angle_str}"
    elif angle_in_range is False and wrist_angle < angle_range.get("p10", 0):
        # 夾角太小 = 太後方
        level = 2
        advice = f"擊球時右手腕位置太後方{angle_str}。建議手腕往前移動，擊到球時手腕可以在身體前方。"
    elif angle_in_range is False and wrist_angle > angle_range.get("p90", 180):
        # 夾角太大 = 太前面
        level = 3
        advice = f"擊球時右手腕位置太前面{angle_str}。建議手腕往後調整，擊到球時手腕位置拉回來一些。"
    elif not height_ok:
        # 高度問題也歸類為位置問題
        if wrist_y < y_min:
            level = 2
            advice = f"擊球時右手腕位置太後方{angle_str}。建議擊到球時手腕可以在身體前方。"
        else:
            level = 3
            advice = f"擊球時右手腕位置太前面{angle_str}。建議擊到球時手腕位置拉回來一些。"
    else:
        level = 2
        advice = f"擊球時右手腕位置太後方{angle_str}。建議擊到球時手腕可以在身體前方。"
    
    result.update({
        "is_valid": True,
        "wrist_y": float(wrist_y),
        "knee_y": float(knee_y),
        "shoulder_y": float(shoulder_y),
        "height_ok": height_ok,
        "wrist_angle": wrist_angle,
        "angle_in_range": angle_in_range,
        "pro_angle_range": angle_range,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== 主分析函式 ==========
def analyze_hitballswing(trajectory_data, knn_dataset_path: str = "knn_dataset_new.json", expert_filename: str = None) -> Tuple[str, float]:
    """
    擊球出拍/轉身分析（Hitball-swing）
    
    分析時間：擊球幀
    
    兩個分析點：
    A. 轉身動作
    B. 右手腕位置
    
    Args:
        trajectory_data: 軌跡數據（可以是路徑或數據字典）
        knn_dataset_path: KNN數據集路徑（預設 knn_dataset_new.json）
        expert_filename: 專家文件名（可選）
    
    Returns:
        (建議文字, 信心度)
    """
    try:
        # 載入 KNN 數據集
        knn_dataset = load_json(knn_dataset_path)
        
        # 計算專家範圍
        pro_ranges = _calculate_pro_ranges(knn_dataset)
        
        # 處理輸入資料格式
        if isinstance(trajectory_data, str):
            frames_data = load_json(trajectory_data)
        elif isinstance(trajectory_data, dict):
            frames_data = trajectory_data.get("data", trajectory_data)
        else:
            frames_data = trajectory_data
        
        if not isinstance(frames_data, list) or len(frames_data) == 0:
            return "軌跡數據格式不正確", 0.0, None
        
        # 找出擊球幀
        impact_idx = _find_impact_frame(frames_data)
        
        if impact_idx is None:
            return "未找到擊球幀，無法進行擊球出拍分析", 0.0, None
        
        impact_frame = frames_data[impact_idx]
        
        # ========== 兩個分析點 ==========
        
        # A. 轉身動作
        a_result = _analyze_body_rotation(impact_frame, pro_ranges)
        
        # B. 右手腕位置
        b_result = _analyze_wrist_position(impact_frame, pro_ranges)
        
        # ========== 組合建議 ==========
        advice_parts = []
        
        if a_result["is_valid"]:
            advice_parts.append(f"A.轉身:{a_result['advice']}")
        else:
            advice_parts.append("A.轉身:數據不足")
        
        if b_result["is_valid"]:
            advice_parts.append(f"B.手腕位置:{b_result['advice']}")
        else:
            advice_parts.append("B.手腕位置:數據不足")
        
        combined_advice = "".join(advice_parts)
        
        # 計算信心度 (基於 Level)
        # Level 1 = 1.0 (100分), Level 2 = 0.8 (80分), Level 3 = 0.6 (60分), Invalid = 0.0
        score_map = {1: 1.0, 2: 0.8, 3: 0.6, 0: 0.0, None: 0.0}
        
        # 收集個別項目的 Level
        levels = {
            "擊球轉身動作": a_result.get("level"),
            "擊球手腕位置": b_result.get("level")
        }
        
        scores = [score_map.get(lvl, 0.0) for lvl in levels.values()]
        
        # 找出優先改善項目 (Level 3 > Level 2)
        priority_item = None
        max_level = 0
        
        for name, lvl in levels.items():
            if lvl and lvl > max_level:
                max_level = lvl
                priority_item = name
            elif lvl and lvl == max_level and max_level >= 2:
                pass
        
        if max_level < 2:
            priority_item = None
        
        # 如果所有項目都無效，信心度為 0
        if all(s == 0.0 for s in scores) and valid_count == 0:
            confidence = 0.0
        else:
            confidence = sum(scores) / 2.0
        
        return combined_advice, confidence, priority_item
        
    except Exception as e:
        print(f"擊球出拍分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return f"擊球出拍分析失敗: {str(e)}", 0.0, None


# ========== 詳細分析函式 ==========
def analyze_hitballswing_detailed(trajectory_data, knn_dataset_path: str = "knn_dataset_new.json", expert_filename: str = None) -> Dict:
    """
    擊球出拍詳細分析，返回兩個分析點的完整結果
    """
    try:
        # 載入 KNN 數據集
        knn_dataset = load_json(knn_dataset_path)
        
        # 計算專家範圍
        pro_ranges = _calculate_pro_ranges(knn_dataset)
        
        # 處理輸入資料格式
        if isinstance(trajectory_data, str):
            frames_data = load_json(trajectory_data)
        elif isinstance(trajectory_data, dict):
            frames_data = trajectory_data.get("data", trajectory_data)
        else:
            frames_data = trajectory_data
        
        if not isinstance(frames_data, list) or len(frames_data) == 0:
            return {"error": "軌跡數據格式不正確"}
        
        # 找出擊球幀
        impact_idx = _find_impact_frame(frames_data)
        
        if impact_idx is None:
            return {"error": "未找到擊球幀"}
        
        impact_frame = frames_data[impact_idx]
        
        return {
            "impact_idx": impact_idx,
            "pro_ranges": pro_ranges,
            "A_body_rotation": _analyze_body_rotation(impact_frame, pro_ranges),
            "B_wrist_position": _analyze_wrist_position(impact_frame, pro_ranges)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ========== 測試用 ==========
if __name__ == "__main__":
    import os
    
    # 確認檔案路徑
    knn_path = "knn_dataset_new.json"
    if not os.path.exists(knn_path):
        print(f"找不到 {knn_path}")
        exit(1)
    
    # 載入測試數據
    with open(knn_path, 'r') as f:
        data = json.load(f)
    
    test_data = data[0]
    print(f"測試檔案: {test_data.get('filename')}")
    print(f"Level: {test_data.get('level')}")
    print()
    
    # 簡單輸出
    suggestion, confidence = analyze_hitballswing(test_data, knn_path)
    print(f"建議: {suggestion}")
    print(f"信心度: {confidence:.2f}")
    print()
    
    # 詳細輸出
    detailed = analyze_hitballswing_detailed(test_data, knn_path)
    for key, value in detailed.items():
        print(f"{key}: {value}")
