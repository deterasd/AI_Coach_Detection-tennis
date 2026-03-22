"""
拉拍分析模組（v4: 四點獨立分析版）
--------------------------------------------------------
四個獨立分析點：
A. 側身完整度：肩線與腰線在XZ平面的旋轉角度
B. 右手腕高度：手腕Y座標與肩腰中點比較（Y越大=越低）
C. 球拍頭朝向：paddle.top與手腕向量與水平線夾角
D. 拉拍準備時機：球落地時手腕是否已在身體後方
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


def _get_nested_point(frame: Dict, *keys) -> Optional[np.ndarray]:
    """從幀中提取巢狀結構的關鍵點座標（如 paddle.top）"""
    p = frame
    for key in keys:
        if not isinstance(p, dict):
            return None
        p = p.get(key)
        if p is None:
            return None
    x, y, z = p.get("x"), p.get("y"), p.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def _calculate_angle_xz(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點在XZ平面的角度（度）"""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    return float(np.degrees(np.arctan2(dz, dx)))


def _find_ball_bounce_frame(frames: List[Dict], impact_idx: int = None) -> Optional[int]:
    """找出球落地幀的索引（Y值最大的幀，因為Y越大=越低）"""
    max_y = -float('inf')
    bounce_idx = None
    
    # 搜尋範圍：如果已知擊球幀，只找擊球前的幀；否則找全部
    search_end = impact_idx if impact_idx is not None else len(frames)
    
    for i in range(search_end):
        frame = frames[i]
        ball = _get_point(frame, "tennis_ball")
        if ball is not None:
            if ball[1] > max_y:
                max_y = ball[1]
                bounce_idx = i
    
    return bounce_idx


def _find_impact_frame(frames: List[Dict]) -> Optional[int]:
    """找出擊球幀的索引"""
    for i, frame in enumerate(frames):
        if frame.get("tennis_ball_hit"):
            return i
    return None


def _find_backswing_ready_frame(frames: List[Dict], bounce_idx: Optional[int], impact_idx: Optional[int]) -> Optional[int]:
    """
    找出拉拍準備好的幀（同時考慮手腕最高點和最遠點）
    
    使用加權組合分數來同時考慮：
    1. 手腕高度（Y值最小 = 最高）
    2. 手腕在XZ平面距離身體的距離（最遠）
    
    這樣可以處理「最高點」和「最遠點」不在同一幀的情況
    
    Args:
        frames: 所有幀的列表
        bounce_idx: 球落地幀索引
        impact_idx: 擊球幀索引
    
    Returns:
        拉拍準備好的幀索引，如果找不到則返回 None
    """
    if impact_idx is None:
        return None
    
    # 決定搜尋範圍
    # 優先：從球落地到擊球前
    if bounce_idx is not None and bounce_idx < impact_idx:
        search_start = bounce_idx
        search_end = impact_idx
    # 備用：從開始到擊球前（至少保留10幀的搜尋空間）
    elif impact_idx > 10:
        search_start = max(0, impact_idx - 30)  # 擊球前30幀開始搜尋
        search_end = impact_idx
    else:
        return None
    
    if search_end - search_start < 3:
        return None
    
    # 先計算所有候選幀的分數，找出最大值和最小值用於正規化
    candidates = []
    
    for i in range(search_start, search_end):
        frame = frames[i]
        right_wrist = _get_point(frame, "right_wrist")
        right_shoulder = _get_point(frame, "right_shoulder")
        
        if right_wrist is None or right_shoulder is None:
            continue
        
        # 指標1: 手腕高度（Y越小=越高，分數越高）
        wrist_y = right_wrist[1]
        height_score = -wrist_y  # 負號因為Y越大=越低，所以取負值
        
        # 指標2: 手腕在XZ平面距離身體的距離（距離越遠，分數越高）
        wrist_relative = right_wrist - right_shoulder
        xz_distance = np.sqrt(wrist_relative[0]**2 + wrist_relative[2]**2)
        distance_score = xz_distance
        
        candidates.append({
            'frame_idx': i,
            'height_score': height_score,
            'distance_score': distance_score,
            'wrist_y': wrist_y,
            'xz_distance': xz_distance
        })
    
    if len(candidates) == 0:
        return None
    
    # 正規化分數（讓兩個指標都在相似的範圍內）
    height_scores = [c['height_score'] for c in candidates]
    distance_scores = [c['distance_score'] for c in candidates]
    
    height_min, height_max = min(height_scores), max(height_scores)
    distance_min, distance_max = min(distance_scores), max(distance_scores)
    
    height_range = height_max - height_min if height_max != height_min else 1.0
    distance_range = distance_max - distance_min if distance_max != distance_min else 1.0
    
    # 計算組合分數（可以調整權重：0.6 高度 + 0.4 距離）
    best_score = -float('inf')
    best_frame_idx = None
    
    for c in candidates:
        normalized_height = (c['height_score'] - height_min) / height_range
        normalized_distance = (c['distance_score'] - distance_min) / distance_range
        
        # 加權組合：高度權重 0.6，距離權重 0.4
        combined_score = normalized_height * 0.6 + normalized_distance * 0.4
        
        if combined_score > best_score:
            best_score = combined_score
            best_frame_idx = c['frame_idx']
    
    return best_frame_idx


def _calculate_paddle_angle_from_frame(frame: Dict) -> Optional[float]:
    """從幀中計算球拍角度（與水平面夾角，正值=拍頭朝上）"""
    paddle_top = _get_nested_point(frame, "paddle", "top")
    paddle_bottom = _get_nested_point(frame, "paddle", "bottom")
    if paddle_top is None or paddle_bottom is None:
        return None
    paddle_vector = paddle_top - paddle_bottom
    horizontal_dist = np.sqrt(paddle_vector[0]**2 + paddle_vector[2]**2)
    vertical_dist = -paddle_vector[1]
    return float(np.degrees(np.arctan2(vertical_dist, horizontal_dist)))


def _calculate_pro_backswing_ranges(knn_dataset) -> Dict:
    """
    從 knn_dataset 計算 pro 專家的 A（肩腰角度差）與 C（球拍角度）的 P10/P50/P90。
    對每位 pro 取拉拍準備幀或球落地幀，計算 rotation_diff 與 paddle_angle。
    """
    if isinstance(knn_dataset, str):
        try:
            data = load_json(knn_dataset)
            knn_dataset = data if isinstance(data, list) else data.get("data", data)
        except Exception:
            return {}
    if not isinstance(knn_dataset, list):
        return {}
    
    pro_data = [d for d in knn_dataset if isinstance(d, dict) and d.get("level") == "pro"]
    rotation_diffs = []
    paddle_angles = []
    
    for expert in pro_data:
        frames = expert.get("data", [])
        if not frames:
            continue
        impact_idx = _find_impact_frame(frames)
        bounce_idx = _find_ball_bounce_frame(frames, impact_idx)
        backswing_idx = _find_backswing_ready_frame(frames, bounce_idx, impact_idx)
        frame_idx = backswing_idx if backswing_idx is not None else bounce_idx
        if frame_idx is None or frame_idx >= len(frames):
            continue
        frame = frames[frame_idx]
        
        # A: rotation_diff
        ls = _get_point(frame, "left_shoulder")
        rs = _get_point(frame, "right_shoulder")
        lh = _get_point(frame, "left_hip")
        rh = _get_point(frame, "right_hip")
        if all(p is not None for p in [ls, rs, lh, rh]):
            sa = _calculate_angle_xz(ls, rs)
            ha = _calculate_angle_xz(lh, rh)
            rd = abs(sa - ha)
            if rd > 180:
                rd = 360 - rd
            rotation_diffs.append(rd)
        
        # C: paddle_angle
        pa = _calculate_paddle_angle_from_frame(frame)
        if pa is not None:
            paddle_angles.append(pa)
    
    result = {}
    if rotation_diffs:
        result["rotation_diff"] = {
            "p10": float(np.percentile(rotation_diffs, 10)),
            "p50": float(np.percentile(rotation_diffs, 50)),
            "p90": float(np.percentile(rotation_diffs, 90)),
        }
    if paddle_angles:
        result["paddle_angle"] = {
            "p10": float(np.percentile(paddle_angles, 10)),
            "p50": float(np.percentile(paddle_angles, 50)),
            "p90": float(np.percentile(paddle_angles, 90)),
        }
    return result


# ========== A. 側身完整度分析 ==========
def _analyze_body_rotation(frame: Dict, pro_ranges: Optional[Dict] = None) -> Dict:
    """
    分析側身完整度
    檢查肩線向量與腰線向量是否平行（角度差越小越好）
    同時檢查四個點是否都有值
    """
    result = {
        "is_valid": False,
        "shoulder_angle": None,
        "hip_angle": None,
        "rotation_diff": None,
        "level": None,
        "advice": None,
        "missing_points": []
    }
    
    left_shoulder = _get_point(frame, "left_shoulder")
    right_shoulder = _get_point(frame, "right_shoulder")
    left_hip = _get_point(frame, "left_hip")
    right_hip = _get_point(frame, "right_hip")
    
    # 檢查哪些點缺失
    missing = []
    if left_shoulder is None: missing.append("left_shoulder")
    if right_shoulder is None: missing.append("right_shoulder")
    if left_hip is None: missing.append("left_hip")
    if right_hip is None: missing.append("right_hip")
    
    # 如果有任何點缺失，判定為側身嚴重不足
    if missing:
        result.update({
            "is_valid": True,
            "missing_points": missing,
            "level": 3,
            "advice": "拉拍側身嚴重不足夠，建議擊球前拉拍準備要轉動側身"
        })
        return result
    
    # 計算肩線和腰線在XZ平面的角度
    shoulder_angle = _calculate_angle_xz(left_shoulder, right_shoulder)
    hip_angle = _calculate_angle_xz(left_hip, right_hip)
    
    # 角度差（處理跨越180度的情況）
    rotation_diff = abs(shoulder_angle - hip_angle)
    if rotation_diff > 180:
        rotation_diff = 360 - rotation_diff
    
    # 判斷側身程度（角度差越小 = 越平行 = 側身越完整）
    rd = round(rotation_diff, 1)
    rot_range = pro_ranges.get("rotation_diff", {}) if pro_ranges else {}
    if rot_range and "p10" in rot_range and "p90" in rot_range:
        # 與 pro 比較：在 P10～P90 內 = 得宜，> P90 = 不足
        r_p50 = rot_range.get("p50", (rot_range["p10"] + rot_range["p90"]) / 2)
        diff_deg = round(rotation_diff - r_p50, 1)
        if rotation_diff <= rot_range["p90"]:
            level = 1
            advice = "拉拍側身完整"
        elif rotation_diff <= rot_range["p90"] * 1.5:  # 略超出
            level = 2
            advice = f"拉拍側身略不足夠（與專家中位數差 {diff_deg}°，需再減少），建議擊球前拉拍準備要轉動側身"
        else:
            level = 3
            advice = f"拉拍側身嚴重不足夠（與專家中位數差 {diff_deg}°，需再減少），建議擊球前拉拍準備要轉動側身"
    else:
        level = 0
        advice = f"拉拍側身：肩腰角度差 {rd}°（無專家數據，無法比較）"
    
    result.update({
        "is_valid": True,
        "shoulder_angle": float(shoulder_angle),
        "hip_angle": float(hip_angle),
        "rotation_diff": float(rotation_diff),
        "level": level,
        "advice": advice
    })
    
    return result


# ========== B. 右手腕高度分析 ==========
def _analyze_wrist_height(frame: Dict) -> Dict:
    """
    分析右手腕高度
    手腕要在「右肩 Y ~ 肩腰中點 Y」區間內才算適當
    （Y越大=越低）
    """
    result = {
        "is_valid": False,
        "wrist_y": None,
        "shoulder_y": None,
        "shoulder_hip_mid_y": None,
        "level": None,
        "advice": None
    }
    
    right_wrist = _get_point(frame, "right_wrist")
    right_shoulder = _get_point(frame, "right_shoulder")
    right_hip = _get_point(frame, "right_hip")
    
    if any(p is None for p in [right_wrist, right_shoulder, right_hip]):
        return result
    
    wrist_y = right_wrist[1]
    shoulder_y = right_shoulder[1]
    shoulder_hip_mid_y = (right_shoulder[1] + right_hip[1]) / 2
    
    # Y越大=越低
    # 適當範圍：shoulder_y <= wrist_y <= shoulder_hip_mid_y
    # （手腕在肩膀高度到肩腰中點之間）
    
    wy, sy, shy = round(wrist_y, 0), round(shoulder_y, 0), round(shoulder_hip_mid_y, 0)
    if wrist_y < shoulder_y:
        # 手腕比肩膀還高（Y較小）→ 高度過高，視為較嚴重
        level = 3
        advice = f"拉拍右手腕高度太高（手腕 Y={wy} mm、肩 Y={sy} mm，建議在肩～肩腰中點 {sy}～{shy} mm），建議拉拍準備時可以將手腕降低到胸部位置"
    elif wrist_y > shoulder_hip_mid_y:
        # 手腕比肩腰中點還低（Y較大）
        level = 2
        advice = f"拉拍右手腕高度太低（手腕 Y={wy} mm、肩腰中點 Y={shy} mm，建議在肩～肩腰中點之間），建議拉拍準備時可以將手腕舉到胸部位置"
    else:
        # 手腕在適當範圍內
        level = 1
        advice = "拉拍右手腕高度適當"
    
    result.update({
        "is_valid": True,
        "wrist_y": float(wrist_y),
        "shoulder_y": float(shoulder_y),
        "shoulder_hip_mid_y": float(shoulder_hip_mid_y),
        "level": level,
        "advice": advice
    })
    
    return result


# ========== C. 球拍頭朝向分析 ==========
def _analyze_paddle_direction(frame: Dict, pro_ranges: Optional[Dict] = None) -> Dict:
    """
    分析球拍頭朝向
    計算 paddle.top 到 paddle.bottom 的向量與水平線的夾角
    與 pro 專家 P10～P90 比較；無專家數據時回傳「無法比較」
    """
    result = {
        "is_valid": False,
        "paddle_angle": None,
        "level": None,
        "advice": None
    }
    
    paddle_top = _get_nested_point(frame, "paddle", "top")
    paddle_bottom = _get_nested_point(frame, "paddle", "bottom")
    
    if paddle_top is None or paddle_bottom is None:
        return result
    
    paddle_angle = _calculate_paddle_angle_from_frame(frame)
    if paddle_angle is None:
        return result
    
    pa = round(paddle_angle, 1)
    pad_range = pro_ranges.get("paddle_angle", {}) if pro_ranges else {}
    if pad_range and "p10" in pad_range and "p90" in pad_range:
        # 與 pro 比較：角度越大越好（拍頭朝上），在 P10～P90 內 = 得宜，< P10 = 太低
        p_p50 = pad_range.get("p50", (pad_range["p10"] + pad_range["p90"]) / 2)
        diff_deg = round(p_p50 - paddle_angle, 1)
        if paddle_angle >= pad_range["p10"]:
            level = 1
            advice = "拉拍球拍頭得宜"
        else:
            level = 2
            advice = f"拉拍掉下來，球拍頭太低（與專家中位數差 {diff_deg}°，需再抬高），建議拉拍準備時球拍頭可以指向天空。"
    else:
        level = 0
        advice = f"拉拍球拍頭：球拍角度 {pa}°（無專家數據，無法比較）"
    
    result.update({
        "is_valid": True,
        "paddle_angle": paddle_angle,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== D. 拉拍準備時機分析 ==========
def _analyze_backswing_timing(frames: List[Dict], bounce_idx: int) -> Dict:
    """
    分析拉拍準備時機
    球落地彈起時，手腕要在身體的右後方（相對於身體朝向）
    """
    result = {
        "is_valid": False,
        "wrist_behind": None,
        "paddle_behind": None,
        "level": None,
        "advice": None
    }
    
    if bounce_idx is None or bounce_idx >= len(frames):
        return result
    
    frame = frames[bounce_idx]
    
    right_wrist = _get_point(frame, "right_wrist")
    right_shoulder = _get_point(frame, "right_shoulder")
    left_shoulder = _get_point(frame, "left_shoulder")
    
    if any(p is None for p in [right_wrist, right_shoulder, left_shoulder]):
        return result
    
    # 1. 定義正右方（從左肩指向右肩的方向）
    body_right_3d = right_shoulder - left_shoulder
    body_right = np.array([body_right_3d[0], body_right_3d[2]])  # XZ平面投影
    body_right = body_right / (np.linalg.norm(body_right) + 1e-6)  # 正規化
    
    # 2. 定義正前方（從胸口指出去的方向）- 將右方逆時針旋轉90度
    # 在 XZ 平面：如果右方是 [x, z]，逆時針90度是 [-z, x]
    body_forward_xz = np.array([-body_right[1], body_right[0]])
    
    # 3. 身體後方 = 正前方的反方向
    body_backward = -body_forward_xz
    
    # 檢查手腕相對於右肩的位置
    wrist_relative = right_wrist - right_shoulder
    wrist_relative_xz = np.array([wrist_relative[0], wrist_relative[2]])
    
    # 檢查手腕位置
    wrist_right = np.dot(wrist_relative_xz, body_right) > 0
    wrist_back = np.dot(wrist_relative_xz, body_backward) > 0
    
    # 判斷拉拍準備時機（只根據手腕位置）
    if wrist_right and wrist_back:
        # 手腕在右後方（同時在右方且後方）
        level = 1  # 拉拍準備時間充分
        advice = "拉拍準備時間充分"
    elif wrist_right or wrist_back:
        # 手腕在右方或後方（只滿足其中一個條件）
        level = 2  # 拉拍有點慢
        advice = "拉拍有點慢，建議提早在球落地彈起時做好拉拍準備動作。"
    else:
        # 手腕既不在右方也不在後方（兩個條件都不滿足）
        level = 3  # 拉拍過慢
        advice = "拉拍過慢，建議提早在球落地彈起時做好拉拍準備動作。"
    
    result.update({
        "is_valid": True,
        "wrist_right": wrist_right,
        "wrist_back": wrist_back,
        "wrist_behind": wrist_right and wrist_back,  # 在右後方（同時在右方且後方）
        "paddle_behind": None,  # 不再檢查球拍位置
        "level": level,
        "advice": advice
    })
    
    return result


# ========== 主分析函式 ==========
def analyze_backswing(trajectory_data, knn_dataset_path: str = None, expert_filename: str = None) -> Tuple[str, float]:
    """
    拉拍分析（v4: 四點獨立分析版）
    
    四個獨立分析點：
    A. 側身完整度
    B. 右手腕高度
    C. 球拍頭朝向
    D. 拉拍準備時機
    
    Args:
        trajectory_data: 軌跡數據（可以是路徑或數據字典）
        knn_dataset_path: KNN數據集路徑（可選）
        expert_filename: 專家文件名（可選）
    
    Returns:
        (建議文字, 信心度)
    """
    try:
        # 處理輸入資料格式
        if isinstance(trajectory_data, str):
            frames_data = load_json(trajectory_data)
        elif isinstance(trajectory_data, dict):
            frames_data = trajectory_data.get("data", trajectory_data)
        else:
            frames_data = trajectory_data
        
        if not isinstance(frames_data, list) or len(frames_data) == 0:
            return "軌跡數據格式不正確", 0.0, None
        
        # 找出關鍵幀
        impact_idx = _find_impact_frame(frames_data)
        bounce_idx = _find_ball_bounce_frame(frames_data, impact_idx)
        
        # 決定分析幀（優先使用拉拍準備好的幀）
        backswing_ready_idx = _find_backswing_ready_frame(frames_data, bounce_idx, impact_idx)
        
        if backswing_ready_idx is not None:
            analysis_frame_idx = backswing_ready_idx  # 拉拍準備好的幀（手腕最高點+最遠點）
        elif bounce_idx is not None:
            analysis_frame_idx = bounce_idx  # 備用：球落地幀
        else:
            # 如果連球落地幀都找不到，返回錯誤
            return "無法找到合適的分析幀（需要球落地幀或拉拍準備好的幀）", 0.0, None
        
        analysis_frame = frames_data[analysis_frame_idx]
        
        # 載入 pro 範圍（A、C 與專家比較）
        pro_ranges = _calculate_pro_backswing_ranges(knn_dataset_path) if knn_dataset_path else {}
        
        # ========== 四點獨立分析 ==========
        
        # A. 側身完整度（與 pro 比較）
        rotation_result = _analyze_body_rotation(analysis_frame, pro_ranges)
        
        # B. 右手腕高度
        wrist_height_result = _analyze_wrist_height(analysis_frame)
        
        # C. 球拍頭朝向（與 pro 比較）
        paddle_result = _analyze_paddle_direction(analysis_frame, pro_ranges)
        
        # D. 拉拍準備時機
        timing_result = _analyze_backswing_timing(frames_data, bounce_idx)
        
        # ========== 組合建議 ==========
        advice_parts = []
        
        # A. 側身建議
        if rotation_result["is_valid"]:
            advice_parts.append(f"A.側身:{rotation_result['advice']}")
        else:
            advice_parts.append("A.側身:數據不足")
        
        # B. 手腕高度建議
        if wrist_height_result["is_valid"]:
            advice_parts.append(f"B.手腕高度:{wrist_height_result['advice']}")
        else:
            advice_parts.append("B.手腕高度:數據不足")
        
        # C. 球拍朝向建議
        if paddle_result["is_valid"]:
            advice_parts.append(f"C.球拍朝向:{paddle_result['advice']}")
        else:
            advice_parts.append("C.球拍朝向:數據不足")
        
        # D. 準備時機建議
        if timing_result["is_valid"]:
            advice_parts.append(f"D.準備時機:{timing_result['advice']}")
        else:
            advice_parts.append("D.準備時機:數據不足")
        
        # 各子項以「。」結尾後再銜接下一項
        def _end_period(s):
            return s if s.rstrip().endswith("。") else s + "。"
        combined_advice = "".join(_end_period(p) for p in advice_parts)
        
        # 計算信心度 (基於 Level)
        # Level 1 = 1.0 (100分), Level 2 = 0.8 (80分), Level 3 = 0.6 (60分), Invalid = 0.0
        score_map = {1: 1.0, 2: 0.8, 3: 0.6, 0: 0.0, None: 0.0}
        
        # 收集個別項目的 Level
        levels = {
            "拉拍側身": rotation_result.get("level"),
            "拉拍手腕高度": wrist_height_result.get("level"),
            "拉拍球拍朝向": paddle_result.get("level"),
            "拉拍準備時機": timing_result.get("level")
        }
        
        scores = [score_map.get(lvl, 0.0) for lvl in levels.values()]
        
        # 找出優先改善項目 (Level 3 > Level 2)
        priority_item = None
        max_level = 0
        
        for name, lvl in levels.items():
            if lvl and lvl > max_level:
                max_level = lvl
                # 暫存這個等級的第一個遇到的項目
                priority_item = name
            elif lvl and lvl == max_level and max_level >= 2:
                # 同等級不覆蓋，保留第一個
                pass
        
        if max_level < 2:
            priority_item = None  # Level 1 或沒有資料，不需改善
            
        # 如果所有項目都無效，信心度為 0
        if all(s == 0.0 for s in scores) and valid_count == 0:
            confidence = 0.0
        else:
            # 取平均 (總分 / 項目數)
            confidence = sum(scores) / 4.0
        
        return combined_advice, confidence, priority_item
        
    except Exception as e:
        print(f"拉拍分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return f"拉拍分析失敗: {str(e)}", 0.0, None


# ========== 詳細分析函式（返回完整結果） ==========
def analyze_backswing_detailed(trajectory_data, knn_dataset_path: str = None, expert_filename: str = None) -> Dict:
    """
    拉拍詳細分析，返回四個分析點的完整結果
    
    Returns:
        包含四個分析點詳細結果的字典
    """
    try:
        # 處理輸入資料格式
        if isinstance(trajectory_data, str):
            frames_data = load_json(trajectory_data)
        elif isinstance(trajectory_data, dict):
            frames_data = trajectory_data.get("data", trajectory_data)
        else:
            frames_data = trajectory_data
        
        if not isinstance(frames_data, list) or len(frames_data) == 0:
            return {"error": "軌跡數據格式不正確"}
        
        # 找出關鍵幀
        impact_idx = _find_impact_frame(frames_data)
        bounce_idx = _find_ball_bounce_frame(frames_data, impact_idx)
        
        # 決定分析幀（優先使用拉拍準備好的幀）
        backswing_ready_idx = _find_backswing_ready_frame(frames_data, bounce_idx, impact_idx)
        
        if backswing_ready_idx is not None:
            analysis_frame_idx = backswing_ready_idx  # 拉拍準備好的幀（手腕最高點+最遠點）
        elif bounce_idx is not None:
            analysis_frame_idx = bounce_idx  # 備用：球落地幀
        else:
            # 如果連球落地幀都找不到，返回錯誤
            return {"error": "無法找到合適的分析幀（需要球落地幀或拉拍準備好的幀）"}
        
        analysis_frame = frames_data[analysis_frame_idx]
        pro_ranges = _calculate_pro_backswing_ranges(knn_dataset_path) if knn_dataset_path else {}
        
        return {
            "analysis_frame_idx": analysis_frame_idx,
            "backswing_ready_idx": backswing_ready_idx,
            "impact_idx": impact_idx,
            "bounce_idx": bounce_idx,
            "pro_ranges": pro_ranges,
            "A_body_rotation": _analyze_body_rotation(analysis_frame, pro_ranges),
            "B_wrist_height": _analyze_wrist_height(analysis_frame),
            "C_paddle_direction": _analyze_paddle_direction(analysis_frame, pro_ranges),
            "D_backswing_timing": _analyze_backswing_timing(frames_data, bounce_idx)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ========== 測試用 ==========
if __name__ == "__main__":
    test_data_path = "trajectory/newtest_123/測試者2__1(3D_trajectory_smoothed).json"
    
    try:
        suggestion, confidence = analyze_backswing(test_data_path)
        print(f"建議: {suggestion}")
        print(f"信心度: {confidence:.2f}")
        
        print("\n詳細分析結果:")
        detailed = analyze_backswing_detailed(test_data_path)
        for key, value in detailed.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
