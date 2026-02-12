"""
擊球前段出拍分析模組（Forward-swing）
--------------------------------------------------------
分析時間範圍：從拉拍完成（球落地彈起）→ 擊球前

三個分析點：
A. 手腕往下放 + 球拍頭朝後方（比較起點和終點）
B. 手腕向前帶動球拍（比較起點和終點）
C. 球拍頭朝下角度（只看終點）
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


def _calculate_paddle_angle(frame: Dict) -> Optional[float]:
    """
    計算球拍角度（A線與水平線的夾角）
    A線 = paddle.bottom → paddle.top 的向量
    正值 = 拍頭朝上，負值 = 拍頭朝下，0 = 水平
    """
    paddle_top = _get_nested_point(frame, "paddle", "top")
    paddle_bottom = _get_nested_point(frame, "paddle", "bottom")
    
    if paddle_top is None or paddle_bottom is None:
        return None
    
    # A線向量（從 bottom 到 top）
    paddle_vector = paddle_top - paddle_bottom
    
    # 計算與水平面的夾角
    # Y越大=越低，所以 paddle_top.Y < paddle_bottom.Y 表示拍頭朝上
    horizontal_dist = np.sqrt(paddle_vector[0]**2 + paddle_vector[2]**2)
    vertical_dist = -paddle_vector[1]  # 負號因為Y軸向下
    
    angle = float(np.degrees(np.arctan2(vertical_dist, horizontal_dist)))
    return angle


def _calculate_distance_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    計算點到線段的垂直距離（在3D空間）
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return np.linalg.norm(point_vec)
    
    # 使用外積計算距離
    cross = np.cross(point_vec, line_vec)
    distance = np.linalg.norm(cross) / line_len
    
    return float(distance)


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


# ========== A. 手腕下放 + 球拍角度趨近0° ==========
def _analyze_wrist_drop_and_paddle(start_frame: Dict, end_frame: Dict) -> Dict:
    """
    分析A：手腕往下放 + 球拍頭朝後方
    比較起點和終點：
    - 手腕 Y 變大（往下）
    - 球拍角度逐漸趨近 0°（角度絕對值變小，表示逐漸接近水平）
    """
    result = {
        "is_valid": False,
        "start_wrist_y": None,
        "end_wrist_y": None,
        "wrist_dropped": None,
        "start_paddle_angle": None,
        "end_paddle_angle": None,
        "paddle_leveled": None,
        "level": None,
        "advice": None
    }
    
    # 取得起點和終點的手腕位置
    start_wrist = _get_point(start_frame, "right_wrist")
    end_wrist = _get_point(end_frame, "right_wrist")
    
    if start_wrist is None or end_wrist is None:
        return result
    
    # 取得起點和終點的球拍角度
    start_paddle_angle = _calculate_paddle_angle(start_frame)
    end_paddle_angle = _calculate_paddle_angle(end_frame)
    
    if start_paddle_angle is None or end_paddle_angle is None:
        return result
    
    # 判斷手腕是否下放（Y 變大 = 往下）
    wrist_dropped = end_wrist[1] > start_wrist[1]
    
    # 判斷球拍角度是否趨近 0°（檢查變化趨勢：角度絕對值是否變小）
    # 從起點到終點，角度絕對值應該變小，表示逐漸接近水平（0°）
    # 這樣不會與 C 分析點衝突（C 要求終點角度在 -30° ~ -45°）
    paddle_leveled = abs(end_paddle_angle) < abs(start_paddle_angle)
    
    # 判斷等級
    spa, epa = round(start_paddle_angle, 1), round(end_paddle_angle, 1)
    if wrist_dropped and paddle_leveled:
        level = 1
        advice = "擊球出拍前球拍往下和拍頭倒下得宜"
    else:
        level = 2
        advice = f"擊球出拍前球拍下放和拍頭倒下不足（起點球拍 {spa}°、終點 {epa}°，應逐漸趨近水平 0°）。建議要出拍時，手腕在身體後方往下放。"
    
    result.update({
        "is_valid": True,
        "start_wrist_y": float(start_wrist[1]),
        "end_wrist_y": float(end_wrist[1]),
        "wrist_dropped": wrist_dropped,
        "start_paddle_angle": start_paddle_angle,
        "end_paddle_angle": end_paddle_angle,
        "paddle_leveled": paddle_leveled,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== B. 手腕向前帶動球拍 ==========
def _analyze_wrist_elbow_approach(start_frame: Dict, end_frame: Dict) -> Dict:
    """
    分析B：手腕和手肘向右肩腰線靠近
    比較起點和終點：手腕/手肘到「右肩-腰線」的距離變小
    """
    result = {
        "is_valid": False,
        "start_wrist_distance": None,
        "end_wrist_distance": None,
        "wrist_approached": None,
        "start_elbow_distance": None,
        "end_elbow_distance": None,
        "elbow_approached": None,
        "level": None,
        "advice": None
    }
    
    # 取得起點的點位
    start_wrist = _get_point(start_frame, "right_wrist")
    start_elbow = _get_point(start_frame, "right_elbow")
    start_shoulder = _get_point(start_frame, "right_shoulder")
    start_hip = _get_point(start_frame, "right_hip")
    
    # 取得終點的點位
    end_wrist = _get_point(end_frame, "right_wrist")
    end_elbow = _get_point(end_frame, "right_elbow")
    end_shoulder = _get_point(end_frame, "right_shoulder")
    end_hip = _get_point(end_frame, "right_hip")
    
    # 檢查必要點位
    if any(p is None for p in [start_wrist, start_shoulder, start_hip,
                                end_wrist, end_shoulder, end_hip]):
        return result
    
    # 計算手腕到肩腰線的距離
    start_wrist_dist = _calculate_distance_to_line(start_wrist, start_shoulder, start_hip)
    end_wrist_dist = _calculate_distance_to_line(end_wrist, end_shoulder, end_hip)
    wrist_approached = end_wrist_dist < start_wrist_dist
    
    # 計算手肘到肩腰線的距離（如果有手肘數據）
    elbow_approached = None
    start_elbow_dist = None
    end_elbow_dist = None
    
    if start_elbow is not None and end_elbow is not None:
        start_elbow_dist = _calculate_distance_to_line(start_elbow, start_shoulder, start_hip)
        end_elbow_dist = _calculate_distance_to_line(end_elbow, end_shoulder, end_hip)
        elbow_approached = end_elbow_dist < start_elbow_dist
    
    # 判斷等級
    swd, ewd = round(start_wrist_dist, 0), round(end_wrist_dist, 0)
    if wrist_approached and (elbow_approached is None or elbow_approached):
        level = 1
        advice = "出拍手腕向前帶動球拍得宜"
    else:
        level = 2
        advice = f"出拍手腕或手肘向前帶動不足（手腕到肩腰線距離 {swd} → {ewd} mm，應變小）。建議揮拍時手腕和手肘甩動到身體前方。"
    
    result.update({
        "is_valid": True,
        "start_wrist_distance": start_wrist_dist,
        "end_wrist_distance": end_wrist_dist,
        "wrist_approached": wrist_approached,
        "start_elbow_distance": start_elbow_dist,
        "end_elbow_distance": end_elbow_dist,
        "elbow_approached": elbow_approached,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== C. 球拍頭朝下角度 ==========
def _analyze_paddle_final_angle(end_frame: Dict) -> Dict:
    """
    分析C：球拍頭最終角度
    只看終點：球拍角度應在 -30° ~ -45° 範圍
    """
    result = {
        "is_valid": False,
        "paddle_angle": None,
        "in_range": None,
        "level": None,
        "advice": None
    }
    
    paddle_angle = _calculate_paddle_angle(end_frame)
    
    if paddle_angle is None:
        return result
    
    # 判斷角度是否在 -30° ~ -45° 範圍
    in_range = -45 <= paddle_angle <= -30
    pa = round(paddle_angle, 1)
    if in_range:
        level = 1
        advice = "出拍手腕向前時球拍頭朝後和朝下得宜"
    else:
        level = 2
        advice = f"出拍手腕向前時球拍頭朝後或朝下不足（終點球拍 {pa}°，理想 -30°～-45°）。建議出拍時球拍頭往下壓後再向前送。"
    
    result.update({
        "is_valid": True,
        "paddle_angle": paddle_angle,
        "in_range": in_range,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== 主分析函式 ==========
def analyze_forwardswing(trajectory_data, knn_dataset_path: str = None, expert_filename: str = None) -> Tuple[str, float]:
    """
    擊球前段出拍分析（Forward-swing）
    
    分析時間範圍：從球落地彈起 → 擊球前
    
    三個分析點：
    A. 手腕往下放 + 球拍頭朝後方
    B. 手腕向前帶動球拍
    C. 球拍頭朝下角度
    
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
        
        if impact_idx is None:
            return "未找到擊球幀，無法進行前段出拍分析", 0.0, None
        
        if bounce_idx is None:
            return "未找到球落地幀，無法進行前段出拍分析", 0.0, None
        
        # Forward-swing 區間：bounce_idx → impact_idx
        if bounce_idx >= impact_idx:
            return "球落地幀在擊球幀之後，數據異常", 0.0, None
        
        # 起點幀和終點幀
        start_frame = frames_data[bounce_idx]
        end_frame = frames_data[impact_idx - 1]  # 擊球前一幀
        
        # ========== 三個分析點 ==========
        
        # A. 手腕下放 + 球拍角度
        a_result = _analyze_wrist_drop_and_paddle(start_frame, end_frame)
        
        # B. 手腕向前帶動
        b_result = _analyze_wrist_elbow_approach(start_frame, end_frame)
        
        # C. 球拍最終角度
        c_result = _analyze_paddle_final_angle(end_frame)
        
        # ========== 組合建議 ==========
        advice_parts = []
        
        if a_result["is_valid"]:
            advice_parts.append(f"A.手腕下放:{a_result['advice']}")
        else:
            advice_parts.append("A.手腕下放:數據不足")
        
        if b_result["is_valid"]:
            advice_parts.append(f"B.向前帶動:{b_result['advice']}")
        else:
            advice_parts.append("B.向前帶動:數據不足")
        
        if c_result["is_valid"]:
            advice_parts.append(f"C.球拍角度:{c_result['advice']}")
        else:
            advice_parts.append("C.球拍角度:數據不足")
        
        combined_advice = "".join(advice_parts)
        
        # 計算信心度 (基於 Level)
        # Level 1 = 1.0 (100分), Level 2 = 0.8 (80分), Level 3 = 0.6 (60分), Invalid = 0.0
        score_map = {1: 1.0, 2: 0.8, 3: 0.6, 0: 0.0, None: 0.0}
        
        # 收集個別項目的 Level
        levels = {
            "出拍手腕下放": a_result.get("level"),
            "出拍向前帶動": b_result.get("level"),
            "出拍球拍角度": c_result.get("level")
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
            confidence = sum(scores) / 3.0
        
        return combined_advice, confidence, priority_item
        
    except Exception as e:
        print(f"前段出拍分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return f"前段出拍分析失敗: {str(e)}", 0.0, None


# ========== 詳細分析函式 ==========
def analyze_forwardswing_detailed(trajectory_data, knn_dataset_path: str = None, expert_filename: str = None) -> Dict:
    """
    擊球前段出拍詳細分析，返回三個分析點的完整結果
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
        
        if impact_idx is None:
            return {"error": "未找到擊球幀"}
        
        if bounce_idx is None:
            return {"error": "未找到球落地幀"}
        
        if bounce_idx >= impact_idx:
            return {"error": "球落地幀在擊球幀之後"}
        
        start_frame = frames_data[bounce_idx]
        end_frame = frames_data[impact_idx - 1]
        
        return {
            "bounce_idx": bounce_idx,
            "impact_idx": impact_idx,
            "start_frame_idx": bounce_idx,
            "end_frame_idx": impact_idx - 1,
            "A_wrist_drop_paddle": _analyze_wrist_drop_and_paddle(start_frame, end_frame),
            "B_wrist_elbow_approach": _analyze_wrist_elbow_approach(start_frame, end_frame),
            "C_paddle_final_angle": _analyze_paddle_final_angle(end_frame)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ========== 測試用 ==========
if __name__ == "__main__":
    test_data_path = "trajectory/newtest_123/測試者2__1(3D_trajectory_smoothed).json"
    
    try:
        suggestion, confidence = analyze_forwardswing(test_data_path)
        print(f"建議: {suggestion}")
        print(f"信心度: {confidence:.2f}")
        
        print("\n詳細分析結果:")
        detailed = analyze_forwardswing_detailed(test_data_path)
        for key, value in detailed.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
