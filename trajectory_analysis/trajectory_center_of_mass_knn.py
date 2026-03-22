"""
重心分析模組（Body Weight / Center of Mass）
--------------------------------------------------------
依分析說明/重心分析使用說明.md 實作

兩個分析點：
A. 擊球時膝蓋彎曲（重心高低）：與 pro P10/P90/P100 比較
B. 擊球後重心前移：自我比較（擊球前 vs 擊球後、左膝 vs 右膝）
--------------------------------------------------------
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


# ========== 工具函式 ==========
def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_point(frame: Dict, name: str) -> Optional[np.ndarray]:
    """從幀中提取關鍵點座標 (x, y, z)"""
    p = frame.get(name)
    if not p:
        return None
    x, y, z = p.get("x"), p.get("y"), p.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點 3D 距離"""
    return float(np.linalg.norm(p2 - p1))


def _knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    計算膝部內角度（hip-knee-ankle 三點夾角）
    回傳 0~180°，角度越大膝蓋越直、重心越高
    """
    a = _distance(knee, ankle)
    c = _distance(hip, knee)
    b = _distance(hip, ankle)
    if a < 1e-6 or c < 1e-6:
        return 180.0
    cos_val = (a * a + c * c - b * b) / (2 * a * c)
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.degrees(np.arccos(cos_val)))


def _find_impact_frame(frames: List[Dict]) -> Optional[int]:
    """找出擊球幀索引"""
    for i, frame in enumerate(frames):
        if frame.get("tennis_ball_hit"):
            return i
    return None


# ========== 專家範圍計算 ==========
def _calculate_pro_knee_ranges(knn_dataset: List[Dict]) -> Dict:
    """
    從 knn_dataset_new.json 的 pro 數據計算擊球幀膝部角度的 P10/P50/P90/P100
    取 max(左膝, 右膝) 作為該幀代表值
    """
    pro_data = [d for d in knn_dataset if d.get("level") == "pro"]
    knee_angles = []

    for expert in pro_data:
        frames = expert.get("data", [])
        if not frames:
            continue
        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            continue
        frame = frames[impact_idx]
        lh = _get_point(frame, "left_hip")
        rh = _get_point(frame, "right_hip")
        lk = _get_point(frame, "left_knee")
        rk = _get_point(frame, "right_knee")
        la = _get_point(frame, "left_ankle")
        ra = _get_point(frame, "right_ankle")
        if all(p is not None for p in [lh, lk, la]):
            left_angle = _knee_angle(lh, lk, la)
        else:
            left_angle = None
        if all(p is not None for p in [rh, rk, ra]):
            right_angle = _knee_angle(rh, rk, ra)
        else:
            right_angle = None
        if left_angle is not None and right_angle is not None:
            knee_angles.append(max(left_angle, right_angle))
        elif left_angle is not None:
            knee_angles.append(left_angle)
        elif right_angle is not None:
            knee_angles.append(right_angle)

    result = {"sample_count": len(knee_angles)}
    if knee_angles:
        result["knee_angle_impact"] = {
            "p10": float(np.percentile(knee_angles, 10)),
            "p50": float(np.percentile(knee_angles, 50)),
            "p90": float(np.percentile(knee_angles, 90)),
            "p100": float(np.max(knee_angles)),
        }
    return result


# ========== 分析點 A：擊球時膝蓋彎曲 ==========
def _analyze_knee_bend(
    impact_frame: Dict, pro_ranges: Dict
) -> Dict:
    """
    A. 擊球時膝蓋彎曲（重心高低）
    判斷：≤P90 得宜；P90<x<P100 略高；≥P100 過高
    """
    lh = _get_point(impact_frame, "left_hip")
    rh = _get_point(impact_frame, "right_hip")
    lk = _get_point(impact_frame, "left_knee")
    rk = _get_point(impact_frame, "right_knee")
    la = _get_point(impact_frame, "left_ankle")
    ra = _get_point(impact_frame, "right_ankle")

    left_angle = _knee_angle(lh, lk, la) if all(p is not None for p in [lh, lk, la]) else None
    right_angle = _knee_angle(rh, rk, ra) if all(p is not None for p in [rh, rk, ra]) else None

    if left_angle is None and right_angle is None:
        return {
            "is_valid": False,
            "knee_angle_impact": None,
            "level": None,
            "advice": "數據不足，無法分析擊球時膝蓋彎曲",
        }

    knee_angle = max(left_angle or 0, right_angle or 0) if (left_angle and right_angle) else (left_angle or right_angle)
    kr = pro_ranges.get("knee_angle_impact", {})
    p50 = kr.get("p50")
    p90 = kr.get("p90")
    p100 = kr.get("p100")

    if p90 is None or p100 is None:
        return {
            "is_valid": True,
            "knee_angle_impact": float(knee_angle),
            "level": None,
            "advice": "專家數據不足，無法比較",
        }

    diff_deg = round(knee_angle - (p50 or p90), 1)
    diff_str = f"（膝部角度 {knee_angle:.1f}°，與專家中位數差 {diff_deg:+.1f}°）" if (p50 is not None and diff_deg != 0) else ""

    if knee_angle <= p90:
        level = "a"
        advice = "擊球身體重心得宜"
    elif knee_angle < p100:
        level = "b"
        advice = f"擊球時整體重心略高{diff_str}，建議揮拍時可以微微屈膝"
    else:
        level = "c"
        advice = f"擊球時整體重心太高{diff_str}，建議揮拍時可以屈膝"

    return {
        "is_valid": True,
        "knee_angle_impact": float(knee_angle),
        "in_range": knee_angle <= p90,
        "level": level,
        "advice": advice,
    }


# ========== 分析點 B：擊球後重心前移 ==========
POST_IMPACT_FRAMES = 8


def _analyze_weight_shift(
    frames: List[Dict], impact_idx: int
) -> Dict:
    """
    B. 擊球後重心前移（自我比較）
    條件1: 右膝擊球後 > 擊球前
    條件2: 擊球後左膝 < 擊球後右膝
    """
    n = len(frames)
    if impact_idx < 0 or impact_idx >= n:
        return {"is_valid": False, "advice": "擊球幀索引無效"}

    def _get_knee_angles(idx: int) -> Tuple[Optional[float], Optional[float]]:
        f = frames[idx]
        lh, rh = _get_point(f, "left_hip"), _get_point(f, "right_hip")
        lk, rk = _get_point(f, "left_knee"), _get_point(f, "right_knee")
        la, ra = _get_point(f, "left_ankle"), _get_point(f, "right_ankle")
        left_a = _knee_angle(lh, lk, la) if all(p is not None for p in [lh, lk, la]) else None
        right_a = _knee_angle(rh, rk, ra) if all(p is not None for p in [rh, rk, ra]) else None
        return left_a, right_a

    # 擊球前：取擊球幀前數幀平均，不足則用擊球幀
    before_start = max(0, impact_idx - 5)
    before_end = impact_idx
    right_befores = []
    for i in range(before_start, before_end):
        _, ra = _get_knee_angles(i)
        if ra is not None:
            right_befores.append(ra)
    right_knee_before = float(np.mean(right_befores)) if right_befores else None

    # 擊球後：取擊球幀後數幀平均
    after_start = impact_idx + 1
    after_end = min(n, impact_idx + POST_IMPACT_FRAMES + 1)
    left_afters, right_afters = [], []
    for i in range(after_start, after_end):
        la, ra = _get_knee_angles(i)
        if la is not None:
            left_afters.append(la)
        if ra is not None:
            right_afters.append(ra)
    left_knee_after = float(np.mean(left_afters)) if left_afters else None
    right_knee_after = float(np.mean(right_afters)) if right_afters else None

    if right_knee_before is None:
        right_knee_before = right_knee_after
    if right_knee_before is None or right_knee_after is None or left_knee_after is None:
        return {
            "is_valid": False,
            "advice": "擊球前後膝蓋數據不足，無法分析重心前移",
        }

    cond1 = right_knee_after > right_knee_before
    cond2 = left_knee_after < right_knee_after

    if cond1 and cond2:
        level = "a"
        advice = "擊球時身體重心往前得宜"
    elif cond1 or cond2:
        level = "b"
        advice = "擊球時身體重心往前略不足，建議揮拍時重心要往左腳移動"
    else:
        level = "c"
        advice = "擊球時身體重心往前明顯不足，建議揮拍時重心要往左腳移動"

    return {
        "is_valid": True,
        "right_knee_before": right_knee_before,
        "right_knee_after": right_knee_after,
        "left_knee_after": left_knee_after,
        "cond1_ok": cond1,
        "cond2_ok": cond2,
        "level": level,
        "advice": advice,
    }


# ========== 主分析函式 ==========
def analyze_center_of_mass(
    trajectory_data,
    knn_dataset_path: str = "knn_dataset_new.json",
    expert_filename: str = None,
) -> Tuple[str, float, Optional[str]]:
    """
    重心分析（Body Weight）

    兩個分析點：
    A. 擊球時膝蓋彎曲（與 pro P10/P90/P100 比較）
    B. 擊球後重心前移（自我比較）

    Args:
        trajectory_data: 軌跡數據（路徑或 dict，或 list of frames）
        knn_dataset_path: KNN 數據集路徑
        expert_filename: 專家文件名（本模組不依賴單一專家，保留參數以兼容整合分析）

    Returns:
        (建議文字, 信心度, 優先改善項目)
    """
    try:
        knn_dataset = load_json(knn_dataset_path)
        if not isinstance(knn_dataset, list):
            knn_dataset = knn_dataset.get("data", [])

        pro_ranges = _calculate_pro_knee_ranges(knn_dataset)

        if isinstance(trajectory_data, str):
            traj = load_json(trajectory_data)
        elif isinstance(trajectory_data, dict):
            traj = trajectory_data
        else:
            traj = trajectory_data
        frames = traj if isinstance(traj, list) else traj.get("data", [])
        if not frames:
            return "軌跡數據為空", 0.0, None

        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            return "未找到擊球幀", 0.0, None

        impact_frame = frames[impact_idx]

        a_result = _analyze_knee_bend(impact_frame, pro_ranges)
        b_result = _analyze_weight_shift(frames, impact_idx)

        advice_parts = []
        if a_result["is_valid"]:
            advice_parts.append(f"A.膝蓋彎曲:{a_result['advice']}")
        else:
            advice_parts.append("A.膝蓋彎曲:數據不足")
        if b_result["is_valid"]:
            advice_parts.append(f"B.重心前移:{b_result['advice']}")
        else:
            advice_parts.append("B.重心前移:數據不足")

        # 各子項以「。」結尾後再銜接下一項
        def _end_period(s):
            return s if s.rstrip().endswith("。") else s + "。"
        combined = "".join(_end_period(p) for p in advice_parts)

        # 依 level 給分：a=1.0, b=0.7, c=0.4, 專家數據不足=0.5，無效=不計入
        level_to_score = {"a": 1.0, "b": 0.7, "c": 0.4, None: 0.5}
        scores = []
        for res in [a_result, b_result]:
            if res.get("is_valid"):
                lvl = res.get("level")
                scores.append(level_to_score.get(lvl, 0.5))
        confidence = sum(scores) / len(scores) if scores else 0.0

        priority = None
        level_priority = {"c": 2, "b": 1, "a": 0}
        for name, res in [("膝蓋彎曲", a_result), ("重心前移", b_result)]:
            if res.get("is_valid") and res.get("level") in ("b", "c"):
                if level_priority.get(res["level"], 0) >= 1:
                    priority = name
                    break

        return combined, confidence, priority
    except Exception as e:
        return f"重心分析失敗: {e}", 0.0, None


def analyze_center_of_mass_detailed(
    trajectory_data,
    knn_dataset_path: str = "knn_dataset_new.json",
    expert_filename: str = None,
) -> Dict:
    """
    重心分析詳細結果，返回兩個分析點與專家範圍
    """
    try:
        knn_dataset = load_json(knn_dataset_path)
        if not isinstance(knn_dataset, list):
            knn_dataset = knn_dataset.get("data", [])

        pro_ranges = _calculate_pro_knee_ranges(knn_dataset)

        if isinstance(trajectory_data, str):
            traj = load_json(trajectory_data)
        elif isinstance(trajectory_data, dict):
            traj = trajectory_data
        else:
            traj = trajectory_data
        frames = traj if isinstance(traj, list) else traj.get("data", [])
        if not frames:
            return {"error": "軌跡數據為空"}

        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            return {"error": "未找到擊球幀"}

        a_result = _analyze_knee_bend(frames[impact_idx], pro_ranges)
        b_result = _analyze_weight_shift(frames, impact_idx)

        return {
            "impact_idx": impact_idx,
            "post_impact_end": min(len(frames), impact_idx + POST_IMPACT_FRAMES + 1),
            "pro_ranges": pro_ranges,
            "A_knee_bend": a_result,
            "B_weight_shift": b_result,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import os
    knn_path = "knn_dataset_new.json"
    traj_path = "trajectory/CHZ322__trajectory/trajectory_1/CHZ322__球1_segment(3D_trajectory_smoothed).json"
    if os.path.exists(knn_path) and os.path.exists(traj_path):
        suggestion, confidence, priority = analyze_center_of_mass(traj_path, knn_path)
        print(f"建議: {suggestion}")
        print(f"信心度: {confidence}")
        print(f"優先改善: {priority}")
        detailed = analyze_center_of_mass_detailed(traj_path, knn_path)
        print(json.dumps(detailed, ensure_ascii=False, indent=2))
    else:
        print("請提供 knn_dataset_new.json 與軌跡檔路徑進行測試")
