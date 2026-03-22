"""
擊球拍面角度分析模組（Racket-face）
--------------------------------------------------------
依分析說明/擊球拍面角度分析說明.md 實作

單一分析點：球過擊球點後飛出去的路徑向量與地平線夾角
（因擊球點常遺失球拍，以球軌跡作為拍面角度代理指標）

座標系：Y 軸向下為正
--------------------------------------------------------
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


# ========== 工具函式 ==========
def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_ball_point(frame: Dict) -> Optional[np.ndarray]:
    """從幀中提取 tennis_ball 座標 (x, y, z)"""
    p = frame.get("tennis_ball")
    if not p:
        return None
    x, y, z = p.get("x"), p.get("y"), p.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def _find_impact_frame(frames: List[Dict]) -> Optional[int]:
    """找出擊球幀索引"""
    for i, frame in enumerate(frames):
        if frame.get("tennis_ball_hit"):
            return i
    return None


# Y 軸向下為正：球往上飛時 y 減小，故 vertical = y_start - y_end（正值表示往上）
POST_IMPACT_FRAMES = 10


def _compute_flight_angle(
    ball_start: np.ndarray, ball_end: np.ndarray
) -> float:
    """
    計算球飛出軌跡與地平線（XZ 平面）的夾角（仰角）
    Y 軸向下為正：球往上飛時 y_end < y_start，vertical = y_start - y_end > 0
    回傳角度（度），正值 = 球往上飛 = 拍面較仰
    """
    dx = ball_end[0] - ball_start[0]
    dy = ball_end[1] - ball_start[1]
    dz = ball_end[2] - ball_start[2]
    horizontal = np.sqrt(dx * dx + dz * dz)
    if horizontal < 1e-6:
        return 0.0
    # Y 向下：球往上飛 → dy < 0，仰角應為正
    vertical = ball_start[1] - ball_end[1]
    angle_rad = np.arctan2(vertical, horizontal)
    return float(np.degrees(angle_rad))


# ========== 專家範圍計算 ==========
def _calculate_pro_flight_angles(knn_dataset: List[Dict]) -> Dict:
    """
    從 knn_dataset_new.json 的 pro 數據計算擊球後球飛出角度的 P10/P50/P90/P100
    """
    pro_data = [d for d in knn_dataset if d.get("level") == "pro"]
    flight_angles = []

    for expert in pro_data:
        frames = expert.get("data", [])
        if not frames:
            continue
        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            continue
        n = len(frames)
        end_idx = min(impact_idx + POST_IMPACT_FRAMES, n - 1)
        if end_idx <= impact_idx:
            continue
        ball_start = _get_ball_point(frames[impact_idx])
        if ball_start is None:
            for j in range(impact_idx + 1, end_idx + 1):
                ball_start = _get_ball_point(frames[j])
                if ball_start is not None:
                    impact_idx = j
                    break
        if ball_start is None:
            continue
        ball_end = _get_ball_point(frames[end_idx])
        if ball_end is None:
            for j in range(end_idx - 1, impact_idx, -1):
                ball_end = _get_ball_point(frames[j])
                if ball_end is not None:
                    break
        if ball_end is None:
            continue
        angle = _compute_flight_angle(ball_start, ball_end)
        flight_angles.append(angle)

    result = {"sample_count": len(flight_angles)}
    if flight_angles:
        result["flight_angle"] = {
            "p10": float(np.percentile(flight_angles, 10)),
            "p50": float(np.percentile(flight_angles, 50)),
            "p90": float(np.percentile(flight_angles, 90)),
            "p100": float(np.max(flight_angles)),
        }
    return result


# ========== 分析：球飛出角度 ==========
def _analyze_flight_angle(
    frames: List[Dict], impact_idx: int, pro_ranges: Dict
) -> Dict:
    """
    球飛出角度（拍面角度代理）
    判斷：≤P90 得宜；P90<x<P100 略仰；≥P100 過仰
    """
    n = len(frames)
    end_idx = min(impact_idx + POST_IMPACT_FRAMES, n - 1)
    if end_idx <= impact_idx:
        return {
            "is_valid": False,
            "flight_angle_deg": None,
            "level": None,
            "advice": "擊球後幀數不足，無法計算球飛出角度",
        }
    ball_start = _get_ball_point(frames[impact_idx])
    if ball_start is None:
        for j in range(impact_idx + 1, end_idx + 1):
            ball_start = _get_ball_point(frames[j])
            if ball_start is not None:
                impact_idx = j
                break
    if ball_start is None:
        return {
            "is_valid": False,
            "flight_angle_deg": None,
            "level": None,
            "advice": "擊球幀及擊球後球座標遺失，無法分析拍面角度",
        }
    ball_end = _get_ball_point(frames[end_idx])
    if ball_end is None:
        for j in range(end_idx - 1, impact_idx, -1):
            ball_end = _get_ball_point(frames[j])
            if ball_end is not None:
                break
    if ball_end is None:
        return {
            "is_valid": False,
            "flight_angle_deg": None,
            "level": None,
            "advice": "擊球後球座標遺失，無法分析拍面角度",
        }
    angle_deg = _compute_flight_angle(ball_start, ball_end)
    fr = pro_ranges.get("flight_angle", {})
    p50 = fr.get("p50")
    p90 = fr.get("p90")
    p100 = fr.get("p100")
    if p90 is None or p100 is None:
        return {
            "is_valid": True,
            "flight_angle_deg": float(angle_deg),
            "level": None,
            "advice": "專家數據不足，無法比較",
        }
    diff_deg = round(angle_deg - (p50 or p90), 1)
    diff_str = f"（球飛出角度 {angle_deg:.1f}°，與專家中位數差 {diff_deg:+.1f}°）" if (p50 is not None and diff_deg != 0) else ""
    if angle_deg <= p90:
        level = "a"
        advice = "擊球拍面得宜"
    elif angle_deg < p100:
        level = "b"
        advice = f"擊球拍面略仰{diff_str}，建議擊球時拍面要微往下壓"
    else:
        level = "c"
        advice = f"擊球拍面過仰{diff_str}，建議擊球時拍面要往下壓"
    return {
        "is_valid": True,
        "flight_angle_deg": float(angle_deg),
        "level": level,
        "advice": advice,
    }


# ========== 主分析函式 ==========
def analyze_racket_face(
    trajectory_data,
    knn_dataset_path: str = "knn_dataset_new.json",
    expert_filename: str = None,
) -> Tuple[str, float, Optional[str]]:
    """
    擊球拍面角度分析（Racket-face）

    以球過擊球點後飛出的軌跡向量與地平線夾角作為拍面角度代理，與 pro 比較。

    Args:
        trajectory_data: 軌跡數據（路徑或 dict 或 list of frames）
        knn_dataset_path: KNN 數據集路徑
        expert_filename: 專家文件名（本模組不依賴單一專家，保留參數以兼容整合分析）

    Returns:
        (建議文字, 信心度, 優先改善項目)
    """
    try:
        knn_dataset = load_json(knn_dataset_path)
        if not isinstance(knn_dataset, list):
            knn_dataset = knn_dataset.get("data", [])

        pro_ranges = _calculate_pro_flight_angles(knn_dataset)

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

        result = _analyze_flight_angle(frames, impact_idx, pro_ranges)
        if not result["is_valid"]:
            return result["advice"], 0.0, None

        level = result.get("level")
        if level == "a":
            confidence = 1.0
        elif level == "b":
            confidence = 0.7
        elif level == "c":
            confidence = 0.4
        else:
            # 專家數據不足，無法比較
            confidence = 0.5

        priority = None
        if level in ("b", "c"):
            priority = "擊球拍面角度"

        return result["advice"], confidence, priority
    except Exception as e:
        return f"擊球拍面角度分析失敗: {e}", 0.0, None


def analyze_racket_face_detailed(
    trajectory_data,
    knn_dataset_path: str = "knn_dataset_new.json",
    expert_filename: str = None,
) -> Dict:
    """
    擊球拍面角度分析詳細結果
    """
    try:
        knn_dataset = load_json(knn_dataset_path)
        if not isinstance(knn_dataset, list):
            knn_dataset = knn_dataset.get("data", [])

        pro_ranges = _calculate_pro_flight_angles(knn_dataset)

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

        result = _analyze_flight_angle(frames, impact_idx, pro_ranges)
        end_idx = min(impact_idx + POST_IMPACT_FRAMES, len(frames) - 1)
        return {
            "impact_idx": impact_idx,
            "post_impact_end": end_idx,
            "pro_ranges": pro_ranges,
            "flight_angle_deg": result.get("flight_angle_deg"),
            "level": result.get("level"),
            "advice": result.get("advice"),
            **result,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import os
    knn_path = "knn_dataset_new.json"
    traj_path = "trajectory/CHZ322__trajectory/trajectory_1/CHZ322__球1_segment(3D_trajectory_smoothed).json"
    if os.path.exists(knn_path) and os.path.exists(traj_path):
        suggestion, confidence, priority = analyze_racket_face(traj_path, knn_path)
        print(f"建議: {suggestion}")
        print(f"信心度: {confidence}")
        print(f"優先改善: {priority}")
        detailed = analyze_racket_face_detailed(traj_path, knn_path)
        print(json.dumps(detailed, ensure_ascii=False, indent=2))
    else:
        print("請提供 knn_dataset_new.json 與軌跡檔路徑進行測試")
