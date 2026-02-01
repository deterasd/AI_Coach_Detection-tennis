"""
擊球點區域分析（球 vs 專業分佈）
依據身體局部座標（以雙髖與踝建立）評估：
- 上下（height）：球在 up 軸的帶符號投影
- 前後（depth）：球在 forward 軸的帶符號投影
- 左右（lateral）：球在 right 軸的帶符號投影

標準化：以骨盆寬度（||right_hip - left_hip||）為尺度單位。
專業分佈：從 knn_dataset_new.json 篩選 level=="pro"，挑選近似擊球瞬間的代表幀（球不為空），計算各指標並取百分位區間（預設 P10-P90）。
"""

from typing import Dict, List, Optional, Tuple
import json
import numpy as np


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_point(frame: Dict, name: str) -> Optional[np.ndarray]:
    p = frame.get(name)
    if not p:
        return None
    x, y, z = p.get("x"), p.get("y"), p.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _build_local_axes(frame: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """
    使用髖與踝建立穩健的局部座標：
    - right_axis: 右髖 - 左髖（左右）
    - up_axis:   髖中心 - 踝中心（由下往上）
    - forward:   cross(up, right)（右手座標）
    返回: (hip_center, right_axis, up_axis, forward_axis, pelvis_width, body_height)
    若關鍵點不足返回 None
    - body_height: eye_center 到 ankle_center 的距離（作為「身高」用於標準化，只使用絕對值的點）
    """
    lh = _get_point(frame, "left_hip")
    rh = _get_point(frame, "right_hip")
    la = _get_point(frame, "left_ankle")
    ra = _get_point(frame, "right_ankle")
    le = _get_point(frame, "left_eye")
    re = _get_point(frame, "right_eye")
    
    if lh is None or rh is None or la is None or ra is None:
        return None
    # eye 是可選的，若不存在則回退到 hip_center
    if le is None or re is None:
        # 回退：使用 hip_center 作為身高參考點
        eye_center = None
    else:
        eye_center = (le + re) / 2.0

    hip_center = (lh + rh) / 2.0
    ankle_center = (la + ra) / 2.0

    right_axis = _safe_normalize(rh - lh)
    up_axis = _safe_normalize(hip_center - ankle_center)
    forward_axis = _safe_normalize(np.cross(up_axis, right_axis))

    pelvis_width = _norm(rh - lh)
    # 身高：使用 eye_center 到 ankle_center（更接近完整身高），若 eye 不存在則用 hip_center
    if eye_center is not None:
        body_height = _norm(eye_center - ankle_center)
    else:
        body_height = _norm(hip_center - ankle_center)  # 回退方案
    if pelvis_width <= 0 or body_height <= 0:
        return None

    return hip_center, right_axis, up_axis, forward_axis, pelvis_width, body_height


def _project_ball_metrics(frame: Dict) -> Optional[Dict[str, float]]:
    ball = _get_point(frame, "tennis_ball")
    axes = _build_local_axes(frame)
    if ball is None or axes is None:
        return None

    hip_center, right_axis, up_axis, fwd_axis, pelvis_width, body_height = axes
    rel = ball - hip_center

    # 簽名投影
    lateral = float(np.dot(rel, right_axis))
    height = float(np.dot(rel, up_axis))
    depth = float(np.dot(rel, fwd_axis))

    # 混合標準化：
    # - 左右（lateral）：用骨盆寬度（與左右維度直接相關）
    # - 上下（height）：用身高（eye_center 到 ankle_center，更接近完整身高）
    # - 前後（depth）：用身高（與身體比例更一致）
    lateral_n = lateral / pelvis_width
    height_n = height / body_height
    depth_n = depth / body_height

    return {
        "lateral": lateral,
        "height": height,
        "depth": depth,
        "lateral_n": lateral_n,
        "height_n": height_n,
        "depth_n": depth_n,
        "pelvis_width": pelvis_width,
        "body_height": body_height,
    }


def _find_user_impact_frame(frames: List[Dict]) -> Optional[Dict]:
    # 直接用標記的擊球幀
    for f in frames:
        if f.get("tennis_ball_hit"):
            return f
    return None


def _collect_pro_contact_metrics(knn_dataset: List[Dict]) -> List[Dict]:
    """
    從所有 pro 樣本收集「擊球幀(tennis_ball_hit==True)」的球座標：
    - 每位 pro 僅取其標記為擊球的那一幀（若有多幀為 True，取第一個）。
    - 忽略沒有球座標或關鍵點不足的幀。
    - 返回的字典包含 metrics 與 filename（用於 hover 顯示）。
    """
    results: List[Dict] = []
    for entry in knn_dataset:
        if not isinstance(entry, dict):
            continue
        if entry.get("level") != "pro":
            continue
        filename = entry.get("filename", "unknown")
        frames = entry.get("data", [])
        hit_frame = None
        for f in frames:
            if f.get("tennis_ball_hit"):
                hit_frame = f
                break
        if hit_frame is None:
            continue
        ball = hit_frame.get("tennis_ball") or {}
        if ball.get("x") is None:
            continue
        m = _project_ball_metrics(hit_frame)
        if m is not None:
            m_with_filename = m.copy()
            m_with_filename["filename"] = filename
            results.append(m_with_filename)
    return results


def _percentile_range(values: List[float], p_low: float = 10.0, p_high: float = 90.0) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    low = float(np.percentile(arr, p_low))
    high = float(np.percentile(arr, p_high))
    med = float(np.median(arr))
    return low, med, high


def _fmt_range_check(val: float, lo: float, med: float, hi: float, unit: str, name: str, axis_type: str) -> Tuple[bool, str]:
    """
    依軸向輸出更口語且明確的建議。
    在專家區間內：不秀數值。
    不在專家區間：明確寫出與 P50 的差值。
    """
    val_r = round(val, 2)
    diff_r = round(val - med, 2)
    unit_desc = "pw（骨盆寬度倍數）" if axis_type == "lateral" else "倍身高"
    if val < lo:
        diff_str = f"（與專家中位數差 {diff_r} {unit_desc}）"
        if axis_type == "height":
            return False, f"{name}偏低{diff_str}，建議在球上升過程中提早擊球，可再提高約{abs(diff_r)}倍身高。"
        elif axis_type == "depth":
            return False, f"{name}偏後{diff_str}，出拍太慢，建議提早迎前擊球，可再前移約{abs(diff_r)}倍身高。"
        elif axis_type == "lateral":
            return False, f"{name}太靠近身體{diff_str}，建議拉開揮拍距離約{abs(diff_r)} pw。"
        else:
            return False, f"{name}偏離理想區間{diff_str}，建議調整擊球點約{abs(diff_r)}標準化單位。"

    elif val > hi:
        diff_str = f"（與專家中位數差 {diff_r} {unit_desc}）"
        if axis_type == "height":
            return False, f"{name}偏高{diff_str}，建議擊球時重心降低，讓球下降一些再擊球，可再降低約{abs(diff_r)}倍身高。"
        elif axis_type == "depth":
            return False, f"{name}偏前{diff_str}，出拍太早，建議稍晚迎球，可再後移約{abs(diff_r)}倍身高。"
        elif axis_type == "lateral":
            return False, f"{name}太外側{diff_str}，建議靠近身體中心線約{abs(diff_r)} pw，保持身體平衡。"
        else:
            return False, f"{name}偏離理想區間{diff_str}，建議調整擊球點約{abs(diff_r)}標準化單位。"

    else:
        # 在專家區間內，不秀數值
        if axis_type == "height":
            return True, f"{name}在理想區間，擊球高度掌握良好。"
        elif axis_type == "depth":
            return True, f"{name}在理想區間，擊球時機掌握良好。"
        elif axis_type == "lateral":
            return True, f"{name}在理想區間，揮拍距離掌握良好。"
        else:
            return True, f"{name}在理想區間。"


def analyze_contact_zone(knn_dataset_path: str, trajectory_path: str, p_low: float = 10.0, p_high: float = 90.0) -> Dict:
    """
    主入口：
    - 從 knn_dataset_new.json 擷取 pro 的代表幀，計算分佈區間（混合標準化）。
    - 從使用者 3D 軌跡取擊球幀（tennis_ball_hit==True），計算三指標並與區間比較。
    - 標準化方式：
      * 左右（lateral_n）：用骨盆寬度（pw）
      * 上下（height_n）：用身高（eye_center 到 ankle_center，若 eye 不可用則回退到 hip_center 到 ankle_center）
      * 前後（depth_n）：用身高（同上）
    返回包含建議與統計的字典。
    """
    knn_dataset = _load_json(knn_dataset_path)
    user_data = _load_json(trajectory_path)
    frames = user_data if isinstance(user_data, list) else user_data.get("data", [])

    # 專業分佈（使用每位 pro 的擊球幀建立分佈）
    pro_metrics = _collect_pro_contact_metrics(knn_dataset)
    lateral_ns = [m["lateral_n"] for m in pro_metrics]
    height_ns = [m["height_n"] for m in pro_metrics]
    depth_ns = [m["depth_n"] for m in pro_metrics]

    lat_lo, lat_med, lat_hi = _percentile_range(lateral_ns, p_low, p_high)
    h_lo, h_med, h_hi = _percentile_range(height_ns, p_low, p_high)
    d_lo, d_med, d_hi = _percentile_range(depth_ns, p_low, p_high)

    # 使用者擊球幀
    impact = _find_user_impact_frame(frames)
    if impact is None:
        return {
            "feature": "contact_zone",
            "status": "no_impact_frame",
            "message": "未找到擊球幀(tennis_ball_hit)。",
        }

    user_m = _project_ball_metrics(impact)
    if user_m is None:
        return {
            "feature": "contact_zone",
            "status": "insufficient_keypoints",
            "message": "擊球幀球或髖/踝關鍵點缺失。",
        }

    # 區間檢查（用標準化單位）
    in_lat, lat_msg = _fmt_range_check(user_m["lateral_n"], lat_lo, lat_med, lat_hi, " pw", "左右位置", "lateral")
    in_h, h_msg = _fmt_range_check(user_m["height_n"], h_lo, h_med, h_hi, " pw", "上下高度", "height")
    in_d, d_msg = _fmt_range_check(user_m["depth_n"], d_lo, d_med, d_hi, " pw", "前後深度", "depth")

    advice_parts = [lat_msg, h_msg, d_msg]
    advice = "".join(advice_parts)

    return {
        "feature": "contact_zone",
        "status": "ok",
        "advice": advice,
        "user_values": {
            "lateral": user_m["lateral"],
            "height": user_m["height"],
            "depth": user_m["depth"],
            "lateral_n": user_m["lateral_n"],
            "height_n": user_m["height_n"],
            "depth_n": user_m["depth_n"],
            "pelvis_width": user_m["pelvis_width"],
            "body_height": user_m.get("body_height", 0.0),
        },
        "pro_ranges": {
            "lateral_n": {"p10": lat_lo, "p50": lat_med, "p90": lat_hi},
            "height_n": {"p10": h_lo, "p50": h_med, "p90": h_hi},
            "depth_n": {"p10": d_lo, "p50": d_med, "p90": d_hi},
        },
        "flags": {
            "lateral_in_range": in_lat,
            "height_in_range": in_h,
            "depth_in_range": in_d,
        },
    }


if __name__ == "__main__":
    # 簡單本地測試（需存在對應路徑）
    KN = "knn_dataset_new.json"
    TR = "trajectory/testing_123/testing_(3D_trajectory_smoothed).json"
    out = analyze_contact_zone(KN, TR)
    print(json.dumps(out, ensure_ascii=False, indent=2))
