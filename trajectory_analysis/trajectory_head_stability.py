"""
頭部穩定度分析模組（眼睛盯球）
--------------------------------------------------------
分析要點：
1. 右耳點與右腰位置平行移動
2. 擊球前至擊球出去後5幀，右耳與右眼線距離不變，直線水平角度不變
3. 成功擊球，擊球點在球拍上，沒有揮空拍或打框
--------------------------------------------------------
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


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


def _calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點間的距離"""
    return float(np.linalg.norm(p2 - p1))


def _calculate_angle_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點在水平面（XZ平面）的角度（度）"""
    dx = p2[0] - p1[0]  # x方向
    dz = p2[2] - p1[2]  # z方向
    angle = np.degrees(np.arctan2(dz, dx))
    return float(angle)


def _find_impact_frame(frames: List[Dict]) -> Optional[int]:
    """找出擊球幀的索引"""
    for i, frame in enumerate(frames):
        if frame.get("tennis_ball_hit"):
            return i
    return None


def _check_ball_contact(frames: List[Dict], impact_idx: int) -> Tuple[bool, str]:
    """
    檢查擊球品質。
    「是否擊中球」以 tennis_ball_hit 標記為主：僅在擊球幀無球資料、
    或擊球後球軌跡完全消失時才判定為未擊中球；不依速度變化推翻為未擊中球。
    返回: (是否成功擊球, 擊球狀態描述)
    """
    if impact_idx >= len(frames) or impact_idx < 0:
        return False, "無法判斷"

    impact_frame = frames[impact_idx]
    tennis_ball = _get_point(impact_frame, "tennis_ball")
    if tennis_ball is None:
        return False, "未擊中球"

    post_frames_end = min(len(frames) - 1, impact_idx + 3)
    post_ball_positions = []
    for i in range(impact_idx + 1, post_frames_end + 1):
        ball = _get_point(frames[i], "tennis_ball")
        if ball is not None:
            post_ball_positions.append(ball)

    if len(post_ball_positions) == 0:
        return False, "未擊中球"

    # 擊球幀有球且擊球後有軌跡 → 視為成功擊球
    return True, "成功擊球"


def _analyze_ear_hip_parallel_movement(frames: List[Dict], impact_idx: int, window: int = 10) -> Dict:
    """
    分析右耳點與右腰位置的平行移動
    參數:
    - frames: 所有幀數據
    - impact_idx: 擊球幀索引
    - window: 分析窗口大小（擊球前後各取window幀）
    """
    start_idx = max(0, impact_idx - window)
    end_idx = min(len(frames), impact_idx + window + 1)
    
    ear_y_positions = []
    hip_y_positions = []
    ear_hip_y_diff = []
    
    for i in range(start_idx, end_idx):
        right_ear = _get_point(frames[i], "right_ear")
        right_hip = _get_point(frames[i], "right_hip")
        
        if right_ear is not None and right_hip is not None:
            ear_y_positions.append(right_ear[1])  # Y座標（垂直方向）
            hip_y_positions.append(right_hip[1])
            ear_hip_y_diff.append(abs(right_ear[1] - right_hip[1]))
    
    if len(ear_y_positions) < 3:
        return {
            "is_valid": False,
            "parallel_score": 0.0,
            "ear_y_variance": float('inf'),
            "hip_y_variance": float('inf'),
            "diff_variance": float('inf')
        }
    
    # 計算Y座標的變異數（變異數小表示平行移動）
    ear_y_variance = float(np.var(ear_y_positions))
    hip_y_variance = float(np.var(hip_y_positions))
    diff_variance = float(np.var(ear_hip_y_diff))
    
    # 平行移動評分：變異數越小，評分越高
    # 標準化評分（假設變異數在0-100之間，可根據實際數據調整）
    max_variance = 100.0
    parallel_score = max(0.0, 1.0 - (ear_y_variance + hip_y_variance) / (2 * max_variance))
    
    return {
        "is_valid": True,
        "parallel_score": parallel_score,
        "ear_y_variance": ear_y_variance,
        "hip_y_variance": hip_y_variance,
        "diff_variance": diff_variance
    }


def _analyze_head_stability(frames: List[Dict], impact_idx: int, post_frames: int = 5) -> Dict:
    """
    分析擊球前至擊球出去後5幀，右耳與右眼線距離不變，直線水平角度不變
    參數:
    - frames: 所有幀數據
    - impact_idx: 擊球幀索引
    - post_frames: 擊球後分析的幀數
    """
    # 分析範圍：擊球前10幀到擊球後post_frames幀
    start_idx = max(0, impact_idx - 10)
    end_idx = min(len(frames), impact_idx + post_frames + 1)
    
    ear_eye_distances = []
    ear_eye_angles = []
    
    for i in range(start_idx, end_idx):
        right_ear = _get_point(frames[i], "right_ear")
        right_eye = _get_point(frames[i], "right_eye")
        
        if right_ear is not None and right_eye is not None:
            # 計算右耳與右眼的距離
            distance = _calculate_distance(right_ear, right_eye)
            ear_eye_distances.append(distance)
            
            # 計算右耳與右眼在水平面（XZ平面）的角度
            angle = _calculate_angle_2d(right_ear, right_eye)
            ear_eye_angles.append(angle)
    
    if len(ear_eye_distances) < 3:
        return {
            "is_valid": False,
            "distance_stability": 0.0,
            "angle_stability": 0.0,
            "distance_variance": float('inf'),
            "angle_variance": float('inf')
        }
    
    # 計算距離和角度的變異數
    distance_variance = float(np.var(ear_eye_distances))
    angle_variance = float(np.var(ear_eye_angles))
    
    # 穩定性評分：變異數越小，穩定性越高
    # 這裡先以距離/角度變異數的倒數形式建立 0~1 分數，之後再用 pro 的分佈做相對比較
    distance_stability = 1.0 / (1.0 + distance_variance)
    angle_stability = 1.0 / (1.0 + angle_variance)
    
    return {
        "is_valid": True,
        "distance_stability": distance_stability,
        "angle_stability": angle_stability,
        "distance_variance": distance_variance,
        "angle_variance": angle_variance,
        "avg_distance": float(np.mean(ear_eye_distances)),
        "avg_angle": float(np.mean(ear_eye_angles))
    }


def _compute_overall_stability(parallel_score: float, distance_stability: float, angle_stability: float) -> float:
    """綜合頭部穩定度分數（0~1），之後會拿去跟 pro 分佈比較。"""
    return float(parallel_score * 0.3 + distance_stability * 0.35 + angle_stability * 0.35)


_PRO_BASELINE_CACHE: Dict[str, Dict[str, float]] = {}


def _get_pro_head_stability_baseline(knn_dataset_path: Optional[str]) -> Optional[Dict[str, float]]:
    """
    從 knn_dataset_new.json 中擷取 level='pro' 的頭部穩定度分佈，
    回傳整體穩定度的 p25 / p50 做為建議分級基準。
    """
    if not knn_dataset_path:
        return None
    if knn_dataset_path in _PRO_BASELINE_CACHE:
        return _PRO_BASELINE_CACHE[knn_dataset_path]

    try:
        dataset = load_json(knn_dataset_path)
    except Exception as e:
        print(f"載入頭部穩定度 pro 基準失敗: {e}")
        return None

    pro_scores: List[float] = []
    # 為了效能，只抽樣前 N 個 pro 樣本
    MAX_PRO_SAMPLES = 200

    for item in dataset:
        if item.get("level") != "pro":
            continue
        frames = item.get("data") or item.get("trajectory") or []
        if not isinstance(frames, list) or not frames:
            continue
        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            continue

        parallel = _analyze_ear_hip_parallel_movement(frames, impact_idx)
        head = _analyze_head_stability(frames, impact_idx)
        if not parallel.get("is_valid") or not head.get("is_valid"):
            continue

        parallel_score = parallel.get("parallel_score", 0.0)
        distance_stability = head.get("distance_stability", 0.0)
        angle_stability = head.get("angle_stability", 0.0)
        overall = _compute_overall_stability(parallel_score, distance_stability, angle_stability)
        pro_scores.append(overall)

        if len(pro_scores) >= MAX_PRO_SAMPLES:
            break

    if not pro_scores:
        return None

    arr = np.array(pro_scores, dtype=float)
    baseline = {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }
    _PRO_BASELINE_CACHE[knn_dataset_path] = baseline
    return baseline


def analyze_head_stability(trajectory_data, knn_dataset_path: str = None, expert_filename: str = None) -> Tuple[str, float]:
    """
    分析頭部穩定度（眼睛盯球）
    
    Args:
        trajectory_data: 軌跡數據（可以是路徑或數據字典）
        knn_dataset_path: KNN數據集路徑（可選，目前未使用）
        expert_filename: 專家文件名（可選，目前未使用）
    
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
            return "軌跡數據格式不正確", 0.0
        
        # 找出擊球幀
        impact_idx = _find_impact_frame(frames_data)
        if impact_idx is None:
            return "未找到擊球幀，無法進行頭部穩定度分析", 0.0
        
        impact_frame = frames_data[impact_idx]
        
        # 1. 是否擊到球：以 tennis_ball_hit 標記為主；僅在擊球幀無球且擊球後也無軌跡時才判為未擊中
        ball_contact_success, ball_contact_status = _check_ball_contact(frames_data, impact_idx)
        if impact_frame.get("tennis_ball_hit") is True and _get_point(impact_frame, "tennis_ball") is not None:
            ball_contact_success = True  # pipeline 已標記擊中且該幀有球，視為有擊到球
        
        # 2. 分析右耳與右腰的平行移動
        parallel_analysis = _analyze_ear_hip_parallel_movement(frames_data, impact_idx)
        
        # 3. 分析頭部穩定性（右耳與右眼）
        head_stability = _analyze_head_stability(frames_data, impact_idx)
        
        # 依此次揮拍是否有擊到球給出建議
        if ball_contact_success:
            combined_advice = "頭部穩定眼睛有盯球。"
        else:
            combined_advice = "擊球過程中頭部略不穩定，建議揮拍過程中，眼睛盯好擊球點，並維持頭不轉動。"
        
        # 計算信心度
        confidence = 1.0
        if not parallel_analysis.get("is_valid", False):
            confidence *= 0.5
        if not head_stability.get("is_valid", False):
            confidence *= 0.5
        if impact_idx < 10:  # 擊球幀太早，數據可能不足
            confidence *= 0.8
        
        return combined_advice, confidence
        
    except Exception as e:
        print(f"頭部穩定度分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return f"頭部穩定度分析失敗: {str(e)}", 0.0


# ========== 測試用 ==========
if __name__ == "__main__":
    test_data_path = "trajectory/testing_123/testing_(3D_trajectory_smoothed).json"
    
    try:
        suggestion, confidence = analyze_head_stability(test_data_path)
        print(f"建議: {suggestion}")
        print(f"信心度: {confidence:.2f}")
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()

