"""
擊球後收拍分析模組（Follow-through）
--------------------------------------------------------
分析時間範圍：擊球幀 → 右手肘離左肩最近的那一幀（收拍完成幀）

終點定義：右手肘與左肩的 3D 距離最小的那一幀
  - 比「最後一幀」更合理，因每隻影片長度不同
  - 收拍頂點即手臂越過身體、手肘最接近左肩的時刻

分析點：
- 擊球後右手腕往左肩方向移動
- 擊球後右手腕移動到接近左肩高度（使用專家 P10~P90 範圍）
- 球拍頭是否在終點朝下（拍頭 Y > 拍尾 Y）
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


def _calculate_3d_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """計算兩點間的 3D 距離"""
    return float(np.linalg.norm(p2 - p1))


def _get_nested_point(frame: Dict, *keys) -> Optional[np.ndarray]:
    """從巢狀結構中取點（如 paddle.top）"""
    p = frame
    for k in keys:
        p = p.get(k) if isinstance(p, dict) else None
        if p is None:
            return None
    x, y, z = p.get("x"), p.get("y"), p.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def _is_paddle_head_down(frame: Dict) -> Optional[bool]:
    """
    判斷球拍頭是否朝下（簡單方法：直接比較 Y 值）
    在座標系統中：Y 越大 = 越低
    如果 paddle.top.y > paddle.bottom.y，表示拍頭比拍尾低，即拍頭朝下
    """
    paddle_top = _get_nested_point(frame, "paddle", "top")
    paddle_bottom = _get_nested_point(frame, "paddle", "bottom")
    if paddle_top is None or paddle_bottom is None:
        return None
    # 拍頭 Y 值 > 拍尾 Y 值 = 拍頭朝下
    return paddle_top[1] > paddle_bottom[1]


def _find_impact_frame(frames: List[Dict]) -> Optional[int]:
    """找出擊球幀的索引"""
    for i, frame in enumerate(frames):
        if frame.get("tennis_ball_hit"):
            return i
    return None


def _find_end_frame_by_elbow_closest_to_shoulder(frames: List[Dict], impact_idx: int) -> int:
    """
    找出右手肘離左肩最近的那一幀，作為收拍完成幀（終點）。
    搜尋範圍：擊球幀之後的所有幀。
    若無有效點位則回退到最後一幀。
    """
    min_dist = float("inf")
    best_idx = len(frames) - 1
    for i in range(impact_idx, len(frames)):
        frame = frames[i]
        elbow = _get_point(frame, "right_elbow")
        shoulder = _get_point(frame, "left_shoulder")
        if elbow is not None and shoulder is not None:
            d = _calculate_3d_distance(elbow, shoulder)
            if d < min_dist:
                min_dist = d
                best_idx = i
    return best_idx


# ========== 專家數據統計 ==========
def _calculate_pro_height_ranges(knn_dataset: List[Dict]) -> Dict:
    """
    從 knn_dataset_new.json 計算 pro 專家的高度範圍（P10/P50/P90）
    收集每位 pro 的「右手肘離左肩最近幀」中，終點手腕 Y 與左肩 Y 的差值
    """
    pro_data = [d for d in knn_dataset if d.get("level") == "pro"]
    
    height_diffs = []
    
    for expert in pro_data:
        frames = expert.get("data", [])
        if not frames:
            continue
        
        # 找擊球幀
        impact_idx = _find_impact_frame(frames)
        if impact_idx is None:
            continue
        
        # 使用右手肘離左肩最近幀作為終點
        end_idx = _find_end_frame_by_elbow_closest_to_shoulder(frames, impact_idx)
        end_frame = frames[end_idx]
        
        # 取得終點數據
        end_wrist = _get_point(end_frame, "right_wrist")
        end_left_shoulder = _get_point(end_frame, "left_shoulder")
        
        if end_wrist is not None and end_left_shoulder is not None:
            # 計算高度差（絕對值）
            height_diff = abs(end_wrist[1] - end_left_shoulder[1])
            height_diffs.append(height_diff)
    
    # 計算分位數
    result = {
        "height_diff": {},
        "sample_count": len(pro_data)
    }
    
    if height_diffs:
        result["height_diff"] = {
            "p10": float(np.percentile(height_diffs, 10)),
            "p50": float(np.percentile(height_diffs, 50)),
            "p90": float(np.percentile(height_diffs, 90))
        }
    
    return result


# ========== 收拍動作分析 ==========
def _analyze_followthrough(start_frame: Dict, end_frame: Dict, pro_ranges: Dict = None) -> Dict:
    """
    分析收拍動作
    1. 右手腕是否往左肩方向移動（3D 距離變近）
    2. 右手腕是否移動到左肩高度附近（用左肩 Y，不再依賴眼睛）
    3. 球拍頭是否在終點朝下（拍頭 Y > 拍尾 Y）
    """
    result = {
        "is_valid": False,
        "start_wrist_to_shoulder_dist": None,
        "end_wrist_to_shoulder_dist": None,
        "moved_toward_shoulder": None,
        "end_wrist_y": None,
        "target_shoulder_y": None,
        "height_diff": None,
        "height_ok": None,
        "paddle_head_down": None,
        "level": None,
        "advice": None
    }
    
    # 起點數據
    start_wrist = _get_point(start_frame, "right_wrist")
    start_left_shoulder = _get_point(start_frame, "left_shoulder")
    
    # 終點數據
    end_wrist = _get_point(end_frame, "right_wrist")
    end_left_shoulder = _get_point(end_frame, "left_shoulder")
    
    # 球拍（只需要終點）：判斷拍頭是否朝下
    paddle_head_down = _is_paddle_head_down(end_frame)
    
    # 檢查必要點位
    if any(p is None for p in [start_wrist, start_left_shoulder, 
                                end_wrist, end_left_shoulder]):
        return result
    
    # 1. 方向判斷：右手腕到左肩的 3D 距離是否變近
    start_dist = _calculate_3d_distance(start_wrist, start_left_shoulder)
    end_dist = _calculate_3d_distance(end_wrist, end_left_shoulder)
    moved_toward_shoulder = end_dist < start_dist
    
    # 2. 高度判斷：終點手腕 Y ≈ 左肩 Y（使用專家 P10~P90 範圍）
    end_wrist_y = end_wrist[1]
    target_shoulder_y = end_left_shoulder[1]
    height_diff = abs(end_wrist_y - target_shoulder_y)
    
    # 使用專家範圍判斷
    if pro_ranges and pro_ranges.get("height_diff"):
        height_range = pro_ranges["height_diff"]
        height_ok = height_range["p10"] <= height_diff <= height_range["p90"]
    else:
        # 如果沒有專家數據，回退到固定閾值
        height_ok = height_diff < 150
    
    # 3. 球拍頭路徑：終點拍頭朝下
    # 拍頭 Y > 拍尾 Y = 拍頭朝下（已在上面計算）
    
    # 判斷等級與評語（依說明文件評語建議表格）
    # 在專家範圍內只顯示使用者值；超出範圍則加 P50 差值
    sd, ed = round(start_dist, 0), round(end_dist, 0)
    hd = round(height_diff, 0)
    hr = pro_ranges.get("height_diff", {}) if pro_ranges else {}
    h_p50 = hr.get("p50") if hr else None
    dist_str = f"（手腕～左肩距離 起點 {sd} → 終點 {ed} mm"
    # 在專家區間內不秀數值；不在區間則明確寫出與 P50 的差值
    if height_ok:
        height_str = f"；高度差 {hd} mm）"
    elif h_p50 is not None:
        diff_h = round(height_diff - h_p50, 0)
        height_str = f"；與專家中位數差 {diff_h} mm）"
    else:
        height_str = f"；高度差 {hd} mm）"
    num_suffix = dist_str + height_str
    
    if moved_toward_shoulder and height_ok and (paddle_head_down in [None, True]):
        level = 1
        advice = "擊球後收拍動作得宜"
    else:
        level = 2
        # 依 11 種 level 2 情境給予對應建議
        if moved_toward_shoulder and height_ok and paddle_head_down is False:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時球拍頭往左肩後方朝下。"
        elif moved_toward_shoulder and not height_ok and paddle_head_down is True:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕抬高至接近左肩高度。"
        elif moved_toward_shoulder and not height_ok and paddle_head_down is None:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕抬高至接近左肩高度。"
        elif moved_toward_shoulder and not height_ok and paddle_head_down is False:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕抬高至接近左肩高度，且球拍頭往左肩後方朝下。"
        elif not moved_toward_shoulder and height_ok and paddle_head_down is True:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕往左肩方向移動。"
        elif not moved_toward_shoulder and height_ok and paddle_head_down is None:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕往左肩方向移動。"
        elif not moved_toward_shoulder and height_ok and paddle_head_down is False:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕往左肩方向移動，且球拍頭往左肩後方朝下。"
        elif not moved_toward_shoulder and not height_ok and paddle_head_down is True:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕往左肩方向移動，並抬高至接近左肩高度。"
        elif not moved_toward_shoulder and not height_ok and paddle_head_down is None:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕和球拍往左肩後方甩動。"
        elif not moved_toward_shoulder and not height_ok and paddle_head_down is False:
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕和球拍往左肩後方甩動。"
        else:
            # 理論上不會進入， fallback
            advice = f"擊球後收拍動作不足{num_suffix}。建議收拍時右手手腕和球拍往左肩後方甩動。"
    
    result.update({
        "is_valid": True,
        "start_wrist_to_shoulder_dist": start_dist,
        "end_wrist_to_shoulder_dist": end_dist,
        "moved_toward_shoulder": moved_toward_shoulder,
        "end_wrist_y": float(end_wrist[1]),
        "target_shoulder_y": float(target_shoulder_y),
        "height_diff": float(height_diff),
        "height_ok": height_ok,
        "pro_height_range": pro_ranges.get("height_diff", {}) if pro_ranges else {},
        "paddle_head_down": paddle_head_down,
        "level": level,
        "advice": advice
    })
    
    return result


# ========== 主分析函式 ==========
def analyze_followthrough(trajectory_data, knn_dataset_path: str = "knn_dataset_new.json", expert_filename: str = None) -> Tuple[str, float]:
    """
    擊球後收拍分析（Follow-through）
    
    分析時間：擊球幀 → 右手肘離左肩最近幀
    
    分析點：
    - 右手腕往左肩方向移動 + 高度接近左肩（使用專家 P10~P90 範圍）
    
    Args:
        trajectory_data: 軌跡數據（可以是路徑或數據字典）
        knn_dataset_path: KNN數據集路徑（預設 knn_dataset_new.json）
        expert_filename: 專家文件名（可選，目前未使用）
    
    Returns:
        (建議文字, 信心度)
    """
    try:
        # 載入 KNN 數據集並計算專家範圍
        pro_ranges = None
        if knn_dataset_path:
            try:
                knn_dataset = load_json(knn_dataset_path)
                pro_ranges = _calculate_pro_height_ranges(knn_dataset)
            except Exception as e:
                print(f"無法載入專家數據: {e}")
        
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
            return "未找到擊球幀，無法進行收拍分析", 0.0, None
        
        # 起點：擊球幀，終點：右手肘離左肩最近幀
        start_frame = frames_data[impact_idx]
        end_idx = _find_end_frame_by_elbow_closest_to_shoulder(frames_data, impact_idx)
        end_frame = frames_data[end_idx]
        
        # 分析收拍動作
        result = _analyze_followthrough(start_frame, end_frame, pro_ranges)
        
        # 組合建議
        # 計算信心度 (基於 Level)
        # Level 1 = 1.0 (100分), Level 2 = 0.8 (80分), Level 3 = 0.6 (60分), Invalid = 0.0
        score_map = {1: 1.0, 2: 0.8, 3: 0.6, 0: 0.0, None: 0.0}
        
        priority_item = None
        
        if result["is_valid"]:
            advice = f"收拍:{result['advice']}"
            lvl = result.get("level")
            confidence = score_map.get(lvl, 0.0)
            
            # 設定優先改善項目
            if lvl and lvl >= 2:
                priority_item = "收拍動作"
        else:
            advice = "收拍:數據不足"
            confidence = 0.0
            priority_item = None
        
        return advice, confidence, priority_item
        
    except Exception as e:
        print(f"收拍分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return f"收拍分析失敗: {str(e)}", 0.0, None


# ========== 詳細分析函式 ==========
def analyze_followthrough_detailed(trajectory_data, knn_dataset_path: str = "knn_dataset_new.json", expert_filename: str = None) -> Dict:
    """
    擊球後收拍詳細分析，返回完整結果
    """
    try:
        # 載入 KNN 數據集並計算專家範圍
        pro_ranges = None
        if knn_dataset_path:
            try:
                knn_dataset = load_json(knn_dataset_path)
                pro_ranges = _calculate_pro_height_ranges(knn_dataset)
            except Exception as e:
                print(f"無法載入專家數據: {e}")
        
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
        
        start_frame = frames_data[impact_idx]
        end_idx = _find_end_frame_by_elbow_closest_to_shoulder(frames_data, impact_idx)
        end_frame = frames_data[end_idx]
        
        return {
            "impact_idx": impact_idx,
            "end_idx": end_idx,
            "pro_ranges": pro_ranges,
            "followthrough_analysis": _analyze_followthrough(start_frame, end_frame, pro_ranges)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ========== 測試用 ==========
if __name__ == "__main__":
    import os
    
    knn_path = "knn_dataset_new.json"
    if not os.path.exists(knn_path):
        print(f"找不到 {knn_path}")
        exit(1)
    
    with open(knn_path, 'r') as f:
        data = json.load(f)
    
    test_data = data[0]
    print(f"測試檔案: {test_data.get('filename')}")
    print(f"Level: {test_data.get('level')}")
    print()
    
    # 簡單輸出
    suggestion, confidence = analyze_followthrough(test_data)
    print(f"建議: {suggestion}")
    print(f"信心度: {confidence:.2f}")
    print()
    
    # 詳細輸出
    detailed = analyze_followthrough_detailed(test_data)
    for key, value in detailed.items():
        print(f"{key}: {value}")
