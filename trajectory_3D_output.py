import numpy as np
import json
import time
import math
from scipy.interpolate import interp1d

class NanToNullEncoder(json.JSONEncoder):
    """自定義 JSON encoder，將 NaN 轉換為 null"""
    def encode(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 'null'
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        for chunk in super().iterencode(obj, _one_shot):
            yield chunk.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')

def triangulate_point(P1, P2, point1, point2):
    """使用兩台相機的投影矩陣與對應的 2D 點座標計算 3D 座標"""
    A = np.zeros((4, 4))
    A[0] = point1[1] * P1[2] - P1[1]
    A[1] = P1[0] - point1[0] * P1[2]
    A[2] = point2[1] * P2[2] - P2[1]
    A[3] = P2[0] - point2[0] * P2[2]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


# ✅ ② 改良版 fix_trajectory：使用 cubic interpolation（樣條插值）
def fix_trajectory(data):
    valid_idx = [i for i, f in enumerate(data) if f["tennis_ball"]["x"] is not None]
    if len(valid_idx) < 3:
        print("[WARNING] fix_trajectory: 球點不足或未偵測到球，略過修正")
        return data

    for axis in ["x", "y", "z"]:
        vals = [data[i]["tennis_ball"][axis] for i in valid_idx]
        interp = interp1d(valid_idx, vals, kind="cubic", fill_value="extrapolate")
        for i in range(valid_idx[0], valid_idx[-1]):
            data[i]["tennis_ball"][axis] = float(interp(i))

    print(f"[INFO] fix_trajectory: 修正完成 (frame {valid_idx[0]} → {valid_idx[-1]})")
    return data


# ✅ 以下為B版本的球拍幾何校正函數
def ensure_paddle_structure(data):
    """補充缺失的球拍結構點"""
    print("[INFO] 正在檢查並重建缺失的球拍結構...")
    for frame in data:
        if "paddle" not in frame: continue
        p = frame["paddle"]
        
        # 如果 grip_bottom 缺失，嘗試從 center 和 grip_top 推算
        if p.get("grip_bottom", {}).get("x") is None:
            gt = p.get("grip_top", {})
            c = p.get("center", {})
            
            if gt.get("x") is not None and c.get("x") is not None:
                vec = np.array([gt["x"]-c["x"], gt["y"]-c["y"], gt["z"]-c["z"]])
                extension_ratio = 0.8 
                gb = np.array([gt["x"], gt["y"], gt["z"]]) + vec * extension_ratio
                p["grip_bottom"] = {"x": float(gb[0]), "y": float(gb[1]), "z": float(gb[2])}

    return data


def fix_paddle_relative_to_wrist(data):
    """使用手腕軌跡帶動球拍軌跡"""
    paddle_parts = ["top", "bottom", "right", "left", "center", "grip_top", "grip_bottom"]
    wrist_key = "right_wrist" 
    
    # 先確保手腕有值 (對手腕做簡單線性插值)
    wrist_valid = [i for i, f in enumerate(data) if f.get(wrist_key, {}).get("x") is not None]

    if len(wrist_valid) > 1:
        for axis in ["x", "y", "z"]:
            vals = [data[i][wrist_key][axis] for i in wrist_valid]
            f_interp = interp1d(wrist_valid, vals, kind="linear", fill_value="extrapolate")
            for i in range(len(data)):
                if data[i].get(wrist_key, {}).get("x") is None:
                    if wrist_key not in data[i]: data[i][wrist_key] = {}
                    data[i][wrist_key][axis] = float(f_interp(i))

    # 開始修補球拍
    for part in paddle_parts:
        valid_indices = []
        offsets = {"x": [], "y": [], "z": []}
        
        for i, frame in enumerate(data):
            p_pt = frame.get("paddle", {}).get(part, {})
            w_pt = frame.get(wrist_key, {})
            
            if p_pt.get("x") is not None and w_pt.get("x") is not None:
                valid_indices.append(i)
                offsets["x"].append(p_pt["x"] - w_pt["x"])
                offsets["y"].append(p_pt["y"] - w_pt["y"])
                offsets["z"].append(p_pt["z"] - w_pt["z"])
        
        if len(valid_indices) < 2:
            continue
            
        interp_funcs = {}
        for axis in ["x", "y", "z"]:
            interp_funcs[axis] = interp1d(valid_indices, offsets[axis], kind="cubic", fill_value="extrapolate")
            
        start_f = valid_indices[0]
        end_f = valid_indices[-1]
        
        for i in range(start_f, end_f + 1):
            if data[i]["paddle"][part].get("x") is None:
                w_pt = data[i].get(wrist_key, {})
                if w_pt.get("x") is not None:
                    data[i]["paddle"][part]["x"] = w_pt["x"] + float(interp_funcs["x"](i))
                    data[i]["paddle"][part]["y"] = w_pt["y"] + float(interp_funcs["y"](i))
                    data[i]["paddle"][part]["z"] = w_pt["z"] + float(interp_funcs["z"](i))
    
    return data


def enforce_rigid_paddle_geometry(data):
    """強制剛體幾何校正 - Snap to Wrist + Rigid Body"""
    print("[INFO] 正在執行網球拍幾何校正 (Snap to Wrist + Rigid Body)...")

    # 網球拍標準尺寸 (單位: mm)
    HANDLE_LENGTH = 190.0
    NECK_LENGTH = 70.0
    FACE_LENGTH = 340.0
    FACE_WIDTH_HALF = 135.0
    
    DIST_TO_GRIP_TOP = HANDLE_LENGTH
    DIST_TO_BOTTOM = HANDLE_LENGTH + NECK_LENGTH
    DIST_TO_CENTER = HANDLE_LENGTH + NECK_LENGTH + (FACE_LENGTH / 2.0)
    DIST_TO_TOP = HANDLE_LENGTH + NECK_LENGTH + FACE_LENGTH

    wrist_key = "right_wrist"
    prev_u_right = None

    for frame in data:
        if "paddle" not in frame: continue
        p = frame["paddle"]
        w = frame.get(wrist_key, {})
        
        # 1. 決定錨點 -> 強制設為手腕
        if w.get("x") is not None and not math.isnan(w["x"]):
            anchor_pos = np.array([w["x"], w["y"], w["z"]])
        else:
            if p.get("grip_bottom", {}).get("x") is not None:
                anchor_pos = np.array([p["grip_bottom"]["x"], p["grip_bottom"]["y"], p["grip_bottom"]["z"]])
            else:
                continue
        
        # 2. 決定軸向
        g_top = p.get("grip_top", {})
        p_top = p.get("top", {})
        
        valid_direction = False
        u_axis = np.array([0.0, 1.0, 0.0])
        
        if p_top.get("x") is not None and g_top.get("x") is not None:
             v1 = np.array([g_top["x"], g_top["y"], g_top["z"]])
             v2 = np.array([p_top["x"], p_top["y"], p_top["z"]])
             axis_vec = v2 - v1
             if np.linalg.norm(axis_vec) > 1.0:
                 u_axis = axis_vec / np.linalg.norm(axis_vec)
                 valid_direction = True
        
        if not valid_direction and p_top.get("x") is not None:
             v_top = np.array([p_top["x"], p_top["y"], p_top["z"]])
             axis_vec = v_top - anchor_pos
             if np.linalg.norm(axis_vec) > 1.0:
                 u_axis = axis_vec / np.linalg.norm(axis_vec)
                 valid_direction = True

        if not valid_direction: continue

        # 3. 決定拍面朝向
        u_right = None
        p_right = p.get("right", {})
        
        if p_right.get("x") is not None and not math.isnan(p_right["x"]):
            v_right = np.array([p_right["x"], p_right["y"], p_right["z"]])
            vec_to_right = v_right - anchor_pos
            vec_perp = vec_to_right - np.dot(vec_to_right, u_axis) * u_axis
            
            if np.linalg.norm(vec_perp) > 20:
                u_right = vec_perp / np.linalg.norm(vec_perp)
                prev_u_right = u_right
        
        if u_right is None and prev_u_right is not None:
            u_right = prev_u_right - np.dot(prev_u_right, u_axis) * u_axis
            if np.linalg.norm(u_right) > 0.1:
                u_right = u_right / np.linalg.norm(u_right)
            else:
                u_right = None

        # 4. 重建所有點
        p["grip_bottom"] = {"x": float(anchor_pos[0]), "y": float(anchor_pos[1]), "z": float(anchor_pos[2])}
        
        new_g_top = anchor_pos + u_axis * DIST_TO_GRIP_TOP
        p["grip_top"] = {"x": float(new_g_top[0]), "y": float(new_g_top[1]), "z": float(new_g_top[2])}
        
        new_bottom = anchor_pos + u_axis * DIST_TO_BOTTOM
        p["bottom"] = {"x": float(new_bottom[0]), "y": float(new_bottom[1]), "z": float(new_bottom[2])}
        
        new_center = anchor_pos + u_axis * DIST_TO_CENTER
        p["center"] = {"x": float(new_center[0]), "y": float(new_center[1]), "z": float(new_center[2])}
        
        new_top = anchor_pos + u_axis * DIST_TO_TOP
        p["top"] = {"x": float(new_top[0]), "y": float(new_top[1]), "z": float(new_top[2])}
        
        if u_right is not None:
            new_right = new_center + u_right * FACE_WIDTH_HALF
            new_left  = new_center - u_right * FACE_WIDTH_HALF
            
            p["right"] = {"x": float(new_right[0]), "y": float(new_right[1]), "z": float(new_right[2])}
            p["left"]  = {"x": float(new_left[0]),  "y": float(new_left[1]),  "z": float(new_left[2])}
        
    return data


def process_trajectories(left_path, leftfront_path, P1, P2):
    """
    使用兩台相機的 2D 軌跡 (left, leftfront) 計算所有 keypoints 的 3D 軌跡。
    ✅ 支援巢狀 paddle 結構 (paddle: {top, right, bottom, left, center})
    """
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle", "tennis_ball"
    ]
    # 更新為7個球拍點（來自B版本）
    paddle_points = ["top", "bottom", "right", "left", "grip_top", "grip_bottom", "center"]

    with open(left_path) as f1, open(leftfront_path) as f2:
        left_data = json.load(f1)
        leftfront_data = json.load(f2)

    # ✅ ① 雙鏡頭幀數檢查
    min_len = min(len(left_data), len(leftfront_data))
    if len(left_data) != len(leftfront_data):
        print(f"[WARNING] 雙鏡頭幀數不同，取最短長度：{min_len}")
        left_data = left_data[:min_len]
        leftfront_data = leftfront_data[:min_len]

    points_3d = []

    for frame_idx, (left_point, leftfront_point) in enumerate(zip(left_data, leftfront_data)):
        frame_data = {"frame": frame_idx}

        # --- (一) 人體與球 ---
        for keypoint in keypoints:
            if (
                keypoint in left_point and keypoint in leftfront_point and
                left_point[keypoint]["x"] is not None and leftfront_point[keypoint]["x"] is not None
            ):
                point1 = np.array([left_point[keypoint]["x"], left_point[keypoint]["y"]])
                point2 = np.array([leftfront_point[keypoint]["x"], leftfront_point[keypoint]["y"]])
                try:
                    X = triangulate_point(P1, P2, point1, point2)
                    frame_data[keypoint] = {
                        "x": float(X[0]),
                        "y": float(-X[1]),
                        "z": float(X[2])
                    }
                except Exception:
                    frame_data[keypoint] = {"x": None, "y": None, "z": None}
            else:
                frame_data[keypoint] = {"x": None, "y": None, "z": None}

        # --- (二) 球拍五點 (巢狀結構) ---
        frame_data["paddle"] = {}
        if "paddle" in left_point and "paddle" in leftfront_point:
            for p in paddle_points:
                lp = left_point["paddle"].get(p, {"x": None, "y": None})
                rp = leftfront_point["paddle"].get(p, {"x": None, "y": None})
                if lp["x"] is not None and rp["x"] is not None:
                    p1 = np.array([lp["x"], lp["y"]])
                    p2 = np.array([rp["x"], rp["y"]])
                    try:
                        X = triangulate_point(P1, P2, p1, p2)
                        frame_data["paddle"][p] = {
                            "x": float(X[0]),
                            "y": float(-X[1]),
                            "z": float(X[2])
                        }
                    except Exception:
                        frame_data["paddle"][p] = {"x": None, "y": None, "z": None}
                else:
                    frame_data["paddle"][p] = {"x": None, "y": None, "z": None}
        else:
            frame_data["paddle"] = {p: {"x": None, "y": None, "z": None} for p in paddle_points}

        # 擊球資訊保留
        frame_data["tennis_ball_hit"] = left_point.get("tennis_ball_hit", False)
        frame_data["tennis_ball_angle"] = left_point.get("tennis_ball_angle", 0)

        points_3d.append(frame_data)

    # --- (三) 輸出結果與修正流水線（來自B版本）---
    fixed_data = fix_trajectory(points_3d)
    
    # 步驟 1: 補結構 (無中生有 Grip_Bottom)
    fixed_data = ensure_paddle_structure(fixed_data)
    
    # 步驟 2: 補動作 (手腕連動)
    fixed_data = fix_paddle_relative_to_wrist(fixed_data)
    
    # 步驟 3: 強制剛體幾何校正 (鎖定大小、形狀)
    fixed_data = enforce_rigid_paddle_geometry(fixed_data)
    
    # 輸出
    output_path = leftfront_path.replace("(2D_trajectory_smoothed).json", "(3D_trajectory).json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)

    # ✅ ⑤ 輸出統計資訊
    valid_ball_frames = sum(1 for f in points_3d if f["tennis_ball"]["x"] is not None)
    print(f"✅ 3D 軌跡輸出完成：{output_path}")
    print(f"[INFO] 共處理 {len(points_3d)} 幀，其中有效球點 {valid_ball_frames} 幀。")

    return output_path


if __name__ == "__main__":
    start = time.perf_counter()

    P1 = np.array([
        [2259.248492, 0.000000, 1651.846528, 0.000000],
        [0.000000, 2262.230378, 1553.020963, 0.000000],
        [0.000000, 0.000000, 1.000000, 0.000000],
    ])

    P2 = np.array([
        [795.771338, -329.492024, 2697.441025, -4465886.061337],
        [-966.406397, 2015.459737, 1255.438530, 2097693.969537],
        [-0.593810, -0.198914, 0.779630, 1344.552439],
    ])

    input_path_1 = "junior_forehand/junior_17/17_1/17_1_side(2D_trajectory_smoothed).json"
    input_path_2 = "junior_forehand/junior_17/17_1/17_1_45(2D_trajectory_smoothed).json"

    output_path = process_trajectories(input_path_1, input_path_2, P1, P2)
    print(f"Execution time: {time.perf_counter() - start:.4f}s")
