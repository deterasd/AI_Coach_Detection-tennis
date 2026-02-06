import numpy as np
import json
import time
from scipy.interpolate import interp1d

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
        # print("[WARNING] fix_trajectory: 球點不足或未偵測到球，略過修正")
        return data

    for axis in ["x", "y", "z"]:
        vals = [data[i]["tennis_ball"][axis] for i in valid_idx]
        interp = interp1d(valid_idx, vals, kind="cubic", fill_value="extrapolate")
        for i in range(valid_idx[0], valid_idx[-1]):
            data[i]["tennis_ball"][axis] = float(interp(i))

    # print(f"[INFO] fix_trajectory: 修正完成 (frame {valid_idx[0]} → {valid_idx[-1]})")
    return data

#----------------------------------------------
def ensure_paddle_structure(data):
    print("[INFO] 正在檢查並重建缺失的球拍結構...")
    for frame in data:
        if "paddle" not in frame: continue
        p = frame["paddle"]
        
        # 如果 grip_bottom 缺失，嘗試從 center 和 grip_top 推算
        if p.get("grip_bottom", {}).get("x") is None:
            gt = p.get("grip_top", {})
            c = p.get("center", {}) # 或用 top
            
            if gt.get("x") is not None and c.get("x") is not None:
                # 建立向量 Center -> Grip_Top
                vec = np.array([gt["x"]-c["x"], gt["y"]-c["y"], gt["z"]-c["z"]])
                
                # 假設握把長度大約是 Center到Grip_Top距離的 0.8 倍
                extension_ratio = 0.8 
                gb = np.array([gt["x"], gt["y"], gt["z"]]) + vec * extension_ratio
                
                p["grip_bottom"] = {"x": float(gb[0]), "y": float(gb[1]), "z": float(gb[2])}

    return data

def fix_paddle_relative_to_wrist(data):
    """
    使用 '手腕' 的軌跡來帶動 '球拍' 的軌跡。
    """
    paddle_parts = ["top", "bottom", "right", "left", "center", "grip_top", "grip_bottom"]
    
    # 假設右手持拍
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

# ---------------------------------------------------------
# 4. 強制剛體幾何校正 (解決變形/縮成一團/忽大忽小)
# ---------------------------------------------------------
def enforce_rigid_paddle_geometry(data):
    print("[INFO] 正在執行網球拍幾何校正 (Snap to Wrist + Rigid Body)...")

    # === 網球拍標準尺寸 (單位: mm) ===
    HANDLE_LENGTH = 190.0   # 握把長 19cm
    NECK_LENGTH = 70.0      # 頸部 7cm
    FACE_LENGTH = 340.0     # 拍面長 34cm
    FACE_WIDTH_HALF = 135.0 # 拍面半寬 13.5cm
    
    # 計算各點相對於 Grip_Bottom 的距離
    DIST_TO_GRIP_TOP = HANDLE_LENGTH
    DIST_TO_BOTTOM = HANDLE_LENGTH + NECK_LENGTH
    DIST_TO_CENTER = HANDLE_LENGTH + NECK_LENGTH + (FACE_LENGTH / 2.0)
    DIST_TO_TOP = HANDLE_LENGTH + NECK_LENGTH + FACE_LENGTH

    # 手腕 Key (假設右手)
    wrist_key = "right_wrist"

    # 用來記憶上一幀的拍面方向 (處理垂直視角問題)
    prev_u_right = None

    for frame in data:
        if "paddle" not in frame: continue
        p = frame["paddle"]
        w = frame.get(wrist_key, {})
        
        # 1. 決定球拍的 "錨點" (Anchor Position) -> 強制設為手腕！
        # 如果手腕有資料，Grip_Bottom 就等於手腕
        # 如果手腕沒資料，只好退而求其次用原本的 Grip_Bottom
        if w.get("x") is not None and not np.isnan(w["x"]):
            anchor_pos = np.array([w["x"], w["y"], w["z"]])
        else:
            # 備案：如果這幀剛好沒抓到手腕，就檢查是否有原本的 Grip_Bottom
            if p.get("grip_bottom", {}).get("x") is not None:
                anchor_pos = np.array([p["grip_bottom"]["x"], p["grip_bottom"]["y"], p["grip_bottom"]["z"]])
            else:
                continue # 沒手腕也沒球拍，無法重建
        
        # 2. 決定球拍的 "軸向" (Axis Vector) -> 指向 Top
        # 我們需要 Grip_Top 和 Top 來決定方向
        g_top = p.get("grip_top", {})
        p_top = p.get("top", {})
        
        # 如果這兩點缺失，或是座標是 NaN，我們嘗試用 center 甚至 bottom 來湊
        # 這裡先假設至少有 top (因為它最顯眼)
        valid_direction = False
        u_axis = np.array([0.0, 1.0, 0.0]) # 預設向上
        
        if p_top.get("x") is not None and g_top.get("x") is not None:
             v1 = np.array([g_top["x"], g_top["y"], g_top["z"]])
             v2 = np.array([p_top["x"], p_top["y"], p_top["z"]])
             axis_vec = v2 - v1
             if np.linalg.norm(axis_vec) > 1.0:
                 u_axis = axis_vec / np.linalg.norm(axis_vec)
                 valid_direction = True
        
        # 如果方向算不出來 (例如只有 Grip_Bottom)，這幀可能就爛掉了
        # 但我們可以嘗試用 "Wrist -> Top" 來當方向 (如果 Grip_Top 沒抓到)
        if not valid_direction and p_top.get("x") is not None:
             v_top = np.array([p_top["x"], p_top["y"], p_top["z"]])
             axis_vec = v_top - anchor_pos
             if np.linalg.norm(axis_vec) > 1.0:
                 u_axis = axis_vec / np.linalg.norm(axis_vec)
                 valid_direction = True

        if not valid_direction: continue

        # 3. 決定 "拍面朝向" (Face Orientation - Right/Left)
        u_right = None
        p_right = p.get("right", {})
        
        if p_right.get("x") is not None and not np.isnan(p_right["x"]):
            v_right = np.array([p_right["x"], p_right["y"], p_right["z"]])
            # 計算從軸線指向 Right 的向量
            # 注意：這裡要用 anchor_pos 當原點來算向量嗎？不，用方向就好
            # 我們計算 "目前 Right 點" 相對於 "目前 Top 點" 的方向，這比較穩
            # 或者直接算 Right - Anchor
            vec_to_right = v_right - anchor_pos
            vec_perp = vec_to_right - np.dot(vec_to_right, u_axis) * u_axis
            
            if np.linalg.norm(vec_perp) > 20: # 只要不是垂直視角
                u_right = vec_perp / np.linalg.norm(vec_perp)
                prev_u_right = u_right # 更新記憶
        
        # 如果這幀算不出來，使用記憶
        if u_right is None and prev_u_right is not None:
            # 校正記憶向量，確保它垂直於新的 u_axis
            u_right = prev_u_right - np.dot(prev_u_right, u_axis) * u_axis
            if np.linalg.norm(u_right) > 0.1:
                u_right = u_right / np.linalg.norm(u_right)
            else:
                u_right = None # 真的很不幸，方向重疊

        # 4. 重建所有點 (Snap to Wrist)
        # 這是最關鍵的一步：所有點都從 anchor_pos (手腕) 長出來！
        
        # (A) Grip_Bottom = Anchor (Wrist)
        p["grip_bottom"] = {"x": float(anchor_pos[0]), "y": float(anchor_pos[1]), "z": float(anchor_pos[2])}
        
        # (B) 其他點沿著 u_axis 延伸
        new_g_top = anchor_pos + u_axis * DIST_TO_GRIP_TOP
        p["grip_top"] = {"x": float(new_g_top[0]), "y": float(new_g_top[1]), "z": float(new_g_top[2])}
        
        new_bottom = anchor_pos + u_axis * DIST_TO_BOTTOM
        p["bottom"] = {"x": float(new_bottom[0]), "y": float(new_bottom[1]), "z": float(new_bottom[2])}
        
        new_center = anchor_pos + u_axis * DIST_TO_CENTER
        p["center"] = {"x": float(new_center[0]), "y": float(new_center[1]), "z": float(new_center[2])}
        
        new_top = anchor_pos + u_axis * DIST_TO_TOP
        p["top"] = {"x": float(new_top[0]), "y": float(new_top[1]), "z": float(new_top[2])}
        
        # (C) 左右點強制撐開
        if u_right is not None:
            new_right = new_center + u_right * FACE_WIDTH_HALF
            new_left  = new_center - u_right * FACE_WIDTH_HALF
            
            p["right"] = {"x": float(new_right[0]), "y": float(new_right[1]), "z": float(new_right[2])}
            p["left"]  = {"x": float(new_left[0]),  "y": float(new_left[1]),  "z": float(new_left[2])}
        
    return data


#----------------------------------------------
def process_trajectories(left_path, leftfront_path, P1, P2):
    """
    使用兩台相機的 2D 軌跡 (left, leftfront) 計算所有 keypoints 的 3D 軌跡。
    """
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle", "tennis_ball"
    ]
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

    # --- (三) 輸出結果與修正流水線 ---

    fixed_data = fix_trajectory(points_3d)
    
    # 步驟 1: 補結構 (無中生有 Grip_Bottom)
    fixed_data = ensure_paddle_structure(fixed_data)
    
    # 步驟 2: 補動作 (手腕連動)
    fixed_data = fix_paddle_relative_to_wrist(fixed_data)
    
    # ✅ 步驟 3: [新增] 強制剛體幾何校正 (鎖定大小、形狀)
    fixed_data = enforce_rigid_paddle_geometry(fixed_data)
    
    # 輸出
    output_path = leftfront_path.replace("(2D_trajectory_smoothed).json", "(3D_trajectory).json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False)

    # ✅ ⑤ 輸出統計資訊
    valid_ball_frames = sum(1 for f in points_3d if f["tennis_ball"]["x"] is not None)
    print(f"✅ 3D 軌跡輸出完成：{output_path}")
    print(f"[INFO] 共處理 {len(points_3d)} 幀，其中有效球點 {valid_ball_frames} 幀。")

    return output_path


if __name__ == "__main__":
    start = time.perf_counter()

    # 請確認您的 P1, P2 矩陣是正確的
    P1 = np.array([
    [916.626242, 0, 960.250417, 0],
    [0, 921.951283, 523.154606, 0],
    [0, 0, 1, 0]
    ])
    P2 = np.array([
    [782.909772, -18.15298, 1066.6776, -255341.954492],
    [-25.104948, 925.678666, 514.730223, 46851.486878],
    [-0.122625, 0.020539, 0.992241, 90.876653]
    ])

    input_path_1 = "junior_forehand/junior_17/17_1/17_1_side(2D_trajectory_smoothed).json"
    input_path_2 = "junior_forehand/junior_17/17_1/17_1_45(2D_trajectory_smoothed).json"

    output_path = process_trajectories(input_path_1, input_path_2, P1, P2)
    print(f"Execution time: {time.perf_counter() - start:.4f}s")