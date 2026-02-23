import json
import numpy as np
from scipy.signal import savgol_filter
import time
import math

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

def extrapolate_paddle_with_wrist(data):
    """
    利用手腕位移外推缺失的球拍位置。
    當球拍在尾端幀缺失時，np.interp 只會平坦延伸最後有效值，
    導致球拍在 3D 視覺化中「凍住」。此函數利用手腕的持續運動來推動球拍位置。
    """
    print("[INFO] 正在用手腕位移外推缺失的球拍位置...")
    
    paddle_points = ['top', 'right', 'bottom', 'left', 'grip_top', 'grip_bottom', 'center']
    wrist_key = 'right_wrist'
    
    for p_name in paddle_points:
        # 收集每幀的原始球拍數據（3D 三角測量的結果，None 表示缺失）
        raw_valid = []
        for i, frame in enumerate(data):
            paddle = frame.get('paddle', {})
            pt = paddle.get(p_name, {}) if isinstance(paddle, dict) else {}
            x = pt.get('x')
            # 檢查是否為有效數據（非 None 且非 NaN）
            if x is not None and not (isinstance(x, float) and np.isnan(x)):
                raw_valid.append(i)
        
        if len(raw_valid) == 0:
            continue
            
        last_valid_idx = max(raw_valid)
        first_valid_idx = min(raw_valid)
        
        # 只處理「尾端外推」：last_valid_idx 之後仍有幀的情況
        if last_valid_idx >= len(data) - 1:
            continue
        
        # 取最後一個有效幀的球拍位置和手腕位置
        last_paddle = data[last_valid_idx]['paddle'][p_name]
        last_wrist = data[last_valid_idx].get(wrist_key, {})
        
        if last_wrist.get('x') is None:
            continue
        
        lp = np.array([last_paddle['x'], last_paddle['y'], last_paddle['z']])
        lw = np.array([last_wrist['x'], last_wrist['y'], last_wrist['z']])
        
        # 計算最後有效幀的球拍-手腕偏移
        offset = lp - lw
        
        # 對 last_valid_idx+1 到結尾的幀，用手腕位置 + offset 外推
        for i in range(last_valid_idx + 1, len(data)):
            cur_wrist = data[i].get(wrist_key, {})
            if cur_wrist.get('x') is None:
                continue
            
            cw = np.array([cur_wrist['x'], cur_wrist['y'], cur_wrist['z']])
            new_pos = cw + offset
            
            if 'paddle' not in data[i]:
                data[i]['paddle'] = {}
            if p_name not in data[i]['paddle']:
                data[i]['paddle'][p_name] = {}
            
            data[i]['paddle'][p_name]['x'] = float(new_pos[0])
            data[i]['paddle'][p_name]['y'] = float(new_pos[1])
            data[i]['paddle'][p_name]['z'] = float(new_pos[2])
        
        # 同理處理「頭端外推」：first_valid_idx 之前的缺失
        if first_valid_idx > 0:
            first_paddle = data[first_valid_idx]['paddle'][p_name]
            first_wrist = data[first_valid_idx].get(wrist_key, {})
            
            if first_wrist.get('x') is not None:
                fp = np.array([first_paddle['x'], first_paddle['y'], first_paddle['z']])
                fw = np.array([first_wrist['x'], first_wrist['y'], first_wrist['z']])
                offset_head = fp - fw
                
                for i in range(0, first_valid_idx):
                    cur_wrist = data[i].get(wrist_key, {})
                    if cur_wrist.get('x') is None:
                        continue
                    cw = np.array([cur_wrist['x'], cur_wrist['y'], cur_wrist['z']])
                    new_pos = cw + offset_head
                    
                    if 'paddle' not in data[i]:
                        data[i]['paddle'] = {}
                    if p_name not in data[i]['paddle']:
                        data[i]['paddle'][p_name] = {}
                    
                    data[i]['paddle'][p_name]['x'] = float(new_pos[0])
                    data[i]['paddle'][p_name]['y'] = float(new_pos[1])
                    data[i]['paddle'][p_name]['z'] = float(new_pos[2])
    
    print("[INFO] 手腕外推完成")
    return data


def enforce_rigid_paddle_geometry(data):
    """強制球拍剛體幾何校正 - 將球拍點對齊手腕並保持標準尺寸"""
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
    
    # 計算整個序列中的手腕運動方向（用於推斷球拍軸向）
    wrist_positions = []
    for frame in data:
        w = frame.get(wrist_key, {})
        if w.get("x") is not None:
            wrist_positions.append(np.array([w["x"], w["y"], w["z"]]))
        else:
            wrist_positions.append(None)
    
    # 計算平滑的手腕速度方向
    wrist_velocity_direction = None
    if len(wrist_positions) > 5:
        # 使用中後期的手腕位置計算整體揮拍方向
        valid_wrist = [p for p in wrist_positions if p is not None]
        if len(valid_wrist) > 2:
            # 用末端 vs 中段的向量來估計揮拍方向
            mid_idx = len(valid_wrist) // 2
            if mid_idx < len(valid_wrist) - 1:
                wrist_velocity_direction = (valid_wrist[-1] - valid_wrist[mid_idx])
                norm = np.linalg.norm(wrist_velocity_direction)
                if norm > 1.0:
                    wrist_velocity_direction = wrist_velocity_direction / norm

    for frame_idx, frame in enumerate(data):
        if "paddle" not in frame: continue
        p = frame["paddle"]
        w = frame.get(wrist_key, {})
        
        # 1. 決定球拍的 "錨點" (Anchor Position) -> 強制設為手腕！
        if w.get("x") is not None and not np.isnan(w["x"]):
            anchor_pos = np.array([w["x"], w["y"], w["z"]])
        else:
            # 備案：如果這幀剛好沒抓到手腕，就用原本的 Grip_Bottom
            if p.get("grip_bottom", {}).get("x") is not None:
                anchor_pos = np.array([p["grip_bottom"]["x"], p["grip_bottom"]["y"], p["grip_bottom"]["z"]])
            else:
                continue
        
        # 2. 決定球拍的 "軸向" (Axis Vector) -> 指向 Top
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
        
        # ===== 新增：如果仍無有效方向，用手腕速度方向 =====
        if not valid_direction and wrist_velocity_direction is not None:
            u_axis = wrist_velocity_direction
            valid_direction = True
        
        # ===== 新增：計算本幀的手腕速度方向作為備選 =====
        if not valid_direction and frame_idx > 0 and wrist_positions[frame_idx] is not None and wrist_positions[frame_idx - 1] is not None:
            wrist_motion = wrist_positions[frame_idx] - wrist_positions[frame_idx - 1]
            if np.linalg.norm(wrist_motion) > 0.5:
                u_axis = wrist_motion / np.linalg.norm(wrist_motion)
                valid_direction = True

        if not valid_direction: continue

        # 3. 決定 "拍面朝向" (Face Orientation - Right/Left)
        u_right = None
        p_right = p.get("right", {})
        
        if p_right.get("x") is not None and not np.isnan(p_right["x"]):
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

        # 4. 重建所有點 (Snap to Wrist)
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


def smooth_3D_trajectory(input_file, window_length=15, polyorder=3, tennis_window_length=7, tennis_polyorder=2):
    """
    平滑 3D 軌跡資料 (支援巢狀 paddle 結構)
    """
    # 讀取輸入 JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 儲存擊球資訊（以免被平滑覆蓋）
    original_hit_data = [(frame.get('tennis_ball_hit', False), frame.get('tennis_ball_angle', 0)) for frame in data]

    # 平滑處理的 keypoints（不包含 paddle）
    keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'tennis_ball'
    ]

    # ---------- (一) 平滑網球 ----------
    first_valid, last_valid = None, None
    for i, frame in enumerate(data):
        ball = frame.get('tennis_ball', {})
        if all(ball.get(c) is not None for c in ['x', 'y', 'z']):
            if first_valid is None:
                first_valid = i
            last_valid = i

    if first_valid is not None and last_valid is not None:
        coords = {c: [] for c in ['x', 'y', 'z']}
        for frame in data[first_valid:last_valid + 1]:
            for c in coords:
                coords[c].append(frame['tennis_ball'][c] if frame['tennis_ball'][c] is not None else np.nan)

        for c in coords:
            coords[c] = np.array(coords[c], dtype=float)
            if np.any(np.isnan(coords[c])):
                valid = ~np.isnan(coords[c])
                coords[c][~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), coords[c][valid])
            if len(coords[c]) > tennis_window_length:
                coords[c] = savgol_filter(coords[c], tennis_window_length, tennis_polyorder)
            for j, val in enumerate(coords[c]):
                data[first_valid + j]['tennis_ball'][c] = float(val)

    # ---------- (二) 平滑人體 keypoints ----------
    for key in keypoints:
        if key == 'tennis_ball':
            continue  # 已處理
        coords = {c: [] for c in ['x', 'y', 'z']}
        for frame in data:
            pt = frame.get(key, {})
            for c in coords:
                coords[c].append(pt.get(c, np.nan))
        for c in coords:
            coords[c] = np.array(coords[c], dtype=float)
            if np.any(np.isnan(coords[c])):
                valid = ~np.isnan(coords[c])
                coords[c][~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), coords[c][valid])
            if len(coords[c]) > window_length:
                smooth_vals = savgol_filter(coords[c], window_length, polyorder)
                for j, val in enumerate(smooth_vals):
                    data[j][key][c] = float(val)

    # ---------- (三) 用手腕外推缺失的球拍位置（在平滑前） ----------
    data = extrapolate_paddle_with_wrist(data)

    # ---------- (三-b) 平滑球拍七點 (巢狀結構) ----------
    paddle_points = ['top', 'right', 'bottom', 'left', 'grip_top', 'grip_bottom', 'center']
    for p in paddle_points:
        coords = {c: [] for c in ['x', 'y', 'z']}
        for frame in data:
            paddle = frame.get('paddle', {})
            sub_pt = paddle.get(p, {}) if isinstance(paddle, dict) else {}
            for c in coords:
                coords[c].append(sub_pt.get(c, np.nan))
        for c in coords:
            coords[c] = np.array(coords[c], dtype=float)
            if np.any(np.isnan(coords[c])):
                valid = ~np.isnan(coords[c])
                if np.any(valid):
                    coords[c][~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), coords[c][valid])
            if len(coords[c]) > window_length:
                smooth_vals = savgol_filter(coords[c], window_length, polyorder)
                for j, val in enumerate(smooth_vals):
                    if 'paddle' not in data[j]:
                        data[j]['paddle'] = {}
                    if p not in data[j]['paddle']:
                        data[j]['paddle'][p] = {}
                    data[j]['paddle'][p][c] = float(val)

    # ---------- (四) 還原擊球資訊 ----------
    for i, (hit, angle) in enumerate(original_hit_data):
        data[i]['tennis_ball_hit'] = hit
        data[i]['tennis_ball_angle'] = angle

    # ---------- (五) 執行球拍剛體幾何校正 ----------
    data = enforce_rigid_paddle_geometry(data)

    # ---------- (六) 輸出結果 ----------
    output_file = input_file.replace('(3D_trajectory).json', '(3D_trajectory_smoothed).json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NanToNullEncoder)

    print(f"✅ 3D 平滑完成: {output_file}")
    return output_file


if __name__ == "__main__":
    start = time.perf_counter()
    input_path = "trajectory/hsiao2__trajectory/trajectory__20/hsiao2__20(3D_trajectory).json"
    output_path = smooth_3D_trajectory(input_path)
    print(f"Execution time: {time.perf_counter() - start:.4f}s")
