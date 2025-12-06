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

    # ---------- (三) 平滑球拍五點 (巢狀結構) ----------
    paddle_points = ['top', 'right', 'bottom', 'left', 'center']
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

    # ---------- (五) 輸出結果 ----------
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
