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
    paddle_points = ["top", "right", "bottom", "left", "center"]

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

    # --- (三) 輸出結果 ---
    # ✅ ④ 改成更安全的 replace 命名法
    output_path = leftfront_path.replace("(2D_trajectory_smoothed).json", "(3D_trajectory).json")
    fixed_data = fix_trajectory(points_3d)

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
