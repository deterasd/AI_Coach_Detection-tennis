# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path
import math

# Files
path_3d = Path("C:/Users/chen/Downloads/0306_3/0306_3/trajectory__2/0306_3__2(3D_trajectory_smoothed).json")
path_2d_45 = Path("C:/Users/chen/Downloads/0306_3/0306_3/trajectory__2/0306_3__2_45(2D_trajectory).json")
path_2d_side = Path("C:/Users/chen/Downloads/0306_3/0306_3/trajectory__2/0306_3__2_side(2D_trajectory).json")

# Load data
with open(path_3d, "r", encoding="utf-8") as f:
    data_3d = json.load(f)
with open(path_2d_45, "r", encoding="utf-8") as f:
    data_2d_45 = json.load(f)
with open(path_2d_side, "r", encoding="utf-8") as f:
    data_2d_side = json.load(f)

# Projection matrices from the user
"""
P1 = np.array([
    [682.525930, 0.000000, 637.087464, 0.000000],
    [0.000000, 684.519186, 360.040032, 0.000000],
    [0.000000, 0.000000, 1.000000, 0.000000],
], dtype=float)

P2 = np.array([
    [572.714085, 27.508121, 700.861208, -159410.297486],
    [-27.187871, 663.411218, 332.613656, 51347.982959],
    [-0.102197, 0.053920, 0.993302, 73.630082],
], dtype=float)"""
P1 = np.array([
        [  917.153880,     0.000000,   994.529968,     0.000000],
        [    0.000000,   920.803487,   531.057076,     0.000000],
        [    0.000000,     0.000000,     1.000000,     0.000000],
    ], dtype=float)

P2 = np.array([
        [  286.476533,    43.805594,  1301.943509, -765436.820164],
        [ -309.560886,   957.641377,   401.534167, 365723.173062],
        [   -0.553187,     0.008475,     0.833014,   660.964347],
    ], dtype=float)

def project_points(P, X):
    """Project 3D point(s) X (N x 3) using camera matrix P (3x4). Returns N x 2"""
    if X.ndim == 1:
        X = X[None, :]
    ones = np.ones((X.shape[0], 1))
    Xh = np.hstack([X, ones])
    x = (P @ Xh.T).T  # N x 3
    uv = x[:, :2] / x[:, 2:3]
    return uv

# Keypoint names to evaluate
simple_kps = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
# For paddle, use center as a single keypoint
paddle_kps = [("paddle","center")]

def get_3d_points(frame3d):
    pts = {}
    for k in simple_kps:
        v = frame3d.get(k, None)
        if isinstance(v, dict):
            x,y,z = v.get("x"), v.get("y"), v.get("z")
            if x is not None and y is not None and z is not None:
                pts[k] = np.array([x,y,z], dtype=float)
    # paddle center (optional in some frames)
    p = frame3d.get("paddle",{}).get("center",{})
    if isinstance(p, dict):
        if all(p.get(c) is not None for c in ("x","y","z")):
            pts["paddle_center"] = np.array([p["x"],p["y"],p["z"]], dtype=float)
    return pts

def get_2d_points(frame2d):
    pts = {}
    for k in simple_kps:
        v = frame2d.get(k, None)
        if isinstance(v, dict):
            x,y = v.get("x"), v.get("y")
            if x is not None and y is not None:
                pts[k] = np.array([x,y], dtype=float)
    p = frame2d.get("paddle",{}).get("center",{})
    if isinstance(p, dict):
        if all(p.get(c) is not None for c in ("x","y")):
            pts["paddle_center"] = np.array([p["x"],p["y"]], dtype=float)
    return pts

# Align frame counts
n = min(len(data_3d), len(data_2d_45), len(data_2d_side))

rows = []
per_point_errors_cam1 = {}
per_point_errors_cam2 = {}

for i in range(n):
    f3d = data_3d[i]
    f45 = data_2d_45[i]
    fside = data_2d_side[i]
    pts3d = get_3d_points(f3d)
    pts2d_45 = get_2d_points(f45)
    pts2d_side = get_2d_points(fside)

    # Common keys for which we have all (3D and both 2D)
    keys = sorted(set(pts3d.keys()) & set(pts2d_45.keys()) & set(pts2d_side.keys()))
    if not keys:
        continue

    X = np.stack([pts3d[k] for k in keys], axis=0)  # M x 3
    proj1 = project_points(P1, X)  # M x 2
    proj2 = project_points(P2, X)  # M x 2

    # Compute pixel errors
    gt1 = np.stack([pts2d_45[k] for k in keys], axis=0)
    gt2 = np.stack([pts2d_side[k] for k in keys], axis=0)

    err1 = np.linalg.norm(proj1 - gt1, axis=1)  # per key
    err2 = np.linalg.norm(proj2 - gt2, axis=1)

    # Save per-point errors
    for k, e1, e2 in zip(keys, err1, err2):
        per_point_errors_cam1.setdefault(k, []).append(float(e1))
        per_point_errors_cam2.setdefault(k, []).append(float(e2))

    rows.append({
        "frame": i,
        "num_points": len(keys),
        "mean_err_cam1_px": float(err1.mean()),
        "median_err_cam1_px": float(np.median(err1)),
        "max_err_cam1_px": float(err1.max()),
        "mean_err_cam2_px": float(err2.mean()),
        "median_err_cam2_px": float(np.median(err2)),
        "max_err_cam2_px": float(err2.max()),
    })

summary_df = pd.DataFrame(rows)

# Aggregate per-point statistics
def agg_stats(d):
    out_rows = []
    for k, arr in d.items():
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            continue
        out_rows.append({
            "keypoint": k,
            "count": int(arr.size),
            "mean_px": float(arr.mean()),
            "median_px": float(np.median(arr)),
            "p95_px": float(np.percentile(arr, 95)),
            "max_px": float(arr.max()),
        })
    return pd.DataFrame(out_rows).sort_values("mean_px")

per_point_cam1_df = agg_stats(per_point_errors_cam1)
per_point_cam2_df = agg_stats(per_point_errors_cam2)

# Save to CSVs
out_dir = Path("/mnt/data/validation_reports")
out_dir.mkdir(parents=True, exist_ok=True)
summary_path = out_dir / "per_frame_reprojection_errors.csv"
cam1_path = out_dir / "per_point_errors_cam1.csv"
cam2_path = out_dir / "per_point_errors_cam2.csv"

summary_df.to_csv(summary_path, index=False, encoding="utf-8")
per_point_cam1_df.to_csv(cam1_path, index=False, encoding="utf-8")
per_point_cam2_df.to_csv(cam2_path, index=False, encoding="utf-8")

print("=== Per-frame reprojection errors ===")
print(summary_df.head())

print("\n=== Per-keypoint errors (camera 45°) ===")
print(per_point_cam1_df.head())

print("\n=== Per-keypoint errors (camera side) ===")
print(per_point_cam2_df.head())

