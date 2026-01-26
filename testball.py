import json
import numpy as np
import pandas as pd
#from caas_jupyter_tools import display_dataframe_to_user
import cv2

# --- Load data ---
with open("/mnt/data/hsiao2__37_45(2D_trajectory).json","r",encoding="utf-8") as f:
    data_leftfront = json.load(f)
with open("/mnt/data/hsiao2__37_side(2D_trajectory).json","r",encoding="utf-8") as f:
    data_left = json.load(f)
with open("/mnt/data/hsiao2__37_45(3D_trajectory).json","r",encoding="utf-8") as f:
    data_3d = json.load(f)

def extract_2d_list(data2d):
    pts = []
    for frame in data2d:
        if isinstance(frame, dict):
            c = None
            if "paddle" in frame and "center" in frame["paddle"]:
                c = frame["paddle"]["center"]
                x, y = c.get("x"), c.get("y")
            elif "x" in frame and "y" in frame:
                x, y = frame["x"], frame["y"]
            else:
                x = y = None
            if x is not None and y is not None:
                pts.append((float(x), float(y)))
    return np.array(pts, dtype=np.float64)

def extract_3d_list(data3d):
    pts = []
    for frame in data3d:
        if isinstance(frame, dict):
            if "paddle" in frame and "center" in frame["paddle"]:
                c = frame["paddle"]["center"]
                x, y, z = c.get("x"), c.get("y"), c.get("z")
            else:
                x, y, z = frame.get("x"), frame.get("y"), frame.get("z")
            if None not in (x, y, z):
                pts.append((float(x), float(y), float(z)))
    return np.array(pts, dtype=np.float64)

pts2D_LF = extract_2d_list(data_leftfront)
pts2D_L  = extract_2d_list(data_left)
pts3D    = extract_3d_list(data_3d)

n = min(len(pts2D_LF), len(pts2D_L), len(pts3D))
pts2D_LF, pts2D_L, pts3D = pts2D_LF[:n], pts2D_L[:n], pts3D[:n]

# --- Projection matrices from user ---
P_L = np.array([
    [682.525930, 0.000000, 637.087464, 0.000000],
    [0.000000, 684.519186, 360.040032, 0.000000],
    [0.000000, 0.000000, 1.000000, 0.000000]
], dtype=np.float64)
P_LF = np.array([
    [572.714085, 27.508121, 700.861208, -159410.297486],
    [-27.187871, 663.411218, 332.613656,  51347.982959],
    [-0.102197,   0.053920,   0.993302,     73.630082]
], dtype=np.float64)

def project_points(P, X):
    X_h = np.hstack([X, np.ones((len(X),1))])
    proj = (P @ X_h.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv

def reproj_errors(P, X, uv_obs):
    uv = project_points(P, X)
    err = np.linalg.norm(uv - uv_obs, axis=1)
    return err, uv

# Decompose P to get R,t to compute depths (cheirality)
def decompose_P(P):
    K, R, t_hom = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2,2]
    t = (t_hom[:3] / t_hom[3]).reshape(3,1)
    return K, R, t

def depths_in_camera(R, t, X):
    # X camera = R*X + t
    Xc = (R @ X.T + t).T
    return Xc[:, 2]

# Try two scales for 3D: as-is, and *1000 (m->mm hypothesis)
scales = {"as_is": 1.0, "x1000": 1000.0}
rows = []
perframe_tables = {}

# Precompute extrinsics for cheirality
K_L, R_L, t_L = decompose_P(P_L)
K_LF, R_LF, t_LF = decompose_P(P_LF)

for tag, s in scales.items():
    Xs = pts3D * s
    err_LF, uv_LF = reproj_errors(P_LF, Xs, pts2D_LF)
    err_L,  uv_L  = reproj_errors(P_L,  Xs, pts2D_L)
    # summarize
    for name, err in [("LeftFront", err_LF), ("Left", err_L)]:
        rows.append({
            "scale": tag,
            "camera": name,
            "count": int(len(err)),
            "mean_px": float(np.mean(err)),
            "std_px": float(np.std(err)),
            "median_px": float(np.median(err)),
            "p95_px": float(np.percentile(err,95)),
            "max_px": float(np.max(err)),
        })
    # depths (cheirality)
    Z_L  = depths_in_camera(R_L,  t_L,  Xs)
    Z_LF = depths_in_camera(R_LF, t_LF, Xs)
    rows.append({
        "scale": tag, "camera":"Cheirality",
        "count": int(len(Xs)),
        "mean_px": float(np.mean((Z_L>0) & (Z_LF>0))),
        "std_px": float(np.mean(Z_L>0)),
        "median_px": float(np.mean(Z_LF>0)),
        "p95_px": float(np.median(Z_L)),
        "max_px": float(np.median(Z_LF)),
    })
    # store per-frame for user inspection (first 30)
    per = pd.DataFrame({
        "frame_index": np.arange(len(Xs)),
        "err_leftfront_px": err_LF,
        "err_left_px": err_L
    })
    perframe_tables[tag] = per

summary = pd.DataFrame(rows)
display_dataframe_to_user("Reprojection Summary (two scales tried)", summary)

for tag, df in perframe_tables.items():
    display_dataframe_to_user(f"Per-frame Errors ({tag})", df.head(30))

# Save CSVs
summary.to_csv("/mnt/data/reprojection_summary_two_scales.csv", index=False)
for tag, df in perframe_tables.items():
    df.to_csv(f"/mnt/data/per_frame_errors_{tag}.csv", index=False)

print("Saved: /mnt/data/reprojection_summary_two_scales.csv")
print("Saved: /mnt/data/per_frame_errors_as_is.csv")
print("Saved: /mnt/data/per_frame_errors_x1000.csv")
