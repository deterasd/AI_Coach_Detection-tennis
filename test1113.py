# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path

# === 路徑設定 ===
path_3d = 'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/trajectory/hsiao2__trajectory/trajectory__37/hsiao2__37_45(3D_trajectory).json'
path_2d_45 ='C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/trajectory/hsiao2__trajectory/trajectory__37/hsiao2__37_45(2D_trajectory).json'
path_2d_side = 'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/trajectory/hsiao2__trajectory/trajectory__37/hsiao2__37_side(2D_trajectory).json'
#path_3d = "0306_3/trajectory__2/0306_3__2(3D_trajectory_smoothed).json"
#path_2d_45 = "0306_3/trajectory__2/0306_3__2_45(2D_trajectory_smoothed).json"
#path_2d_side = "0306_3/trajectory__2/0306_3__2_side(2D_trajectory_smoothed).json"

# === 讀取 JSON ===
data_3d = json.load(open(path_3d, "r", encoding="utf-8"))
data_2d_45 = json.load(open(path_2d_45, "r", encoding="utf-8"))
data_2d_side = json.load(open(path_2d_side, "r", encoding="utf-8"))

# === 投影矩陣 ===
P1 = np.array([[682.525930, 0, 637.087464, 0],
                [0, 684.519186, 360.040032, 0],
                [0, 0, 1, 0]], dtype=float)
P2 = np.array([[572.714085, 27.508121, 700.861208, -159410.297486],
                [-27.187871, 663.411218, 332.613656, 51347.982959],
                [-0.102197, 0.053920, 0.993302, 73.630082]], dtype=float)

#P1 = np.array([
#    [  917.153880,     0.000000,   994.529968,     0.000000],
#    [    0.000000,   920.803487,   531.057076,     0.000000],
#   [    0.000000,     0.000000,     1.000000,     0.000000],
#], dtype=float)
#P2 = np.array([
#    [  286.476533,    43.805594,  1301.943509, -765436.820164],
#    [ -309.560886,   957.641377,   401.534167, 365723.173062],
#    [   -0.553187,     0.008475,     0.833014,   660.964347],
#], dtype=float)

def project_points(P, X):
    X = np.hstack([X, np.ones((len(X), 1))])
    x = (P @ X.T).T
    return x[:, :2] / x[:, 2:3]

def extract_points(frame, keys, is3d=False):
    pts = {}
    for k in keys:
        v = frame.get(k)
        if not isinstance(v, dict): continue
        if is3d and all(c in v for c in ("x","y","z")):
            pts[k] = np.array([v["x"], -v["y"], v["z"]])
        elif not is3d and all(c in v for c in ("x","y")):
            pts[k] = np.array([v["x"], v["y"]])
    return pts

# === 主流程 ===
keys_all = ["nose","left_eye","right_eye","left_shoulder","right_shoulder",
            "left_elbow","right_elbow","left_wrist","right_wrist",
            "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
n = min(len(data_3d), len(data_2d_45), len(data_2d_side))

rows, e45, eside = [], {}, {}
for i in range(n):
    f3d, f45, fside = data_3d[i], data_2d_45[i], data_2d_side[i]
    p3d, p45, pside = extract_points(f3d, keys_all, True), extract_points(f45, keys_all), extract_points(fside, keys_all)
    common = sorted(set(p3d) & set(p45) & set(pside))
    if not common: continue

    X = np.stack([p3d[k] for k in common])
    gt_side, gt_45 = np.stack([pside[k] for k in common]), np.stack([p45[k] for k in common])
    proj_side, proj_45 = project_points(P1, X), project_points(P2, X)  # P1→Side, P2→45度
    err_side, err_45 = np.linalg.norm(proj_side - gt_side, axis=1), np.linalg.norm(proj_45 - gt_45, axis=1)

    for k, e_s, e_4 in zip(common, err_side, err_45):
        eside.setdefault(k, []).append(e_s)
        e45.setdefault(k, []).append(e_4)

    rows.append({
        "frame": i,
        "mean_err_cam1_px": err_side.mean(),
        "mean_err_cam2_px": err_45.mean(),
        "max_err_cam1_px": err_side.max(),
        "max_err_cam2_px": err_45.max(),
    })

summary_df = pd.DataFrame(rows)

def summarize(d):
    out = [{"keypoint": k,
            "mean_px": np.mean(v),
            "median_px": np.median(v),
            "p95_px": np.percentile(v, 95),
            "max_px": np.max(v)}
           for k, v in d.items()]
    return pd.DataFrame(out).sort_values("mean_px")

df_cam45 = summarize(e45)
df_camside = summarize(eside)

# === 輸出結果 ===
print("\n=== 每幀平均誤差 ===")
print(summary_df.head())
print("\n=== 相機 45° 每關節誤差 ===")
print(df_cam45)
print("\n=== 側面相機 每關節誤差 ===")
print(df_camside)

# === 計算整體平均誤差 ===
overall_mean_cam45 = df_cam45['mean_px'].mean()
overall_mean_camside = df_camside['mean_px'].mean()
overall_mean_both = (overall_mean_cam45 + overall_mean_camside) / 2

print("\n" + "=" * 80)
print("整體平均重投影誤差")
print("=" * 80)
print(f"相機 45°:     {overall_mean_cam45:>6.2f} pixels")
print(f"側面相機:     {overall_mean_camside:>6.2f} pixels")
print(f"兩角度平均:   {overall_mean_both:>6.2f} pixels")
print("=" * 80)

# 判斷重建品質
if overall_mean_both < 5:
    print("✅ 重建品質: 優秀 (< 5 pixels)")
elif overall_mean_both < 10:
    print("⚠️  重建品質: 良好 (5-10 pixels)")
elif overall_mean_both < 20:
    print("⚠️  重建品質: 可接受 (10-20 pixels)")
else:
    print("❌ 重建品質: 需要改進 (> 20 pixels)")
print("=" * 80)