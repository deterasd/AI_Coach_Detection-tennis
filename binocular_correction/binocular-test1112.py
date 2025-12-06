import numpy as np
import cv2
import os
import json

# 設定標定時的終止條件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# 創建標定板的三維座標點
objp = np.zeros((7 * 10, 3), np.float32)
square_size = 80  # 設定標定板方格的實際大小(mm)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square_size

# 創建存儲容器
objpoints = []      # 存儲標定板上點的三維座標
imgpointsLF = []    # 存儲左前(45°)相機拍攝圖片中檢測到的角點二維座標
imgpointsL = []     # 存儲左(側)相機拍攝圖片中檢測到的角點二維座標

# 確保輸出目錄存在
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 檢測統計數據
stats = {
    'total_images': 0,
    'detected_normal': 0,
    'detected_inverted': 0,
    'detected_both': 0,
    'detected_none': 0
}

# 檢測棋盤格函數，嘗試原始和倒置兩種方式
def detect_chessboard(image, pattern_size=(10, 7)):
    """嘗試使用原始圖像和倒置圖像檢測棋盤格"""
    ret_orig, corners_orig = cv2.findChessboardCorners(image, pattern_size, None)
    image_inv = cv2.bitwise_not(image)
    ret_inv, corners_inv = cv2.findChessboardCorners(image_inv, pattern_size, None)
    if ret_orig and ret_inv:
        return (ret_orig, corners_orig, "原始") if len(corners_orig) >= len(corners_inv) else (ret_inv, corners_inv, "倒置")
    elif ret_orig:
        return ret_orig, corners_orig, "原始"
    elif ret_inv:
        return ret_inv, corners_inv, "倒置"
    else:
        return False, None, "無法檢測"

# 讀取並處理每一張標定圖片（依你的路徑）
for i in range(15):
    t = str(i)
    stats['total_images'] += 1

    lf_path = f'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/binocular_correction/indoor1113/45/Indoor.{i}.JPG'
    l_path  = f'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/binocular_correction/indoor1113/side/Indoor.{i}.JPG'

    if not (os.path.exists(lf_path) and os.path.exists(l_path)):
        print(f"警告: 圖像 {i} 不存在，跳過")
        continue

    ChessImaLF = cv2.imread(lf_path, 0)
    ChessImaL  = cv2.imread(l_path, 0)

    if ChessImaLF is None or ChessImaL is None:
        print(f"警告: 圖像 {i} 讀取失敗，跳過")
        continue

    retLF, cornersLF, methodLF = detect_chessboard(ChessImaLF)
    retL,  cornersL,  methodL  = detect_chessboard(ChessImaL)

    print(f"Image {t} - LeftFront: {retLF} ({methodLF}), Left: {retL} ({methodL})")

    if retLF and not retL:
        stats['detected_normal'] += 1
    elif not retLF and retL:
        stats['detected_inverted'] += 1
    elif retLF and retL:
        stats['detected_both'] += 1
    else:
        stats['detected_none'] += 1

    if retLF and retL:
        objpoints.append(objp.copy())

        # 亞像素角點
        cv2.cornerSubPix(ChessImaLF, cornersLF, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL,  cornersL,  (11, 11), (-1, -1), criteria)

        imgpointsLF.append(cornersLF)
        imgpointsL.append(cornersL)

        # 可選：畫角點存檔
        ChessImaLF_color = cv2.cvtColor(ChessImaLF, cv2.COLOR_GRAY2BGR)
        ChessImaL_color  = cv2.cvtColor(ChessImaL,  cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(ChessImaLF_color, (10,7), cornersLF, retLF)
        cv2.drawChessboardCorners(ChessImaL_color,  (10,7), cornersL,  retL)
        cv2.imwrite(f'{output_dir}/R_{t}_{methodLF}.jpg', ChessImaLF_color)
        cv2.imwrite(f'{output_dir}/L_{t}_{methodL}.jpg',  ChessImaL_color)

print("\n==== 棋盤格檢測統計 ====")
print(f"總圖像數量: {stats['total_images']}")
print(f"兩個相機都成功檢測: {len(objpoints)} ({len(objpoints)/max(stats['total_images'],1)*100:.1f}%)")
print(f"僅左前相機檢測成功: {stats['detected_normal']}")
print(f"僅左相機檢測成功: {stats['detected_inverted']}")
print(f"兩個相機都未檢測成功: {stats['detected_none']}")

if len(objpoints) < 3:
    print("錯誤: 成功檢測的棋盤格數量不足，無法進行相機標定。至少需要 3 個棋盤格圖像。")
    raise SystemExit

# 尺寸
img_size_LF = ChessImaLF.shape[::-1]
img_size_L  = ChessImaL.shape[::-1]
if img_size_LF != img_size_L:
    raise RuntimeError("兩相機影像解析度不一致，請用相同設定拍攝棋盤格。")
img_size = img_size_LF  # (w,h)

# ---------- 單目標定 ----------
print("\n[STEP] Calibrate single cameras")
retLF, mtxLF, distLF, rvecsLF, tvecsLF = cv2.calibrateCamera(objpoints, imgpointsLF, img_size, None, None)
retL,  mtxL,  distL,  rvecsL,  tvecsL  = cv2.calibrateCamera(objpoints, imgpointsL,  img_size, None, None)

def single_rmse(K, D, rvecs, tvecs, imgpoints):
    total_err2, total_pts = 0.0, 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        total_err2 += err*err
        total_pts  += len(proj)
    return float(np.sqrt(total_err2 / max(total_pts,1)))

print(f"[RMS] 左前(45°) = {single_rmse(mtxLF, distLF, rvecsLF, tvecsLF, imgpointsLF):.4f} px")
print(f"[RMS] 左(側)   = {single_rmse(mtxL,  distL,  rvecsL,  tvecsL,  imgpointsL ): .4f} px")

# ---------- 雙目標定（固定內參，解 R,T） ----------
print("\n[STEP] Stereo calibrate (FIX_INTRINSIC)")
flags = cv2.CALIB_FIX_INTRINSIC
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsLF,
    mtxL, distL, mtxLF, distLF,
    img_size,
    criteria=criteria_stereo, flags=flags
)
print(f"[stereo RMS] {retS:.6f}")

# ---------- 原始投影矩陣（pinhole） ----------
I = np.eye(3)
P_left      = MLS @ np.hstack([I, np.zeros((3,1))])      # K1 [I|0]
P_leftfront = MRS @ np.hstack([R, T.reshape(3,1)])       # K2 [R|T]

np.set_printoptions(suppress=True, precision=6, floatmode='fixed', threshold=np.inf)
print("\n==== 標定結果 ====")
print("左相機內參矩陣 MLS:\n", MLS)
print("\n左前相機內參矩陣 MRS:\n", MRS)
print("\n旋轉矩陣 R:\n", R)
print("\n平移向量 T:\n", T)

print("\n==== 投影矩陣計算結果 ====")
print("左相機投影矩陣:\n", P_left)
print("\n左前相機投影矩陣:\n", P_leftfront)

# ---------- 去畸變/極線校正（Rectify） ----------
print("\n[STEP] Stereo rectification + undistort maps")
R1, R2, P1r, P2r, Q, roi1, roi2 = cv2.stereoRectify(MLS, dLS, MRS, dRS, img_size, R, T,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map1L,  map2L  = cv2.initUndistortRectifyMap(MLS, dLS, R1, P1r, img_size, cv2.CV_32FC1)
map1LF, map2LF = cv2.initUndistortRectifyMap(MRS, dRS, R2, P2r, img_size, cv2.CV_32FC1)

# 儲存 rectified 投影與重映射表（供後續 2D 偵測 / 三角化）
np.save(os.path.join(output_dir, "P1_rect.npy"), P1r)
np.save(os.path.join(output_dir, "P2_rect.npy"), P2r)
np.save(os.path.join(output_dir, "Q.npy"), Q)
np.save(os.path.join(output_dir, "map1L.npy"), map1L)
np.save(os.path.join(output_dir, "map2L.npy"), map2L)
np.save(os.path.join(output_dir, "map1LF.npy"), map1LF)
np.save(os.path.join(output_dir, "map2LF.npy"), map2LF)
print("[OK] 已輸出 P1_rect/P2_rect/Q 與 map1/2 檔案至 output/")

# ---------- 重投影誤差（單目） ----------
mean_error = 0.0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsLF[i], tvecsLF[i], mtxLF, distLF)
    error = cv2.norm(imgpointsLF[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("\n左前相機重投影誤差(單目平均): {:.6f}".format(mean_error / max(len(objpoints),1)))

mean_error = 0.0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    error = cv2.norm(imgpointsL[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("左相機重投影誤差(單目平均): {:.6f}".format(mean_error / max(len(objpoints),1)))

# ---------- 保存標定結果（JSON） ----------
calibration_result = {
    "左相機內參矩陣": MLS.tolist(),
    "左相機畸變係數": dLS.tolist(),
    "左前相機內參矩陣": MRS.tolist(),
    "左前相機畸變係數": dRS.tolist(),
    "旋轉矩陣": R.tolist(),
    "平移向量": T.reshape(3).tolist(),
    "左相機投影矩陣": P_left.tolist(),
    "左前相機投影矩陣": P_leftfront.tolist(),
    "P1_rect": P1r.tolist(),
    "P2_rect": P2r.tolist(),
    "Q": Q.tolist(),
    "image_size": list(img_size),
    "pattern_size": [10, 7],
    "square_size_mm": square_size
}
with open(f'{output_dir}/calibration_results.json', 'w', encoding='utf-8') as f:
    json.dump(calibration_result, f, indent=2, ensure_ascii=False)
print(f"\n標定結果已保存至 {output_dir}/calibration_results.json")

# ---------- （選讀）最小改動：如何對單點去畸變 ----------
# 若你不做整張 remap/rectify，而是手頭只有像素點 (u,v)：
# - 用 undistortPoints() 取得「規範化座標」
# - 再用 [I|0] 與 [R|T] 做 triangulatePoints（注意：此時不要再乘 K）
#
# ptsL_px  = np.array([[u1,v1],[u2,v2],...], dtype=np.float32)
# ptsLF_px = np.array([[u1,v1],[u2,v2],...], dtype=np.float32)
#
# ptsL_norm  = cv2.undistortPoints(ptsL_px.reshape(-1,1,2),  MLS, dLS).reshape(-1,2)
# ptsLF_norm = cv2.undistortPoints(ptsLF_px.reshape(-1,1,2), MRS, dRS).reshape(-1,2)
# P1n = np.hstack([np.eye(3), np.zeros((3,1))])
# P2n = np.hstack([R, T.reshape(3,1)])
# X_h = cv2.triangulatePoints(P1n, P2n, ptsL_norm.T.astype(np.float64), ptsLF_norm.T.astype(np.float64))
# X   = (X_h[:3] / X_h[3]).T  # (N,3)
