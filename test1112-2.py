import json
import numpy as np

def validate_3d_reconstruction(json_3d_path, json_2d_side_path, json_2d_45_path, P1, P2):
    """
    驗證3D重建：計算重投影誤差
    
    參數:
        json_3d_path: 3D軌跡JSON路徑
        json_2d_side_path: Side視角2D軌跡路徑
        json_2d_45_path: 45度視角2D軌跡路徑
        P1: Side視角投影矩陣 (3x4)
        P2: 45度視角投影矩陣 (3x4)
    
    返回:
        dict: 包含各關節點的誤差統計
    """
    # 讀取數據
    with open(json_3d_path, 'r') as f:
        data_3d = json.load(f)
    with open(json_2d_side_path, 'r') as f:
        data_2d_side = json.load(f)
    with open(json_2d_45_path, 'r') as f:
        data_2d_45 = json.load(f)
    
    keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle', 'tennis_ball'
    ]
    
    errors_side = {kp: [] for kp in keypoints}
    errors_45 = {kp: [] for kp in keypoints}
    
    for frame_idx in range(len(data_3d)):
        frame_3d = data_3d[frame_idx]
        frame_2d_side = data_2d_side[frame_idx]
        frame_2d_45 = data_2d_45[frame_idx]
        
        for keypoint in keypoints:
            # 檢查3D點是否有效
            if (frame_3d[keypoint]['x'] is None or 
                frame_3d[keypoint]['y'] is None or 
                frame_3d[keypoint]['z'] is None):
                continue
            
            # 3D點（注意：程式中y已經取負，這裡要還原）
            point_3d = np.array([
                frame_3d[keypoint]['x'],
                -frame_3d[keypoint]['y'],  # 還原y的符號
                frame_3d[keypoint]['z'],
                1.0  # 齊次坐標
            ])
            
            # 計算Side視角的重投影誤差
            if (frame_2d_side[keypoint]['x'] is not None and 
                frame_2d_side[keypoint]['y'] is not None):
                
                point_2d_original = np.array([
                    frame_2d_side[keypoint]['x'],
                    frame_2d_side[keypoint]['y']
                ])
                
                # 重投影
                point_2d_proj = P1 @ point_3d
                point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                
                # 計算誤差
                error = np.linalg.norm(point_2d_original - point_2d_proj)
                errors_side[keypoint].append(error)
            
            # 計算45度視角的重投影誤差
            if (frame_2d_45[keypoint]['x'] is not None and 
                frame_2d_45[keypoint]['y'] is not None):
                
                point_2d_original = np.array([
                    frame_2d_45[keypoint]['x'],
                    frame_2d_45[keypoint]['y']
                ])
                
                # 重投影
                point_2d_proj = P2 @ point_3d
                point_2d_proj = point_2d_proj[:2] / point_2d_proj[2]
                
                # 計算誤差
                error = np.linalg.norm(point_2d_original - point_2d_proj)
                errors_45[keypoint].append(error)
    
    # 統計結果
    print("=" * 80)
    print("重投影誤差驗證結果")
    print("=" * 80)
    print(f"{'關節點':<20} {'Side視角 (pixels)':<25} {'45度視角 (pixels)':<25}")
    print("-" * 80)
    
    all_errors_side = []
    all_errors_45 = []
    
    for keypoint in keypoints:
        if errors_side[keypoint]:
            mean_side = np.mean(errors_side[keypoint])
            std_side = np.std(errors_side[keypoint])
            all_errors_side.extend(errors_side[keypoint])
        else:
            mean_side = std_side = 0
        
        if errors_45[keypoint]:
            mean_45 = np.mean(errors_45[keypoint])
            std_45 = np.std(errors_45[keypoint])
            all_errors_45.extend(errors_45[keypoint])
        else:
            mean_45 = std_45 = 0
        
        print(f"{keypoint:<20} {mean_side:>6.2f} ± {std_side:>5.2f}          {mean_45:>6.2f} ± {std_45:>5.2f}")
    
    print("-" * 80)
    print(f"{'總體平均':<20} {np.mean(all_errors_side):>6.2f} ± {np.std(all_errors_side):>5.2f}          {np.mean(all_errors_45):>6.2f} ± {np.std(all_errors_45):>5.2f}")
    print("=" * 80)
    
    # 判斷結果
    avg_error = (np.mean(all_errors_side) + np.mean(all_errors_45)) / 2
    print(f"\n平均重投影誤差: {avg_error:.2f} pixels")
    
    if avg_error < 5:
        print("✅ 重建品質: 優秀 (< 5 pixels)")
    elif avg_error < 10:
        print("⚠️  重建品質: 良好 (5-10 pixels)")
    elif avg_error < 20:
        print("⚠️  重建品質: 可接受 (10-20 pixels)")
    else:
        print("❌ 重建品質: 需要改進 (> 20 pixels)")
    
    return {
        'errors_side': errors_side,
        'errors_45': errors_45,
        'mean_side': np.mean(all_errors_side),
        'mean_45': np.mean(all_errors_45),
        'overall_mean': avg_error
    }

if __name__ == "__main__":
    # 範例一
    """P1 = np.array([
         [  682.525930,     0.000000,   637.087464,     0.000000],
         [    0.000000,   684.519186,   360.040032,     0.000000],
         [    0.000000,     0.000000,     1.000000,     0.000000],
     ])

    P2 = np.array([
         [  572.714085,    27.508121,   700.861208, -159410.297486],
         [  -27.187871,   663.411218,   332.613656, 51347.982959],
         [   -0.102197,     0.053920,     0.993302,    73.630082],
     ])
    """
    # # 驗證
    # results = validate_3d_reconstruction(
    #     'json/hsiao2__(3D_trajectory_smoothed).json',
    #     'json/hsiao2__side(2D_trajectory_smoothed).json',
    #     'json/hsiao2__45(2D_trajectory_smoothed).json',
    #     P1, P2
    # )

    
    # 範例二
    P1 = np.array([
 
   [  613.902729,     0.000000,   638.203915,     0.000000],
    [    0.000000,   617.251817,   364.556522,     0.000000],
    [    0.000000,     0.000000,     1.000000,     0.000000],

    ])

    P2 = np.array([
        [  616.071259,     7.588060,   617.077541, 154727.853092],
    [   -0.773895,   591.674918,   358.669757, -16209.393573],
    [    0.038272,    -0.010667,     0.999210,   -68.044380],
    ])
    
    # 驗證
    results = validate_3d_reconstruction(
        'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/trajectory/hsiao2__trajectory/trajectory__60/hsiao2__60_45(3D_trajectory).json',
        'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/trajectory/hsiao2__trajectory/trajectory__60/hsiao2__60_side(2D_trajectory).json',
        'C:/Users/chen/Desktop/pickleball-version1/Pickleball_Project/trajectory/hsiao2__trajectory/trajectory__60/hsiao2__60_45(2D_trajectory).json',
        P1, P2
    )