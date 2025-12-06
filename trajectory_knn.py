
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
def analyze_trajectory(knn_dataset, input_3d_json, n_neighbors=1):
    """
    使用 KNN 比對使用者的 3D 軌跡與資料庫
    這裡只使用 paddle_center、球、以及右手腕作為比對依據
    """

    # --- 載入資料庫 ---
    with open(knn_dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    X = []
    labels = []
    for entry in dataset:
        frames = entry.get('data') or entry.get('trajectory', [])
        label = entry.get('suggestion') or entry.get('label', 'unknown')

        # 特徵：右手腕 + 球 + 球拍中心的統計特徵
        rw_x, rw_y, rw_z = [], [], []
        ball_x, ball_y, ball_z = [], [], []
        paddle_x, paddle_y, paddle_z = [], [], []
        
        for frame in frames:
            rw = frame.get('right_wrist', {})
            ball = frame.get('tennis_ball', {})
            paddle = frame.get('paddle_center', {})
            
            if rw.get('x') is not None:
                rw_x.append(rw['x'])
                rw_y.append(rw['y'])
                rw_z.append(rw['z'])
            
            if ball.get('x') is not None:
                ball_x.append(ball['x'])
                ball_y.append(ball['y'])
                ball_z.append(ball['z'])
            
            if paddle.get('x') is not None:
                paddle_x.append(paddle['x'])
                paddle_y.append(paddle['y'])
                paddle_z.append(paddle['z'])
        
        # 計算統計特徵：平均值和標準差
        feature = []
        for data_list in [rw_x, rw_y, rw_z, ball_x, ball_y, ball_z, paddle_x, paddle_y, paddle_z]:
            if len(data_list) > 0:
                feature.extend([np.mean(data_list), np.std(data_list)])
            else:
                feature.extend([0.0, 0.0])
        
        X.append(feature)
        labels.append(label)

    X = np.array(X, dtype=float)

    # --- 載入使用者的輸入檔 ---
    with open(input_3d_json, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    rw_x, rw_y, rw_z = [], [], []
    ball_x, ball_y, ball_z = [], [], []
    paddle_x, paddle_y, paddle_z = [], [], []
    
    for frame in user_data:
        rw = frame.get('right_wrist', {})
        ball = frame.get('tennis_ball', {})
        paddle = frame.get('paddle_center', {})
        
        if rw.get('x') is not None:
            rw_x.append(rw['x'])
            rw_y.append(rw['y'])
            rw_z.append(rw['z'])
        
        if ball.get('x') is not None:
            ball_x.append(ball['x'])
            ball_y.append(ball['y'])
            ball_z.append(ball['z'])
        
        if paddle.get('x') is not None:
            paddle_x.append(paddle['x'])
            paddle_y.append(paddle['y'])
            paddle_z.append(paddle['z'])

    user_feature = []
    for data_list in [rw_x, rw_y, rw_z, ball_x, ball_y, ball_z, paddle_x, paddle_y, paddle_z]:
        if len(data_list) > 0:
            user_feature.extend([np.mean(data_list), np.std(data_list)])
        else:
            user_feature.extend([0.0, 0.0])
    
    user_feature = np.array(user_feature, dtype=float).reshape(1, -1)

    # --- KNN ---
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(user_feature)

    results = []
    for idx in indices[0]:
        results.append(labels[idx])

    return results

if __name__ == "__main__":
    knn_dataset = "knn_dataset.json"
    input_file = "trajectory__1(3D_trajectory_smoothed).json"

    results = analyze_trajectory(knn_dataset, input_file, n_neighbors=3)
    print("KNN 比對結果:", results)

"""
import json
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def normalize_data(dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee", "paddle"]
    data_matrix = {key: [] for key in keys}
    trajectory_frames = None
    
    # Extract trajectory frames and collect data for normalization
    for data in dataset:
        if data["filename"] == trajectory_filename:
            trajectory_frames = data["data"]
        
        for frame in data["data"]:
            for key in keys:
                if key in frame and all(k in frame[key] for k in ["x", "y", "z"]):
                    data_matrix[key].append([frame[key]["x"], frame[key]["y"], frame[key]["z"]])
    
    # Normalize data
    scalers = {key: MinMaxScaler() for key in keys}
    normalized_data = {key: scalers[key].fit_transform(data_matrix[key]) for key in keys}
    
    # Update original dataset with normalized values
    frame_idx = 0
    for data in dataset:
        for frame in data["data"]:
            for key in keys:
                if key in frame:
                    frame[key]["x"], frame[key]["y"], frame[key]["z"] = normalized_data[key][frame_idx]
            frame_idx += 1
            
    return trajectory_frames

def find_nearest_suggestion(trajectory_frames, merged_dataset, trajectory_filename):
    keys = ["nose", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_knee", "right_knee"]
    filename_distances = {}
    
    for data in merged_dataset:
        # Skip self and non-hitting suggestions
        if data["filename"] == trajectory_filename or "是否擊球:否" in data.get("suggestion", ""):
            continue
            
        total_distance = 0
        valid_comparisons = 0
        min_frame_count = min(len(trajectory_frames), len(data["data"]))
        
        # Calculate distances between frames
        for i in range(min_frame_count):
            traj_frame = trajectory_frames[i]
            ref_frame = data["data"][i]
            distance = 0
            
            for key in keys:
                if key in traj_frame and key in ref_frame:
                    traj_point = traj_frame[key]
                    ref_point = ref_frame[key]
                    
                    if any(v is None for v in traj_point.values()) or any(v is None for v in ref_point.values()):
                        continue
                        
                    # Euclidean distance
                    traj_array = np.array([traj_point["x"], traj_point["y"], traj_point["z"]])
                    ref_array = np.array([ref_point["x"], ref_point["y"], ref_point["z"]])
                    distance += np.linalg.norm(traj_array - ref_array)
                    
            total_distance += distance
            valid_comparisons += 1
            
        if valid_comparisons > 0:
            filename_distances[data["filename"]] = (total_distance / valid_comparisons, data.get("suggestion", "None"))
    
    if not filename_distances:
        print("No valid comparison data")
        return "None"
        
    best_filename = min(filename_distances, key=lambda x: filename_distances[x][0])
    print(f"Closest file: {best_filename}")
    return filename_distances[best_filename][1]

def analyze_trajectory(merged_dataset_path, dynamic_filename):
    # Load datasets
    merged_dataset = load_json(merged_dataset_path)
    trajectory_data = load_json(dynamic_filename)
    
    # Add trajectory data to merged dataset
    merged_dataset.append({
        "filename": dynamic_filename,
        "level": "unknown",
        "suggestion": "None",
        "data": trajectory_data
    })
    
    # Process data
    normalized_trajectory = normalize_data(merged_dataset, dynamic_filename)
    best_suggestion = find_nearest_suggestion(normalized_trajectory, merged_dataset, dynamic_filename)
    
    # Save suggestion to file
    suggestion_path = dynamic_filename.replace('(3D_trajectory_smoothed).json', '_knn_feedback.txt')
    with open(suggestion_path, "w", encoding="utf-8") as file:
        file.write(best_suggestion)
    
    # Print result
    return suggestion_path

if __name__ == "__main__":
    merged_dataset_path = "knn_dataset.json"
    dynamic_filename = "trajectory/嘉洋__trajectory/trajectory__3/嘉洋__3(3D_trajectory_smoothed).json"
    
    # Add timer
    start_time = time.time()
    suggestion_path = analyze_trajectory(merged_dataset_path, dynamic_filename)
    end_time = time.time()
    
    # Print execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    """