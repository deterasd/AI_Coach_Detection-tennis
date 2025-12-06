import plotly.graph_objects as go
import json

def create_2d_plots(file_path):
    def load_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def extract_coordinates(data, key):
        return [(point['frame'], point[key]['x'], point[key]['y']) 
                for point in data if key in point and point[key]['x'] is not None and point[key]['y'] is not None]
    
    def extract_paddle_points(data):
        """提取球拍的四個keypoints（top/right/bottom/left）"""
        paddle_coords = {k: [] for k in ["top", "right", "bottom", "left"]}
        for point in data:
            if "paddle" in point and isinstance(point["paddle"], dict):
                frame = point.get("frame")
                for k in paddle_coords.keys():
                    if k in point["paddle"]:
                        x = point["paddle"][k].get("x")
                        y = point["paddle"][k].get("y")
                        if x is not None and y is not None:
                            paddle_coords[k].append((frame, x, y))
        return paddle_coords

    # --- Load Data ---
    trajectory_data = load_data(file_path)
    wrist_data = extract_coordinates(trajectory_data, 'right_wrist')
    ball_data = extract_coordinates(trajectory_data, 'tennis_ball')
    paddle_data = extract_paddle_points(trajectory_data)
    
    def get_common_layout(title):
        return dict(
            title=title,
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            width=1800,
            height=800,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                dtick=100,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                minor=dict(
                    dtick=50,
                    gridwidth=0.5,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    ticks='inside'
                ),
            ),
            showlegend=True,
            margin=dict(r=150)
        )

    def create_trace(coords_data, color_scale, line_color, name_label, colorbar_x):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            marker=dict(
                size=5,
                color=frames,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title=f'Frame ({name_label})',
                    x=colorbar_x
                )
            ),
            line=dict(color=line_color, width=2),
            name=name_label
        )

    def create_frame_labels(coords_data, line_color):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='text',
            text=[f'F{frame}' for frame in frames],
            textposition='top center',
            textfont=dict(size=8, color=line_color),
            showlegend=False
        )

    # --- Tennis Ball Trajectory ---
    fig_ball_name = file_path.replace('.json','_2d_ball_trajectory.html')
    fig_ball = go.Figure()
    fig_ball.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.1))
    fig_ball.add_trace(create_frame_labels(ball_data, 'red'))
    fig_ball.update_layout(get_common_layout('Tennis Ball Trajectory'))
    fig_ball.write_html(fig_ball_name)

    # --- Wrist Trajectory ---
    fig_wrist_name = file_path.replace('.json','_2d_wrist_trajectory.html')
    fig_wrist = go.Figure()
    fig_wrist.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_wrist.add_trace(create_frame_labels(wrist_data, 'blue'))
    fig_wrist.update_layout(get_common_layout('Wrist Trajectory'))
    fig_wrist.write_html(fig_wrist_name)

    # --- Paddle 四點 ---
    fig_paddle_name = file_path.replace('.json','_2d_paddle_trajectory.html')
    fig_paddle = go.Figure()
    colors = {"top": "green", "right": "orange", "bottom": "purple", "left": "cyan"}
    for k, coords in paddle_data.items():
        if coords:
            fig_paddle.add_trace(go.Scatter(
                x=[x for _, x, _ in coords],
                y=[y for _, _, y in coords],
                mode='markers+lines',
                marker=dict(size=6, color=colors[k]),
                line=dict(color=colors[k], width=2),
                name=f"Paddle {k}"
            ))
    fig_paddle.update_layout(get_common_layout('Paddle 4 Keypoints Trajectory'))
    fig_paddle.write_html(fig_paddle_name)

    # --- Combined Trajectory ---
    fig_combined = go.Figure()
    fig_combined.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_combined.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.2))

    # 球拍中心：取四點平均作為中心位置
    center_points = []
    for frame_idx in range(len(trajectory_data)):
        paddle = trajectory_data[frame_idx].get("paddle", {})
        if all(k in paddle for k in ["top", "bottom", "left", "right"]):
            xs = [paddle[k]["x"] for k in ["top", "bottom", "left", "right"] if paddle[k]["x"] is not None]
            ys = [paddle[k]["y"] for k in ["top", "bottom", "left", "right"] if paddle[k]["y"] is not None]
            if xs and ys:
                center_points.append((frame_idx, sum(xs)/len(xs), sum(ys)/len(ys)))
    if center_points:
        fig_combined.add_trace(go.Scatter(
            x=[x for _, x, _ in center_points],
            y=[y for _, _, y in center_points],
            mode='lines+markers',
            marker=dict(size=6, color="red"),
            line=dict(color="red", width=2, dash="dot"),
            name="Paddle Center"
        ))

    fig_combined.update_layout(get_common_layout('Combined Wrist, Ball, Paddle Center'))
    fig_combined.write_html(file_path.replace('.json','_2d_combined_trajectory.html'))

if __name__ == "__main__":
    create_2d_plots("凱倫__3_side(2D_trajectory_smoothed).json")


"""
import plotly.graph_objects as go
import json

def create_2d_plots(file_path):
    def load_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def extract_coordinates(data, key, is_paddle=False):
        if not is_paddle:
            return [(point['frame'], point[key]['x'], point[key]['y']) 
                    for point in data if key in point and point[key]['x'] is not None and point[key]['y'] is not None]
        else:
            coords = {k: [] for k in ["top", "right", "bottom", "left", "center"]}
            for point in data:
                for k in coords:
                    if point["paddle"][k]["x"] is not None and point["paddle"][k]["y"] is not None:
                        coords[k].append((point["frame"], point["paddle"][k]["x"], point["paddle"][k]["y"]))
            return coords

    trajectory_data = load_data(file_path)
    wrist_data = extract_coordinates(trajectory_data, 'right_wrist')
    ball_data = extract_coordinates(trajectory_data, 'tennis_ball')
    paddle_data = extract_coordinates(trajectory_data, 'paddle', is_paddle=True)
    
    def get_common_layout(title):
        return dict(
            title=title,
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            width=1800,
            height=800,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                dtick=100,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                minor=dict(
                    dtick=50,
                    gridwidth=0.5,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    ticks='inside'
                ),
            ),
            showlegend=True,
            margin=dict(r=150)
        )

    def create_trace(coords_data, color_scale, line_color, name_label, colorbar_x):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            marker=dict(
                size=5,
                color=frames,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title=f'Frame ({name_label})',
                    x=colorbar_x
                )
            ),
            line=dict(color=line_color, width=2),
            name=name_label
        )

    def create_frame_labels(coords_data, line_color):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='text',
            text=[f'F{frame}' for frame in frames],
            textposition='top center',
            textfont=dict(size=8, color=line_color),
            showlegend=False
        )

    # --- Tennis Ball Trajectory ---
    fig_ball_name = file_path.replace('.json','_2d_ball_trajectory.html')
    fig_ball = go.Figure()
    fig_ball.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.1))
    fig_ball.add_trace(create_frame_labels(ball_data, 'red'))
    fig_ball.update_layout(get_common_layout('Tennis Ball Trajectory'))
    fig_ball.write_html(fig_ball_name)

    # --- Wrist Trajectory ---
    fig_wrist_name = file_path.replace('.json','_2d_wrist_trajectory.html')
    fig_wrist = go.Figure()
    fig_wrist.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_wrist.add_trace(create_frame_labels(wrist_data, 'blue'))
    fig_wrist.update_layout(get_common_layout('Wrist Trajectory'))
    fig_wrist.write_html(fig_wrist_name)

    # --- Paddle 四點 + center ---
    fig_paddle_name = file_path.replace('.json','_2d_paddle_trajectory.html')
    fig_paddle = go.Figure()
    colors = {"top": "blue", "right": "blue", "bottom": "blue", "left": "blue", "center": "red"}
    for k, coords in paddle_data.items():
        if coords:
            fig_paddle.add_trace(go.Scatter(
                x=[x for _, x, _ in coords],
                y=[y for _, _, y in coords],
                mode='markers+lines',
                marker=dict(size=6, color=colors[k]),
                line=dict(color=colors[k], width=2, dash="dot" if k=="center" else "solid"),
                name=f"Paddle {k}"
            ))
    fig_paddle.update_layout(get_common_layout('Paddle Trajectory'))
    fig_paddle.write_html(fig_paddle_name)

    # --- Combined Trajectory ---
    fig_combined = go.Figure()
    fig_combined.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_combined.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.2))
    # 加入球拍中心 (簡化只畫 center，避免太複雜)
    if paddle_data["center"]:
        fig_combined.add_trace(go.Scatter(
            x=[x for _, x, _ in paddle_data["center"]],
            y=[y for _, _, y in paddle_data["center"]],
            mode='lines+markers',
            marker=dict(size=6, color="red"),
            line=dict(color="red", width=2, dash="dot"),
            name="Paddle Center"
        ))
    fig_combined.update_layout(get_common_layout('Combined Wrist, Ball, Paddle'))
    fig_combined.write_html(file_path.replace('.json','_2d_combined_trajectory.html'))

if __name__ == "__main__":
    create_2d_plots("凱倫__3_side(2D_trajectory_smoothed).json")
"""

"""
import plotly.graph_objects as go
import json

def create_2d_plots(file_path):
    def load_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
            
    def extract_coordinates(data, key):
        return [(point['frame'], point[key]['x'], point[key]['y']) 
                for point in data if key in point and point[key]['x'] is not None and point[key]['y'] is not None]

    trajectory_data = load_data(file_path)
    wrist_data = extract_coordinates(trajectory_data, 'right_wrist')
    ball_data = extract_coordinates(trajectory_data, 'tennis_ball')
    
    def get_common_layout(title):
        return dict(
            title=title,
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            width=1800,
            height=800,
            yaxis=dict(autorange="reversed"),
            xaxis=dict(
                dtick=100,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                minor=dict(
                    dtick=50,
                    gridwidth=0.5,
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    ticks='inside'
                ),
            ),
            showlegend=True,
            margin=dict(r=150)
        )

    def create_trace(coords_data, color_scale, line_color, name_label, colorbar_x):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            marker=dict(
                size=5,
                color=frames,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title=f'Frame ({name_label})',
                    x=colorbar_x
                )
            ),
            line=dict(color=line_color, width=2),
            name=name_label
        )

    def create_frame_labels(coords_data, line_color):
        frames = [frame for frame, _, _ in coords_data]
        x_coords = [x for _, x, _ in coords_data]
        y_coords = [y for _, _, y in coords_data]
        
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='text',
            text=[f'F{frame}' for frame in frames],
            textposition='top center',
            textfont=dict(size=8, color=line_color),
            showlegend=False
        )

    # Tennis Ball Trajectory
    fig_ball_name = file_path.replace('.json','_2d_ball_trajectory.html')
    fig_ball = go.Figure()
    fig_ball.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.1))
    fig_ball.add_trace(create_frame_labels(ball_data, 'red'))
    fig_ball.update_layout(get_common_layout('Tennis Ball Trajectory'))
    fig_ball.write_html(fig_ball_name)
    # fig_ball.show()

    # Wrist Trajectory
    fig_wrist_name = file_path.replace('.json','_2d_wrist_trajectory.html')
    fig_wrist = go.Figure()
    fig_wrist.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_wrist.add_trace(create_frame_labels(wrist_data, 'blue'))
    fig_wrist.update_layout(get_common_layout('Wrist Trajectory'))
    fig_wrist.write_html(fig_wrist_name)
    fig_wrist.show()

    # Combined Trajectory
    fig_combined = go.Figure()
    fig_combined.add_trace(create_trace(wrist_data, 'Viridis', 'blue', 'Wrist', 1.1))
    fig_combined.add_trace(create_trace(ball_data, 'Plasma', 'red', 'Tennis Ball', 1.2))
    fig_combined.update_layout(get_common_layout('Combined Wrist and Tennis Ball Trajectory'))
    fig_combined.write_html('2d_combined_trajectory.html')
    fig_combined.show()

if __name__ == "__main__":
    create_2d_plots("凱倫__3_side(2D_trajectory_smoothed).json")

"""