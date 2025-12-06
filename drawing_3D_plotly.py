import plotly.graph_objects as go
import json
import numpy as np

def create_3d_plots(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        trajectory_data = json.load(f)

    total_frames = len(trajectory_data)
    print(f"[INFO] 已載入 {total_frames} 幀 3D 資料")

    # ----------------------------
    # 顏色設定
    # ----------------------------
    joints = {
        'tennis_ball': '#ff0000',
        'nose': '#00ff00',
        'left_eye': '#0000ff',
        'right_eye': '#00ffff',
        'left_ear': '#ff00ff',
        'right_ear': '#ffff00',
        'left_shoulder': '#800000',
        'right_shoulder': '#008000',
        'left_elbow': '#000080',
        'right_elbow': '#808000',
        'left_wrist': '#800080',
        'right_wrist': '#008080',
        'left_hip': '#ff8000',
        'right_hip': '#0080ff',
        'left_knee': '#ff0080',
        'right_knee': '#80ff00',
        'left_ankle': '#8000ff',
        'right_ankle': '#00ff80'
    }

    paddle_points = ['top', 'right', 'bottom', 'left', 'center']
    paddle_color = "#ff7f0e"

    skeleton_connections = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('left_ear', 'left_shoulder'), ('right_ear', 'right_shoulder'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
    ]

    # ----------------------------
    # 收集座標資料
    # ----------------------------
    all_x, all_y, all_z = [], [], []
    ball_x, ball_y, ball_z = [], [], []
    right_wrist_x, right_wrist_y, right_wrist_z = [], [], []
    paddle_center_x, paddle_center_y, paddle_center_z = [], [], []
    frame_labels = []

    for frame_idx, frame in enumerate(trajectory_data):
        for joint in joints:
            if (frame[joint]['x'] is not None and
                frame[joint]['y'] is not None and
                frame[joint]['z'] is not None):
                all_x.append(frame[joint]['x'])
                all_y.append(frame[joint]['y'])
                all_z.append(frame[joint]['z'])

                if joint == 'tennis_ball':
                    ball_x.append(frame[joint]['x'])
                    ball_y.append(frame[joint]['z'])
                    ball_z.append(frame[joint]['y'])
                    frame_labels.append(f'F{frame_idx}')
                elif joint == 'right_wrist':
                    right_wrist_x.append(frame[joint]['x'])
                    right_wrist_y.append(frame[joint]['z'])
                    right_wrist_z.append(frame[joint]['y'])

        # 球拍中心點
        if 'paddle' in frame and frame['paddle'].get('center', {}).get('x') is not None:
            paddle_center_x.append(frame['paddle']['center']['x'])
            paddle_center_y.append(frame['paddle']['center']['z'])
            paddle_center_z.append(frame['paddle']['center']['y'])

    # ----------------------------
    # 初始化 3D 圖形
    # ----------------------------
    fig = go.Figure()

    # --- 畫球場地板 ---
    court_length, court_width, nvz = 13.41, 6.10, 2.13
    xx, yy = np.meshgrid(
        np.linspace(-court_width / 2, court_width / 2, 2),
        np.linspace(0, court_length, 2)
    )
    zz = np.zeros_like(xx)
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, "green"], [1, "green"]],
        opacity=0.6,
        showscale=False
    ))

    # --- 球的軌跡 ---
    fig.add_trace(go.Scatter3d(
        x=ball_x, y=ball_y, z=ball_z,
        mode='lines+markers+text',
        name='Ball Trajectory',
        line=dict(color='red', width=3),
        text=frame_labels,
        textposition='top center',
        textfont=dict(size=8),
        showlegend=True,
    ))

    # --- 球拍中心軌跡 ---
    fig.add_trace(go.Scatter3d(
        x=paddle_center_x, y=paddle_center_y, z=paddle_center_z,
        mode='lines',
        name='Paddle Swing Trajectory',
        line=dict(color=paddle_color, width=4),
        showlegend=True,
    ))

    # ----------------------------
    # 動畫 Frame
    # ----------------------------
    frames = []
    for frame_idx, frame in enumerate(trajectory_data):
        frame_data = []

        # 球
        frame_data.append(go.Scatter3d(
            x=ball_x, y=ball_y, z=ball_z,
            mode='lines+markers+text',
            line=dict(color='red', width=2),
            text=frame_labels,
            textposition='top center',
            textfont=dict(size=8),
            name='Ball Trajectory',
            showlegend=(frame_idx == 0)
        ))

        # 球拍五點
        if 'paddle' in frame:
            for p in paddle_points:
                pt = frame['paddle'].get(p, None)
                if pt and pt['x'] is not None:
                    frame_data.append(go.Scatter3d(
                        x=[pt['x']], y=[pt['z']], z=[pt['y']],
                        mode='markers',
                        marker=dict(size=6 if p != 'center' else 10, color=paddle_color, opacity=0.9),
                        name=f"paddle_{p}",
                        showlegend=False
                    ))

        # joints
        for joint_name, color in joints.items():
            if (frame[joint_name]['x'] is not None and
                frame[joint_name]['y'] is not None and
                frame[joint_name]['z'] is not None):
                frame_data.append(go.Scatter3d(
                    x=[frame[joint_name]['x']],
                    y=[frame[joint_name]['z']],
                    z=[frame[joint_name]['y']],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.8
                    ),
                    showlegend=False
                ))

        # skeleton lines
        for start_joint, end_joint in skeleton_connections:
            if (frame[start_joint]['x'] is not None and frame[end_joint]['x'] is not None):
                frame_data.append(go.Scatter3d(
                    x=[frame[start_joint]['x'], frame[end_joint]['x']],
                    y=[frame[start_joint]['z'], frame[end_joint]['z']],
                    z=[frame[start_joint]['y'], frame[end_joint]['y']],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.8)', width=2),
                    showlegend=False
                ))

        frames.append(go.Frame(data=frame_data, name=f"frame_{frame_idx}"))

    # ----------------------------
    # Layout
    # ----------------------------
    fig.frames = frames
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Z', zaxis_title='Y',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 40, 'redraw': True},
                                 'fromcurrent': True, 'mode': 'immediate'}],
                 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                   'mode': 'immediate'}],
                 'label': 'Pause', 'method': 'animate'}
            ],
            'type': 'buttons', 'showactive': True,
            'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'
        }],
        width=1100, height=850,
        title='3D Pickleball Body + Paddle Trajectory Visualization'
    )

    output_path = data_file.replace('.json', '_3Dplot.html')
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ 已輸出：{output_path}")

if __name__ == "__main__":
    create_3d_plots("凱倫__1(3D_trajectory_smoothed)_only_swing.json")
