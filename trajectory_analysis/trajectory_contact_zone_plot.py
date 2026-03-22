import json
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
# 確保以腳本直接執行時也能匯入同專案內的套件
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from trajectory_analysis.trajectory_contact_zone_eval import (
    analyze_contact_zone,
    _collect_pro_contact_metrics,
    _load_json,
)


def plot_contact_zone_3D(pro_metrics, user_m, pro_stats, save_path):
    fig = go.Figure()

    # 提取 pro 資料與 filename
    xs = [m["depth_n"] for m in pro_metrics]
    ys = [m["lateral_n"] for m in pro_metrics]
    zs = [m["height_n"] for m in pro_metrics]
    filenames = [m.get("filename", "unknown") for m in pro_metrics]
    
    # Pro 散點（hover 顯示 filename）
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.4),
        name='Pro Range',
        text=filenames,
        hovertemplate='<b>%{text}</b><br>' +
                      'Depth: %{x:.3f}<br>' +
                      'Lateral: %{y:.3f}<br>' +
                      'Height: %{z:.3f}<extra></extra>',
    ))

    # 使用者擊球點（3D 不支援 star，改用 diamond 並加大尺寸）
    fig.add_trace(go.Scatter3d(
        x=[user_m["depth_n"]], y=[user_m["lateral_n"]], z=[user_m["height_n"]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        name='User Impact',
        hovertemplate='<b>User Impact</b><br>' +
                      'Depth: %{x:.3f}<br>' +
                      'Lateral: %{y:.3f}<br>' +
                      'Height: %{z:.3f}<extra></extra>',
    ))

    # 繪製區間方盒（用 8 個頂點）
    d_lo, d_hi = pro_stats["depth_n"]["lo"], pro_stats["depth_n"]["hi"]
    l_lo, l_hi = pro_stats["lateral_n"]["lo"], pro_stats["lateral_n"]["hi"]
    h_lo, h_hi = pro_stats["height_n"]["lo"], pro_stats["height_n"]["hi"]
    
    box_x = [d_lo, d_hi, d_hi, d_lo, d_lo, d_hi, d_hi, d_lo]
    box_y = [l_lo, l_lo, l_hi, l_hi, l_lo, l_lo, l_hi, l_hi]
    box_z = [h_lo, h_lo, h_lo, h_lo, h_hi, h_hi, h_hi, h_hi]
    
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='markers',
        marker=dict(size=1, color='cyan', opacity=0.1),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Depth (前後)',
            yaxis_title='Lateral (遠近)',
            zaxis_title='Height (上下)',
        ),
        title='擊球點區域分析 - 3D 視圖',
        width=900,
        height=800,
        margin=dict(l=0, r=0, t=50, b=50),  # 增加頂部與底部空間，確保標題與圖表不被裁切
    )
    
    fig.write_html(str(save_path))


def plot_contact_zone_2D(pro_metrics, user_m, pro_stats, save_path):
    pairs = [("depth_n", "lateral_n"), ("depth_n", "height_n"), ("lateral_n", "height_n")]
    titles = ["Depth vs Lateral", "Depth vs Height", "Lateral vs Height"]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.1,
    )
    
    filenames = [m.get("filename", "unknown") for m in pro_metrics]
    
    for idx, ((xk, yk), title) in enumerate(zip(pairs, titles), 1):
        x = [m[xk] for m in pro_metrics]
        y = [m[yk] for m in pro_metrics]
        
        # Pro 散點
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.4),
            name='Pro Range' if idx == 1 else '',
            text=filenames,
            hovertemplate='<b>%{text}</b><br>' +
                          f'{xk}: %{{x:.3f}}<br>' +
                          f'{yk}: %{{y:.3f}}<extra></extra>',
            showlegend=(idx == 1),
        ), row=1, col=idx)
        
        # 使用者擊球點
        fig.add_trace(go.Scatter(
            x=[user_m[xk]], y=[user_m[yk]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='star'),
            name='User Impact' if idx == 1 else '',
            hovertemplate='<b>User Impact</b><br>' +
                          f'{xk}: %{{x:.3f}}<br>' +
                          f'{yk}: %{{y:.3f}}<extra></extra>',
            showlegend=(idx == 1),
        ), row=1, col=idx)
        
        # 區間陰影（使用矩形形狀）
        x_lo, x_hi = pro_stats[xk]["lo"], pro_stats[xk]["hi"]
        y_lo, y_hi = pro_stats[yk]["lo"], pro_stats[yk]["hi"]
        
        fig.add_shape(
            type="rect",
            x0=x_lo, x1=x_hi, y0=y_lo, y1=y_hi,
            fillcolor="cyan", opacity=0.1, line_width=0,
            row=1, col=idx,
        )

        # 區間延伸線（從區間邊界向外延伸）
        # 以資料的最小/最大值為邊界，將邊界線延伸到整個視圖，方便觀察超出程度
        if len(x) > 0 and len(y) > 0:
            x_min, x_max = (min(x), max(x))
            y_min, y_max = (min(y), max(y))

            # 垂直延伸線：x = x_lo 與 x = x_hi，覆蓋 y 範圍
            fig.add_shape(type="line", x0=x_lo, x1=x_lo, y0=y_min, y1=y_max,
                          line=dict(color="cyan", width=1, dash="dot"), row=1, col=idx)
            fig.add_shape(type="line", x0=x_hi, x1=x_hi, y0=y_min, y1=y_max,
                          line=dict(color="cyan", width=1, dash="dot"), row=1, col=idx)

            # 水平延伸線：y = y_lo 與 y = y_hi，覆蓋 x 範圍
            fig.add_shape(type="line", x0=x_min, x1=x_max, y0=y_lo, y1=y_lo,
                          line=dict(color="cyan", width=1, dash="dot"), row=1, col=idx)
            fig.add_shape(type="line", x0=x_min, x1=x_max, y0=y_hi, y1=y_hi,
                          line=dict(color="cyan", width=1, dash="dot"), row=1, col=idx)
        
        fig.update_xaxes(title_text=xk, row=1, col=idx)
        fig.update_yaxes(title_text=yk, row=1, col=idx)
    
    fig.update_layout(
        title='擊球點區域分析 - 2D 視圖',
        width=1500,
        height=500,
    )
    
    fig.write_html(str(save_path))


def visualize_contact_zone(result_json, save_dir="output"):
    data = json.loads(Path(result_json).read_text())

    if not all(k in data for k in ["user_metrics", "pro_range"]):
        raise ValueError("JSON 資料不包含繪圖所需欄位。請先執行 trajectory_contact_zone_eval_v3.py。")

    user_m = data["user_metrics"]
    pro_stats = data["pro_range"]

    # 讀取 Pro metrics (可選：若前次分析有保存)
    pro_file = Path(save_dir) / "pro_metrics.json"
    if pro_file.exists():
        pro_metrics = json.loads(pro_file.read_text())
    else:
        raise FileNotFoundError("找不到 pro_metrics.json，請從分析模組輸出該檔案後再繪圖。")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    plot_contact_zone_3D(pro_metrics, user_m, pro_stats, save_dir / "contact_zone_plot_3D.png")
    plot_contact_zone_2D(pro_metrics, user_m, pro_stats, save_dir / "contact_zone_plot_2D.png")
    print(f"✅ 已輸出 3D 與 2D 可視化圖至 {save_dir}")


if __name__ == "__main__":
    # 直接根據分析模組計算並繪圖（無需事先輸出 JSON）
    knn_path = "knn_dataset_new.json"
    traj_path = "trajectory/testing_123/testing_(3D_trajectory_smoothed).json"
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 1) 取得分析結果（使用者擊球點與專家分佈區間）
    res = analyze_contact_zone(knn_path, traj_path)
    user_m = res.get("user_values", {})
    pro_ranges = res.get("pro_ranges", {})
    # 轉換為繪圖所需鍵名
    pro_stats = {
        "depth_n": {"lo": pro_ranges.get("depth_n", {}).get("p10", 0.0), "hi": pro_ranges.get("depth_n", {}).get("p90", 0.0)},
        "lateral_n": {"lo": pro_ranges.get("lateral_n", {}).get("p10", 0.0), "hi": pro_ranges.get("lateral_n", {}).get("p90", 0.0)},
        "height_n": {"lo": pro_ranges.get("height_n", {}).get("p10", 0.0), "hi": pro_ranges.get("height_n", {}).get("p90", 0.0)},
    }

    # 2) 收集專家擊球幀散點（所有 pro 的 tennis_ball_hit=True 幀）
    dataset = _load_json(knn_path)
    pro_metrics = _collect_pro_contact_metrics(dataset)

    # 3) 輸出互動式 HTML 圖表
    plot_contact_zone_3D(pro_metrics, user_m, pro_stats, out_dir / "contact_zone_plot_3D.html")
    plot_contact_zone_2D(pro_metrics, user_m, pro_stats, out_dir / "contact_zone_plot_2D.html")
    print(f"✅ 已輸出 3D 與 2D 互動式 HTML 圖表至 {out_dir}")
    print(f"   開啟 HTML 檔案後，滑鼠 hover 到每個點即可看到對應的 filename")
