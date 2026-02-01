#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D軌跡重建誤差分析工具 (GUI版本)
用於分析雙鏡頭系統的3D重建精度

作者: AI Coach Detection Team
日期: 2025-12-10
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import sys
import os

class TrajectoryAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎾 3D軌跡重建誤差分析工具")
        self.root.geometry("900x700")

        # 變數初始化
        self.path_3d = tk.StringVar()
        self.path_2d_45 = tk.StringVar()
        self.path_2d_side = tk.StringVar()

        # 投影矩陣 (預設值)
        self.P1 = np.array([
            [1185.469598, 0.000000, 956.591700, 0.000000],
            [0.000000, 1190.259956, 545.354948, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ], dtype=float)

        self.P2 = np.array([
            [892.314977, -32.441114, 1097.323442, -693191.663377],
            [-61.430127, 1034.208940, 556.228128, -104273.203958],
            [-0.140877, -0.015579, 0.989905, -187.617212]
        ], dtype=float)

        self.setup_ui()

    def setup_ui(self):
        # 創建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 檔案選擇區域
        file_frame = ttk.LabelFrame(main_frame, text="📁 檔案選擇", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # 3D軌跡檔案
        ttk.Label(file_frame, text="3D軌跡檔案:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.path_3d, width=60).grid(row=0, column=1, padx=(5, 0), pady=2)
        ttk.Button(file_frame, text="瀏覽...", command=lambda: self.browse_file(self.path_3d)).grid(row=0, column=2, padx=(5, 0), pady=2)

        # 45度2D軌跡檔案
        ttk.Label(file_frame, text="45度2D軌跡:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.path_2d_45, width=60).grid(row=1, column=1, padx=(5, 0), pady=2)
        ttk.Button(file_frame, text="瀏覽...", command=lambda: self.browse_file(self.path_2d_45)).grid(row=1, column=2, padx=(5, 0), pady=2)

        # 側面2D軌跡檔案
        ttk.Label(file_frame, text="側面2D軌跡:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.path_2d_side, width=60).grid(row=2, column=1, padx=(5, 0), pady=2)
        ttk.Button(file_frame, text="瀏覽...", command=lambda: self.browse_file(self.path_2d_side)).grid(row=2, column=2, padx=(5, 0), pady=2)

        # 控制按鈕
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        ttk.Button(button_frame, text="🚀 開始分析", command=self.start_analysis).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="📋 載入範例", command=self.load_example).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="🧹 清除", command=self.clear_all).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(button_frame, text="❌ 結束", command=self.root.quit).grid(row=0, column=3)

        # 進度條
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # 狀態標籤
        self.status_var = tk.StringVar(value="就緒")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, pady=(5, 0))

        # 結果顯示區域
        result_frame = ttk.LabelFrame(main_frame, text="📊 分析結果", padding="5")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        # 創建notebook用於分頁顯示結果
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 總結頁面
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="📈 總結")

        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, height=15)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 詳細頁面
        detail_frame = ttk.Frame(self.notebook)
        self.notebook.add(detail_frame, text="📋 詳細資料")

        self.detail_text = scrolledtext.ScrolledText(detail_frame, wrap=tk.WORD, height=15)
        self.detail_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 設定grid權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)

    def browse_file(self, path_var):
        """瀏覽並選擇檔案"""
        filename = filedialog.askopenfilename(
            title="選擇軌跡檔案",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            path_var.set(filename)

    def load_example(self):
        """載入範例檔案路徑"""
        try:
            # 嘗試在trajectory目錄中尋找範例檔案
            trajectory_dir = Path("trajectory")
            if trajectory_dir.exists():
                # 尋找最新的trajectory資料夾
                subdirs = [d for d in trajectory_dir.iterdir() if d.is_dir()]
                if subdirs:
                    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
                    trajectory_subdir = latest_dir / "trajectory_1"

                    # 尋找對應的檔案
                    files_3d = list(trajectory_subdir.glob("*3D_trajectory_smoothed*.json"))
                    files_45 = list(trajectory_subdir.glob("*45*2D_trajectory_smoothed*.json"))
                    files_side = list(trajectory_subdir.glob("*side*2D_trajectory_smoothed*.json"))

                    if files_3d:
                        self.path_3d.set(str(files_3d[0]))
                    if files_45:
                        self.path_2d_45.set(str(files_45[0]))
                    if files_side:
                        self.path_2d_side.set(str(files_side[0]))

                    messagebox.showinfo("成功", "已載入最新的軌跡檔案！")
                else:
                    messagebox.showwarning("警告", "找不到trajectory資料夾")
            else:
                messagebox.showwarning("警告", "trajectory目錄不存在")
        except Exception as e:
            messagebox.showerror("錯誤", f"載入範例失敗: {str(e)}")

    def clear_all(self):
        """清除所有輸入和結果"""
        self.path_3d.set("")
        self.path_2d_45.set("")
        self.path_2d_side.set("")
        self.summary_text.delete(1.0, tk.END)
        self.detail_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_var.set("就緒")

    def start_analysis(self):
        """開始分析"""
        if not all([self.path_3d.get(), self.path_2d_45.get(), self.path_2d_side.get()]):
            messagebox.showerror("錯誤", "請選擇所有必要的檔案！")
            return

        # 檢查檔案是否存在
        for path in [self.path_3d.get(), self.path_2d_45.get(), self.path_2d_side.get()]:
            if not Path(path).exists():
                messagebox.showerror("錯誤", f"檔案不存在: {path}")
                return

        # 在背景執行分析
        self.progress_var.set(0)
        self.status_var.set("分析中...")
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        """執行分析（在背景執行緒中）"""
        try:
            self.progress_var.set(10)
            self.status_var.set("讀取檔案中...")

            # 讀取檔案
            data_3d = json.load(open(self.path_3d.get(), "r", encoding="utf-8"))
            data_2d_45 = json.load(open(self.path_2d_45.get(), "r", encoding="utf-8"))
            data_2d_side = json.load(open(self.path_2d_side.get(), "r", encoding="utf-8"))

            self.progress_var.set(30)
            self.status_var.set("分析軌跡中...")

            # 執行分析
            results = self.analyze_trajectory(data_3d, data_2d_45, data_2d_side)

            self.progress_var.set(80)
            self.status_var.set("生成報告中...")

            # 顯示結果
            self.display_results(results)

            self.progress_var.set(100)
            self.status_var.set("分析完成！")

        except Exception as e:
            self.status_var.set("分析失敗")
            messagebox.showerror("錯誤", f"分析過程中發生錯誤:\n{str(e)}")

    def analyze_trajectory(self, data_3d, data_2d_45, data_2d_side):
        """執行軌跡分析"""
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
            proj_side, proj_45 = project_points(self.P1, X), project_points(self.P2, X)
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

        # 計算整體統計
        overall_mean_cam45 = df_cam45['mean_px'].mean()
        overall_mean_camside = df_camside['mean_px'].mean()
        overall_mean_both = (overall_mean_cam45 + overall_mean_camside) / 2

        return {
            'summary_df': summary_df,
            'df_cam45': df_cam45,
            'df_camside': df_camside,
            'overall_mean_cam45': overall_mean_cam45,
            'overall_mean_camside': overall_mean_camside,
            'overall_mean_both': overall_mean_both
        }

    def display_results(self, results):
        """顯示分析結果"""
        # 總結頁面
        summary_text = f"""
🎯 3D軌跡重建誤差分析報告

📊 整體統計
{'='*50}
相機 45° 平均誤差:     {results['overall_mean_cam45']:>6.2f} pixels
側面相機平均誤差:     {results['overall_mean_camside']:>6.2f} pixels
兩相機整體平均誤差:   {results['overall_mean_both']:>6.2f} pixels

🎯 重建品質評估
{'='*50}
"""
        if results['overall_mean_both'] < 5:
            summary_text += "✅ 重建品質: 優秀 (< 5 pixels)\n"
            summary_text += "   3D重建精度非常高，適合專業分析"
        elif results['overall_mean_both'] < 10:
            summary_text += "⚠️  重建品質: 良好 (5-10 pixels)\n"
            summary_text += "   3D重建精度良好，適合一般分析"
        elif results['overall_mean_both'] < 20:
            summary_text += "⚠️  重建品質: 可接受 (10-20 pixels)\n"
            summary_text += "   3D重建精度一般，建議優化校正參數"
        else:
            summary_text += "❌ 重建品質: 需要改進 (> 20 pixels)\n"
            summary_text += "   3D重建精度不佳，建議重新校正相機"

        summary_text += f"""
📋 分析摘要
{'='*50}
總幀數: {len(results['summary_df'])}
分析關節點: {len(results['df_cam45'])} 個
平均每幀誤差: {results['summary_df']['mean_err_cam1_px'].mean():.2f} px (側面), {results['summary_df']['mean_err_cam2_px'].mean():.2f} px (45°)
"""

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary_text)

        # 詳細頁面
        detail_text = f"""
📈 每幀誤差統計
{'='*60}
{results['summary_df'].head(10).to_string(index=False)}

📍 45°相機各關節誤差
{'='*60}
{results['df_cam45'].to_string(index=False)}

📍 側面相機各關節誤差
{'='*60}
{results['df_camside'].to_string(index=False)}
"""

        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, detail_text)


def main():
    root = tk.Tk()
    app = TrajectoryAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()