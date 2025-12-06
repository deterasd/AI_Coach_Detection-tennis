"""
3D 軌跡數據診斷工具
用於檢查 3D 軌跡數據的準確性和合理性
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class TrajectoryDiagnosticTool:
    def __init__(self, root):
        self.root = root
        self.root.title("3D 軌跡數據診斷工具")
        self.root.geometry("1400x900")
        
        # 設定樣式
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 10, 'bold'))
        
        # 數據變數
        self.path_3d = tk.StringVar()
        self.data_3d = None
        
        # 創建介面
        self.create_widgets()
    
    def create_widgets(self):
        """創建所有 UI 元件"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title = ttk.Label(main_frame, text="🔍 3D 軌跡數據診斷工具", style='Title.TLabel')
        title.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # === 檔案選擇區域 ===
        file_frame = ttk.LabelFrame(main_frame, text="📁 檔案選擇", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 3D 軌跡檔案
        ttk.Label(file_frame, text="3D 軌跡檔案:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.path_3d, width=100).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="瀏覽", command=self.browse_file).grid(row=0, column=2)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="▶️ 開始診斷", command=self.run_diagnostic,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        
        # === 結果顯示區域 ===
        result_frame = ttk.LabelFrame(main_frame, text="📊 診斷結果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 建立分頁
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 分頁1: 整體診斷
        tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(tab_summary, text="整體診斷")
        
        self.summary_text = scrolledtext.ScrolledText(tab_summary, width=100, height=20, font=('Courier', 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分頁2: 關節長度分析
        tab_length = ttk.Frame(self.notebook)
        self.notebook.add(tab_length, text="關節長度分析")
        
        self.length_text = scrolledtext.ScrolledText(tab_length, width=100, height=20, font=('Courier', 10))
        self.length_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分頁3: 運動連續性
        tab_motion = ttk.Frame(self.notebook)
        self.notebook.add(tab_motion, text="運動連續性")
        
        self.motion_text = scrolledtext.ScrolledText(tab_motion, width=100, height=20, font=('Courier', 10))
        self.motion_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分頁4: 視覺化圖表
        tab_chart = ttk.Frame(self.notebook)
        self.notebook.add(tab_chart, text="視覺化圖表")
        
        self.chart_frame = tab_chart
        
        # 配置 grid 權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
    
    def browse_file(self):
        """瀏覽並選擇檔案"""
        filename = filedialog.askopenfilename(
            title="選擇 3D 軌跡檔案",
            filetypes=[("JSON 檔案", "*.json"), ("所有檔案", "*.*")]
        )
        if filename:
            self.path_3d.set(filename)
    
    def run_diagnostic(self):
        """執行診斷"""
        if not self.path_3d.get():
            messagebox.showwarning("警告", "請選擇 3D 軌跡檔案")
            return
        
        try:
            # 讀取 JSON 檔案
            with open(self.path_3d.get(), 'r', encoding='utf-8') as f:
                self.data_3d = json.load(f)
            
            # 執行各項診斷
            summary_results = self.diagnose_overall()
            length_results = self.diagnose_joint_lengths()
            motion_results = self.diagnose_motion_continuity()
            
            # 顯示結果
            self.display_summary(summary_results)
            self.display_length_analysis(length_results)
            self.display_motion_analysis(motion_results)
            
            # 創建視覺化圖表
            self.create_charts(length_results, motion_results)
            
            # 切換到結果分頁
            self.notebook.select(0)
            
            messagebox.showinfo("完成", "診斷完成！")
            
        except FileNotFoundError:
            messagebox.showerror("錯誤", "找不到檔案")
        except json.JSONDecodeError:
            messagebox.showerror("錯誤", "JSON 檔案格式錯誤")
        except Exception as e:
            messagebox.showerror("錯誤", f"診斷失敗: {str(e)}")
    
    def get_point_3d(self, frame, keypoint):
        """從幀數據中提取 3D 座標"""
        if keypoint not in frame:
            return None
        
        pt = frame[keypoint]
        if not isinstance(pt, dict):
            return None
        
        if all(c in pt and pt[c] is not None for c in ("x", "y", "z")):
            return np.array([pt["x"], pt["y"], pt["z"]])
        
        return None
    
    def calculate_distance(self, pt1, pt2):
        """計算兩點之間的距離"""
        if pt1 is None or pt2 is None:
            return None
        return np.linalg.norm(pt1 - pt2)
    
    def diagnose_overall(self):
        """整體診斷"""
        results = {
            'total_frames': len(self.data_3d),
            'keypoints': set(),
            'valid_frames': 0,
            'keypoint_coverage': {}
        }
        
        # 統計關節點覆蓋率
        for frame in self.data_3d:
            frame_has_data = False
            for key, value in frame.items():
                if isinstance(value, dict) and 'x' in value and 'y' in value and 'z' in value:
                    results['keypoints'].add(key)
                    results['keypoint_coverage'][key] = results['keypoint_coverage'].get(key, 0) + 1
                    frame_has_data = True
            
            if frame_has_data:
                results['valid_frames'] += 1
        
        results['keypoints'] = sorted(results['keypoints'])
        
        return results
    
    def diagnose_joint_lengths(self):
        """診斷關節長度"""
        # 定義身體各部位的連接關係和預期長度範圍（單位：毫米）
        body_segments = {
            '頭部寬度': {
                'points': ('left_eye', 'right_eye'),
                'expected_range': (50, 100),
                'unit': 'mm'
            },
            '肩寬': {
                'points': ('left_shoulder', 'right_shoulder'),
                'expected_range': (300, 500),
                'unit': 'mm'
            },
            '左大腿': {
                'points': ('left_hip', 'left_knee'),
                'expected_range': (300, 600),
                'unit': 'mm'
            },
            '右大腿': {
                'points': ('right_hip', 'right_knee'),
                'expected_range': (300, 600),
                'unit': 'mm'
            },
            '左小腿': {
                'points': ('left_knee', 'left_ankle'),
                'expected_range': (300, 600),
                'unit': 'mm'
            },
            '右小腿': {
                'points': ('right_knee', 'right_ankle'),
                'expected_range': (300, 600),
                'unit': 'mm'
            },
            '左上臂': {
                'points': ('left_shoulder', 'left_elbow'),
                'expected_range': (200, 400),
                'unit': 'mm'
            },
            '右上臂': {
                'points': ('right_shoulder', 'right_elbow'),
                'expected_range': (200, 400),
                'unit': 'mm'
            },
            '左前臂': {
                'points': ('left_elbow', 'left_wrist'),
                'expected_range': (200, 400),
                'unit': 'mm'
            },
            '右前臂': {
                'points': ('right_elbow', 'right_wrist'),
                'expected_range': (200, 400),
                'unit': 'mm'
            },
            '軀幹長度': {
                'points': ('nose', 'left_hip'),  # 使用鼻子到臀部的距離
                'expected_range': (400, 800),
                'unit': 'mm'
            }
        }
        
        results = {}
        
        for segment_name, segment_info in body_segments.items():
            pt1_name, pt2_name = segment_info['points']
            distances = []
            
            for frame in self.data_3d:
                pt1 = self.get_point_3d(frame, pt1_name)
                pt2 = self.get_point_3d(frame, pt2_name)
                
                dist = self.calculate_distance(pt1, pt2)
                if dist is not None:
                    distances.append(dist)
            
            if distances:
                results[segment_name] = {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'max': np.max(distances),
                    'expected_range': segment_info['expected_range'],
                    'unit': segment_info['unit'],
                    'sample_count': len(distances)
                }
        
        return results
    
    def diagnose_motion_continuity(self):
        """診斷運動連續性"""
        keypoints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'tennis_ball']
        
        results = {}
        
        for keypoint in keypoints:
            movements = []
            
            for i in range(len(self.data_3d) - 1):
                pt1 = self.get_point_3d(self.data_3d[i], keypoint)
                pt2 = self.get_point_3d(self.data_3d[i + 1], keypoint)
                
                dist = self.calculate_distance(pt1, pt2)
                if dist is not None:
                    movements.append(dist)
            
            if movements:
                results[keypoint] = {
                    'mean_movement': np.mean(movements),
                    'std_movement': np.std(movements),
                    'max_movement': np.max(movements),
                    'median_movement': np.median(movements),
                    'sample_count': len(movements),
                    'movements': movements  # 保存所有移動數據用於繪圖
                }
        
        return results
    
    def display_summary(self, results):
        """顯示整體診斷結果"""
        self.summary_text.delete(1.0, tk.END)
        
        text = "=" * 80 + "\n"
        text += "3D 軌跡數據整體診斷報告\n"
        text += "=" * 80 + "\n\n"
        
        text += f"📊 基本資訊\n"
        text += f"{'─' * 80}\n"
        text += f"總幀數: {results['total_frames']}\n"
        text += f"有效幀數: {results['valid_frames']}\n"
        text += f"有效率: {results['valid_frames'] / results['total_frames'] * 100:.1f}%\n"
        text += f"檢測到的關節點: {len(results['keypoints'])} 個\n\n"
        
        text += f"📌 關節點列表\n"
        text += f"{'─' * 80}\n"
        for kp in results['keypoints']:
            coverage = results['keypoint_coverage'].get(kp, 0)
            coverage_pct = coverage / results['total_frames'] * 100
            text += f"  • {kp:<20} 出現於 {coverage}/{results['total_frames']} 幀 ({coverage_pct:.1f}%)\n"
        
        text += "\n" + "=" * 80 + "\n"
        
        self.summary_text.insert(1.0, text)
    
    def display_length_analysis(self, results):
        """顯示關節長度分析結果"""
        self.length_text.delete(1.0, tk.END)
        
        text = "=" * 80 + "\n"
        text += "關節長度分析報告\n"
        text += "=" * 80 + "\n\n"
        
        text += f"{'部位':<15} {'平均長度':<15} {'標準差':<12} {'範圍':<25} {'預期範圍':<20} {'判定'}\n"
        text += "─" * 80 + "\n"
        
        all_valid = True
        
        for segment_name, data in sorted(results.items()):
            mean = data['mean']
            std = data['std']
            min_val = data['min']
            max_val = data['max']
            expected_min, expected_max = data['expected_range']
            unit = data['unit']
            
            # 判定是否在合理範圍內
            if expected_min <= mean <= expected_max:
                status = "✅ 正常"
            elif mean < expected_min * 0.5 or mean > expected_max * 2:
                status = "❌ 異常"
                all_valid = False
            else:
                status = "⚠️  警告"
                all_valid = False
            
            text += f"{segment_name:<15} "
            text += f"{mean:>7.1f} {unit:<6} "
            text += f"{std:>6.1f} {unit:<4} "
            text += f"{min_val:>6.1f} - {max_val:<6.1f} {unit:<3} "
            text += f"({expected_min}-{expected_max} {unit}){'':<5} "
            text += f"{status}\n"
        
        text += "─" * 80 + "\n\n"
        
        if all_valid:
            text += "✅ 所有關節長度都在合理範圍內，3D 數據可信度高！\n"
        else:
            text += "⚠️  部分關節長度超出預期範圍，建議檢查：\n"
            text += "   1. 相機標定是否準確\n"
            text += "   2. 3D 重建演算法是否正確\n"
            text += "   3. 座標單位是否正確（應為毫米）\n"
        
        text += "\n" + "=" * 80 + "\n"
        
        self.length_text.insert(1.0, text)
    
    def display_motion_analysis(self, results):
        """顯示運動連續性分析結果"""
        self.motion_text.delete(1.0, tk.END)
        
        text = "=" * 80 + "\n"
        text += "運動連續性分析報告\n"
        text += "=" * 80 + "\n\n"
        
        text += f"{'關節點':<20} {'平均移動':<15} {'標準差':<12} {'最大移動':<15} {'判定'}\n"
        text += "─" * 80 + "\n"
        
        all_smooth = True
        
        for keypoint, data in sorted(results.items()):
            mean_mov = data['mean_movement']
            std_mov = data['std_movement']
            max_mov = data['max_movement']
            
            # 判定運動是否平滑（假設 30 FPS，每幀移動不應超過 100mm）
            # tennis_ball 允許更大的移動
            threshold = 200 if keypoint == 'tennis_ball' else 100
            
            if max_mov < threshold:
                status = "✅ 平滑"
            elif max_mov < threshold * 2:
                status = "⚠️  輕微跳動"
                all_smooth = False
            else:
                status = "❌ 劇烈跳動"
                all_smooth = False
            
            text += f"{keypoint:<20} "
            text += f"{mean_mov:>7.2f} mm{'':<6} "
            text += f"{std_mov:>6.2f} mm{'':<4} "
            text += f"{max_mov:>7.2f} mm{'':<6} "
            text += f"{status}\n"
        
        text += "─" * 80 + "\n\n"
        
        if all_smooth:
            text += "✅ 所有關節點運動都很平滑，追蹤質量良好！\n"
        else:
            text += "⚠️  部分關節點有跳動現象，建議：\n"
            text += "   1. 檢查姿態估計的準確性\n"
            text += "   2. 增加平滑處理（例如卡爾曼濾波）\n"
            text += "   3. 檢查是否有遮擋導致的誤檢測\n"
        
        text += "\n📌 參考標準:\n"
        text += "   • 身體關節點每幀移動 < 100mm: 平滑\n"
        text += "   • 網球每幀移動 < 200mm: 平滑（球速較快）\n"
        text += "   • 假設影片幀率: 30 FPS\n"
        
        text += "\n" + "=" * 80 + "\n"
        
        self.motion_text.insert(1.0, text)
    
    def create_charts(self, length_results, motion_results):
        """創建視覺化圖表"""
        # 清除舊圖表
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # 創建圖表
        fig = Figure(figsize=(14, 10))
        
        # 圖1: 關節長度分布
        ax1 = fig.add_subplot(2, 2, 1)
        segments = list(length_results.keys())
        means = [length_results[s]['mean'] for s in segments]
        stds = [length_results[s]['std'] for s in segments]
        
        y_pos = np.arange(len(segments))
        ax1.barh(y_pos, means, xerr=stds, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(segments)
        ax1.set_xlabel('長度 (mm)')
        ax1.set_title('關節長度分布')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 圖2: 運動幅度比較
        ax2 = fig.add_subplot(2, 2, 2)
        keypoints = list(motion_results.keys())
        mean_movements = [motion_results[k]['mean_movement'] for k in keypoints]
        
        colors = ['red' if k == 'tennis_ball' else 'lightcoral' for k in keypoints]
        ax2.barh(keypoints, mean_movements, alpha=0.7, color=colors, edgecolor='black')
        ax2.set_xlabel('平均移動距離 (mm/frame)')
        ax2.set_title('各關節點運動幅度')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 圖3: 手腕運動軌跡（前 100 幀）
        ax3 = fig.add_subplot(2, 2, 3)
        if 'right_wrist' in motion_results:
            movements = motion_results['right_wrist']['movements'][:100]
            ax3.plot(movements, label='右手腕', alpha=0.7, linewidth=2)
        if 'left_wrist' in motion_results:
            movements = motion_results['left_wrist']['movements'][:100]
            ax3.plot(movements, label='左手腕', alpha=0.7, linewidth=2)
        
        ax3.set_xlabel('幀數')
        ax3.set_ylabel('移動距離 (mm)')
        ax3.set_title('手腕運動軌跡（前 100 幀）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 圖4: 網球運動軌跡
        ax4 = fig.add_subplot(2, 2, 4)
        if 'tennis_ball' in motion_results:
            movements = motion_results['tennis_ball']['movements'][:200]
            ax4.plot(movements, color='orange', alpha=0.7, linewidth=2)
            ax4.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='警戒線 (200mm)')
            ax4.set_xlabel('幀數')
            ax4.set_ylabel('移動距離 (mm)')
            ax4.set_title('網球運動軌跡（前 200 幀）')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # 嵌入圖表到 Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = TrajectoryDiagnosticTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
