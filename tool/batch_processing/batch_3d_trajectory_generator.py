"""
批量 3D 軌跡產生器
自動配對側面和 45 度影片，批量產生 3D_trajectory_smoothed.json
支援多位打者，自動根據編號分組
"""

import json
import numpy as np
import os
import sys
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from ultralytics import YOLO
import shutil
from datetime import datetime

# Add parent directory to sys.path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 匯入處理模組
from trajector_processing_unified import processing_trajectory_unified

class Batch3DTrajectoryGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("批量 3D 軌跡產生器")
        self.root.geometry("1200x800")
        
        # 設定樣式
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 11, 'bold'))
        
        # 數據變數
        self.input_side_folder = tk.StringVar()
        self.input_45_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        
        # 投影矩陣文字框
        self.P1_text = None
        self.P2_text = None
        
        # 影片配對結果
        self.video_pairs = []
        
        # 身高對照表
        self.height_data = {
            'B1': 177, 'B2': 177, 'B4': 171, 'B5': 170, 'B6': 173,
            'B7': 179, 'B8': 175, 'B9': 175, 'B10': 173, 'B11': 177,
            'B12': 171, 'B13': 145, 'B14': 163, 'B15': 162, 'B16': 150,
            'B17': 163, 'B18': 158, 'B19': 160, 'B20': 164, 'B21': 166,
            'B22': 160, 'B23': 158
        }
        
        # AI 模型
        self.yolo_pose_model = None
        self.yolo_tennis_ball_model = None
        
        # 創建介面
        self.create_widgets()
        
        # 載入預設投影矩陣
        self.load_default_matrices()
    
    def create_widgets(self):
        """創建所有 UI 元件"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title = ttk.Label(main_frame, text="🎾 批量 3D 軌跡產生器", style='Title.TLabel')
        title.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # === 資料夾選擇區域 ===
        folder_frame = ttk.LabelFrame(main_frame, text="📁 資料夾設定", padding="15")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 側面影片資料夾
        ttk.Label(folder_frame, text="側面影片資料夾:", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.input_side_folder, width=70).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="瀏覽", command=lambda: self.browse_folder(self.input_side_folder, "側面影片")).grid(row=0, column=2, pady=5)
        
        # 45度影片資料夾
        ttk.Label(folder_frame, text="45度影片資料夾:", font=('Arial', 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.input_45_folder, width=70).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="瀏覽", command=lambda: self.browse_folder(self.input_45_folder, "45度影片")).grid(row=1, column=2, pady=5)
        
        # 輸出資料夾
        ttk.Label(folder_frame, text="輸出資料夾:", font=('Arial', 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=70).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(folder_frame, text="瀏覽", command=lambda: self.browse_folder(self.output_folder, "輸出")).grid(row=2, column=2, pady=5)
        
        # 掃描按鈕
        ttk.Button(folder_frame, text="🔍 掃描並配對影片", command=self.scan_and_pair_videos).grid(row=3, column=0, columnspan=3, pady=10)
        
        # === 投影矩陣設定區域 ===
        matrix_frame = ttk.LabelFrame(main_frame, text="🔢 投影矩陣設定", padding="15")
        matrix_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # P1 (側面相機)
        p1_frame = ttk.Frame(matrix_frame)
        p1_frame.grid(row=0, column=0, padx=15, sticky=(tk.N, tk.W))
        
        ttk.Label(p1_frame, text="P1 (側面相機) 3×4:", style='Section.TLabel').pack(anchor=tk.W)
        self.P1_text = scrolledtext.ScrolledText(p1_frame, width=45, height=4, font=('Courier', 9))
        self.P1_text.pack(pady=5)
        
        # P2 (45度相機)
        p2_frame = ttk.Frame(matrix_frame)
        p2_frame.grid(row=0, column=1, padx=15, sticky=(tk.N, tk.W))
        
        ttk.Label(p2_frame, text="P2 (45度相機) 3×4:", style='Section.TLabel').pack(anchor=tk.W)
        self.P2_text = scrolledtext.ScrolledText(p2_frame, width=45, height=4, font=('Courier', 9))
        self.P2_text.pack(pady=5)
        
        # === 操作按鈕 ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=15)
        
        ttk.Button(button_frame, text="🔄 載入預設矩陣", command=self.load_default_matrices).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="▶️ 開始批量處理", command=self.start_batch_processing, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        
        # === 結果顯示區域 ===
        result_frame = ttk.LabelFrame(main_frame, text="📊 處理進度與結果", padding="10")
        result_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 建立分頁
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 分頁1: 影片配對
        tab_pairs = ttk.Frame(self.notebook)
        self.notebook.add(tab_pairs, text="影片配對")
        
        self.pairs_text = scrolledtext.ScrolledText(tab_pairs, width=100, height=15, font=('Courier', 9))
        self.pairs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分頁2: 處理日誌
        tab_log = ttk.Frame(self.notebook)
        self.notebook.add(tab_log, text="處理日誌")
        
        self.log_text = scrolledtext.ScrolledText(tab_log, width=100, height=15, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 配置 grid 權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def browse_folder(self, var, folder_type):
        """瀏覽並選擇資料夾"""
        folder = filedialog.askdirectory(title=f"選擇{folder_type}資料夾")
        if folder:
            var.set(folder)
    
    def load_default_matrices(self):
        """載入預設投影矩陣"""
        # 預設 P1 (側面相機)
        P1_default = """[4930.662905, 0.000000, 1779.941295, 0.000000],
[0.000000, 3868.767102, 1001.404479, 0.000000],
[0.000000, 0.000000, 1.000000, 0.000000]"""
        
        # 預設 P2 (45度相機)
        P2_default = """[-2198.710896, -712.692728, 5142.070321, -27039471.260434],
[-1576.351163, 4860.096440, 1341.670710, -3091066.024238],
[-0.978171, 0.138807, 0.154642, 6045.131181]"""
        
        self.P1_text.delete(1.0, tk.END)
        self.P1_text.insert(1.0, P1_default)
        
        self.P2_text.delete(1.0, tk.END)
        self.P2_text.insert(1.0, P2_default)
        
        self.log("✅ 已載入預設投影矩陣")
    
    def parse_matrix(self, text):
        """解析文字格式的矩陣"""
        text = text.strip()
        
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1].strip()
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        matrix = []
        
        for line in lines:
            line = line.strip().rstrip(',').strip()
            if line.startswith('['):
                line = line[1:]
            if line.endswith(']'):
                line = line[:-1]
            
            values = []
            for item in line.replace(',', ' ').split():
                try:
                    values.append(float(item))
                except ValueError:
                    continue
            
            if values:
                matrix.append(values)
        
        return np.array(matrix, dtype=float)
    
    def scan_and_pair_videos(self):
        """掃描並配對影片"""
        if not self.input_side_folder.get() or not self.input_45_folder.get():
            messagebox.showwarning("警告", "請先選擇側面和45度影片資料夾")
            return
        
        side_folder = Path(self.input_side_folder.get())
        deg45_folder = Path(self.input_45_folder.get())
        
        if not side_folder.exists():
            messagebox.showerror("錯誤", f"側面影片資料夾不存在: {side_folder}")
            return
        
        if not deg45_folder.exists():
            messagebox.showerror("錯誤", f"45度影片資料夾不存在: {deg45_folder}")
            return
        
        # 掃描側面影片
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        side_videos = {}
        
        for ext in video_extensions:
            for video in side_folder.glob(f"*{ext}"):
                # 提取影片名稱中的編號 (例如: 2-1, 2-2, 3-1)
                match = re.search(r'(\d+)-(\d+)', video.stem)
                if match:
                    player_id = match.group(1)
                    ball_id = match.group(2)
                    key = f"{player_id}-{ball_id}"
                    side_videos[key] = video
        
        # 掃描45度影片
        deg45_videos = {}
        
        for ext in video_extensions:
            for video in deg45_folder.glob(f"*{ext}"):
                # 提取影片名稱中的編號 (例如: 2-1_45, 2-2_45, 3-1_45)
                match = re.search(r'(\d+)-(\d+)', video.stem)
                if match:
                    player_id = match.group(1)
                    ball_id = match.group(2)
                    key = f"{player_id}-{ball_id}"
                    deg45_videos[key] = video
        
        # 配對影片
        self.video_pairs = []
        
        for key in sorted(side_videos.keys()):
            if key in deg45_videos:
                player_id = key.split('-')[0]
                self.video_pairs.append({
                    'key': key,
                    'player_id': player_id,
                    'side_video': side_videos[key],
                    'deg45_video': deg45_videos[key],
                    'height': self.height_data.get(f'B{player_id}', 170)  # 預設170cm
                })
        
        # 顯示配對結果
        self.display_pairs()
        
        if not self.video_pairs:
            messagebox.showwarning("警告", "沒有找到可配對的影片")
        else:
            messagebox.showinfo("成功", f"找到 {len(self.video_pairs)} 組配對影片")
    
    def display_pairs(self):
        """顯示影片配對結果"""
        self.pairs_text.delete(1.0, tk.END)
        
        text = "=" * 100 + "\n"
        text += "影片配對結果\n"
        text += "=" * 100 + "\n\n"
        
        if not self.video_pairs:
            text += "尚未掃描或沒有找到配對影片\n"
        else:
            # 按打者分組
            players = {}
            for pair in self.video_pairs:
                player_id = pair['player_id']
                if player_id not in players:
                    players[player_id] = []
                players[player_id].append(pair)
            
            text += f"總共找到 {len(self.video_pairs)} 組影片，涵蓋 {len(players)} 位打者\n\n"
            
            for player_id in sorted(players.keys(), key=int):
                pairs = players[player_id]
                height = pairs[0]['height']
                text += f"👤 打者 #{player_id} (身高: {height} cm) - {len(pairs)} 組影片\n"
                text += "-" * 100 + "\n"
                
                for i, pair in enumerate(pairs, 1):
                    text += f"   {i}. {pair['key']}\n"
                    text += f"      側面: {pair['side_video'].name}\n"
                    text += f"      45度: {pair['deg45_video'].name}\n"
                    text += f"      輸出: {pair['key']}(3D_trajectory_smoothed).json\n\n"
                
                text += "\n"
        
        text += "=" * 100 + "\n"
        
        self.pairs_text.insert(1.0, text)
    
    def log(self, message):
        """添加日誌訊息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_batch_processing(self):
        """開始批量處理"""
        if not self.video_pairs:
            messagebox.showwarning("警告", "請先掃描並配對影片")
            return
        
        if not self.output_folder.get():
            messagebox.showwarning("警告", "請選擇輸出資料夾")
            return
        
        # 解析投影矩陣
        try:
            P1 = self.parse_matrix(self.P1_text.get(1.0, tk.END))
            P2 = self.parse_matrix(self.P2_text.get(1.0, tk.END))
            
            if P1.shape != (3, 4) or P2.shape != (3, 4):
                messagebox.showerror("錯誤", "投影矩陣必須是 3×4 的矩陣")
                return
        except Exception as e:
            messagebox.showerror("錯誤", f"投影矩陣解析失敗: {str(e)}")
            return
        
        # 載入 AI 模型
        if self.yolo_pose_model is None or self.yolo_tennis_ball_model is None:
            self.log("🤖 載入 AI 模型...")
            try:
                self.yolo_pose_model = YOLO('model/yolov8n-pose.pt')
                self.yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
                self.log("✅ AI 模型載入成功")
            except Exception as e:
                messagebox.showerror("錯誤", f"模型載入失敗: {str(e)}")
                return
        
        # 創建輸出資料夾
        output_base = Path(self.output_folder.get())
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 切換到日誌分頁
        self.notebook.select(1)
        
        self.log("=" * 100)
        self.log("🚀 開始批量處理")
        self.log(f"📁 輸出資料夾: {output_base}")
        self.log(f"📊 總共 {len(self.video_pairs)} 組影片")
        self.log("=" * 100)
        
        # 按打者分組處理
        players = {}
        for pair in self.video_pairs:
            player_id = pair['player_id']
            if player_id not in players:
                players[player_id] = []
            players[player_id].append(pair)
        
        success_count = 0
        fail_count = 0
        
        for player_id in sorted(players.keys(), key=int):
            pairs = players[player_id]
            height = pairs[0]['height']
            
            # 創建打者專屬資料夾
            player_folder = output_base / player_id
            player_folder.mkdir(parents=True, exist_ok=True)
            
            self.log(f"\n👤 處理打者 #{player_id} (身高: {height} cm) - {len(pairs)} 組影片")
            self.log(f"📁 輸出資料夾: {player_folder.name}/")
            
            for i, pair in enumerate(pairs, 1):
                key = pair['key']
                side_video = str(pair['side_video'])
                deg45_video = str(pair['deg45_video'])
                
                self.log(f"\n   🎾 [{i}/{len(pairs)}] 處理 {key}...")
                self.log(f"      側面影片: {pair['side_video'].name}")
                self.log(f"      45度影片: {pair['deg45_video'].name}")
                
                try:
                    # 使用統一處理函數
                    success = processing_trajectory_unified(
                        P1=P1,
                        P2=P2,
                        yolo_pose_model=self.yolo_pose_model,
                        yolo_tennis_ball_model=self.yolo_tennis_ball_model,
                        video_side=side_video,
                        video_45=deg45_video,
                        knn_dataset='knn_dataset_new.json',
                        name=key,
                        ball_entry_direction="right",
                        confidence_threshold=0.5,
                        output_folder=str(player_folder),
                        segment_videos=False  # 不進行影片分割，直接處理
                    )
                    
                    if success:
                        # 列出資料夾中所有檔案，用於除錯
                        all_files = list(player_folder.glob("*"))
                        self.log(f"      📂 資料夾中的檔案 ({len(all_files)} 個):")
                        for f in all_files:
                            self.log(f"         - {f.name}")
                        
                        # 嘗試多種可能的檔案名稱格式
                        possible_files = [
                            player_folder / f"{key}_segment(3D_trajectory_smoothed).json",
                            player_folder / f"{key}(3D_trajectory_smoothed).json",
                            player_folder / f"{key}_3D_trajectory_smoothed.json",
                            player_folder / f"{key}__1_segment(3D_trajectory_smoothed).json",
                        ]
                        
                        target_file = None
                        for pf in possible_files:
                            if pf.exists():
                                target_file = pf
                                break
                        
                        if target_file:
                            # 重新命名為目標格式
                            final_file = player_folder / f"{key}(3D_trajectory_smoothed).json"
                            if target_file != final_file:
                                shutil.move(str(target_file), str(final_file))
                            self.log(f"      ✅ 成功: {final_file.name}")
                            success_count += 1
                        else:
                            # 檢查是否有包含 3D_trajectory_smoothed 的檔案
                            smoothed_files = list(player_folder.glob("*3D_trajectory_smoothed*.json"))
                            if smoothed_files:
                                self.log(f"      ⚠️  找到平滑軌跡檔案但檔名不符:")
                                for sf in smoothed_files:
                                    self.log(f"         - {sf.name}")
                                # 使用找到的第一個檔案
                                final_file = player_folder / f"{key}(3D_trajectory_smoothed).json"
                                shutil.move(str(smoothed_files[0]), str(final_file))
                                self.log(f"      ✅ 已重新命名為: {final_file.name}")
                                success_count += 1
                            else:
                                self.log(f"      ❌ 找不到輸出檔案")
                                self.log(f"         預期檔名: {key}_segment(3D_trajectory_smoothed).json")
                                fail_count += 1
                    else:
                        self.log(f"      ❌ 失敗: 處理過程發生錯誤")
                        fail_count += 1
                
                except Exception as e:
                    self.log(f"      ❌ 錯誤: {str(e)}")
                    fail_count += 1
        
        # 完成訊息
        self.log("\n" + "=" * 100)
        self.log("🎉 批量處理完成！")
        self.log(f"✅ 成功: {success_count} 個")
        self.log(f"❌ 失敗: {fail_count} 個")
        self.log(f"📁 結果保存在: {output_base}")
        self.log("=" * 100)
        
        messagebox.showinfo("完成", 
                          f"批量處理完成！\n\n成功: {success_count} 個\n失敗: {fail_count} 個\n\n結果保存在:\n{output_base}")


def main():
    root = tk.Tk()
    app = Batch3DTrajectoryGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
