#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFmpeg 圖形化影片分割工具
功能:
- 視覺化時間軸
- 多區間選擇
- 批次分割輸出
- GPU 硬體加速

作者: AI Coach Detection Team
日期: 2025-01-25
"""

import os
import sys
import cv2
import subprocess
import shutil
import json
from pathlib import Path
from datetime import timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading


class VideoSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 影片分割工具 (FFmpeg GPU加速)")
        self.root.geometry("1200x800")
        
        # 變數初始化
        self.video_path = None
        self.video_duration = 0
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.segments = []  # [(start_sec, end_sec, name), ...]
        self.preview_cap = None
        self.current_preview_time = 0
        
        # 檢查 FFmpeg
        self.has_ffmpeg = shutil.which('ffmpeg') is not None
        
        # 建立介面
        self.setup_ui()
        
    def setup_ui(self):
        """建立使用者介面"""
        # ============ 頂部控制區 ============
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="影片檔案:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.video_label = ttk.Label(top_frame, text="尚未載入影片", foreground="gray")
        self.video_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(top_frame, text="📂 載入影片", command=self.load_video, width=15).pack(side=tk.LEFT, padx=5)
        
        # FFmpeg 狀態顯示
        ffmpeg_status = "✅ FFmpeg可用" if self.has_ffmpeg else "⚠️ FFmpeg未安裝"
        ffmpeg_color = "green" if self.has_ffmpeg else "red"
        ttk.Label(top_frame, text=ffmpeg_status, foreground=ffmpeg_color).pack(side=tk.LEFT, padx=10)
        
        # ============ 影片資訊區 ============
        info_frame = ttk.LabelFrame(self.root, text="影片資訊", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=3, state='disabled', bg="#f0f0f0")
        self.info_text.pack(fill=tk.X)
        
        # ============ 時間軸預覽區 ============
        timeline_frame = ttk.LabelFrame(self.root, text="⏱️ 時間軸預覽", padding="10")
        timeline_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 當前時間顯示
        time_control_frame = ttk.Frame(timeline_frame)
        time_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(time_control_frame, text="當前時間:").pack(side=tk.LEFT, padx=5)
        self.time_label = ttk.Label(time_control_frame, text="00:00:00", font=("Courier", 12, "bold"))
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # 時間軸滑桿
        self.timeline_slider = ttk.Scale(timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                        command=self.on_timeline_change)
        self.timeline_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # ============ 區間設定區 ============
        segment_frame = ttk.LabelFrame(self.root, text="✂️ 分割區間設定", padding="10")
        segment_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 新增區間控制
        add_frame = ttk.Frame(segment_frame)
        add_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(add_frame, text="開始時間 (秒，支援小數):").pack(side=tk.LEFT, padx=5)
        self.start_entry = ttk.Entry(add_frame, width=12)
        self.start_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(add_frame, text="結束時間 (秒，支援小數):").pack(side=tk.LEFT, padx=5)
        self.end_entry = ttk.Entry(add_frame, width=12)
        self.end_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(add_frame, text="片段名稱:").pack(side=tk.LEFT, padx=5)
        self.name_entry = ttk.Entry(add_frame, width=20)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(add_frame, text="➕ 新增區間", command=self.add_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(add_frame, text="🎯 使用當前時間", command=self.use_current_time).pack(side=tk.LEFT, padx=5)
        
        # 區間列表
        list_frame = ttk.Frame(segment_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 建立表格
        columns = ("編號", "開始時間", "結束時間", "長度", "片段名稱")
        self.segment_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.segment_tree.heading(col, text=col)
            if col == "編號":
                self.segment_tree.column(col, width=50, anchor=tk.CENTER)
            elif col in ["開始時間", "結束時間", "長度"]:
                self.segment_tree.column(col, width=140, anchor=tk.CENTER)
            else:
                self.segment_tree.column(col, width=200, anchor=tk.W)
        
        # 滾動條
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.segment_tree.yview)
        self.segment_tree.configure(yscroll=scrollbar.set)
        
        self.segment_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 區間操作按鈕
        btn_frame = ttk.Frame(segment_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="🗑️ 刪除選中", command=self.delete_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🧹 清空全部", command=self.clear_segments).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 儲存設定", command=self.save_segments).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📂 載入設定", command=self.load_segments).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⏱️ 匯入時間碼", command=self.import_timecodes).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📋 匯出時間碼", command=self.export_timecodes).pack(side=tk.LEFT, padx=5)
        
        # ============ 輸出設定區 ============
        output_frame = ttk.LabelFrame(self.root, text="🎯 輸出設定", padding="10")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        output_control = ttk.Frame(output_frame)
        output_control.pack(fill=tk.X)
        
        ttk.Label(output_control, text="輸出資料夾:").pack(side=tk.LEFT, padx=5)
        self.output_label = ttk.Label(output_control, text="未設定", foreground="gray")
        self.output_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(output_control, text="📁 選擇資料夾", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # 編碼選項
        encode_frame = ttk.Frame(output_frame)
        encode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(encode_frame, text="編碼器:").pack(side=tk.LEFT, padx=5)
        self.encoder_var = tk.StringVar(value="h264_nvenc")
        ttk.Radiobutton(encode_frame, text="GPU加速 (h264_nvenc)", variable=self.encoder_var, 
                       value="h264_nvenc").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(encode_frame, text="CPU (libx264)", variable=self.encoder_var, 
                       value="libx264").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(encode_frame, text="複製串流 (最快)", variable=self.encoder_var, 
                       value="copy").pack(side=tk.LEFT, padx=5)
        
        # ============ 執行區 ============
        action_frame = ttk.Frame(self.root, padding="10")
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_label = ttk.Label(action_frame, text="準備就緒", foreground="blue")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(action_frame, mode='determinate', length=300)
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        ttk.Button(action_frame, text="▶️ 開始分割", command=self.start_splitting, 
                  style="Accent.TButton", width=20).pack(side=tk.RIGHT, padx=5)
        
    def load_video(self):
        """載入影片"""
        file_path = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[
                ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
                ("所有檔案", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 使用 OpenCV 讀取影片資訊
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("無法開啟影片檔案")
            
            self.video_path = file_path
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            cap.release()
            
            # 更新介面
            self.video_label.config(text=Path(file_path).name, foreground="black")
            self.timeline_slider.config(to=self.video_duration)
            
            # 更新影片資訊
            self.update_video_info()
            
            messagebox.showinfo("成功", f"影片載入成功!\n長度: {self.format_time(self.video_duration)}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"載入影片失敗:\n{str(e)}")
    
    def update_video_info(self):
        """更新影片資訊顯示"""
        info = f"解析度: {self.width}x{self.height}  |  "
        info += f"幀率: {self.fps:.2f} FPS  |  "
        info += f"總幀數: {self.total_frames}  |  "
        info += f"長度: {self.format_time(self.video_duration)}\n"
        info += f"💡 提示: 時間精度支援到毫秒級（例如: 1.234 秒）"
        
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state='disabled')
    
    def on_timeline_change(self, value):
        """時間軸滑桿變化"""
        self.current_preview_time = float(value)
        self.time_label.config(text=self.format_time(self.current_preview_time))
    
    def format_time(self, seconds):
        """格式化時間顯示（支援毫秒）"""
        total_seconds = float(seconds)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def timecode_to_seconds(self, timecode):
        """將時間碼 (HH:MM:SS:FF) 轉換為秒數"""
        try:
            parts = timecode.strip().split(':')
            if len(parts) == 4:
                # HH:MM:SS:FF 格式
                hours, minutes, seconds, frames = map(int, parts)
                total_seconds = hours * 3600 + minutes * 60 + seconds
                # 將幀數轉換為秒數
                if self.fps > 0:
                    total_seconds += frames / self.fps
                return total_seconds
            elif len(parts) == 3:
                # HH:MM:SS 格式
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError("不支援的時間格式")
        except Exception as e:
            raise ValueError(f"時間碼格式錯誤: {timecode} ({e})")
    
    def seconds_to_timecode(self, seconds):
        """將秒數轉換為時間碼 (HH:MM:SS:FF)"""
        total_seconds = float(seconds)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        # 計算幀數
        frames = int((total_seconds % 1) * self.fps) if self.fps > 0 else 0
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    
    def use_current_time(self):
        """使用當前時間軸時間（精確到毫秒）"""
        if not self.video_path:
            messagebox.showwarning("警告", "請先載入影片!")
            return
        
        # 如果開始時間為空,填入當前時間
        if not self.start_entry.get():
            self.start_entry.delete(0, tk.END)
            self.start_entry.insert(0, f"{self.current_preview_time:.3f}")
        # 否則填入結束時間
        else:
            self.end_entry.delete(0, tk.END)
            self.end_entry.insert(0, f"{self.current_preview_time:.3f}")
    
    def add_segment(self):
        """新增分割區間"""
        if not self.video_path:
            messagebox.showwarning("警告", "請先載入影片!")
            return
        
        try:
            start = float(self.start_entry.get())
            end = float(self.end_entry.get())
            name = self.name_entry.get().strip() or f"segment_{len(self.segments)+1:03d}"
            
            # 驗證
            if start < 0 or end > self.video_duration:
                raise ValueError(f"時間範圍必須在 0 ~ {self.video_duration:.2f} 秒之間")
            
            if start >= end:
                raise ValueError("開始時間必須小於結束時間")
            
            # 新增到列表
            self.segments.append((start, end, name))
            
            # 更新表格
            duration = end - start
            self.segment_tree.insert('', tk.END, values=(
                len(self.segments),
                self.format_time(start),
                self.format_time(end),
                self.format_time(duration),
                name
            ))
            
            # 清空輸入框
            self.start_entry.delete(0, tk.END)
            self.end_entry.delete(0, tk.END)
            self.name_entry.delete(0, tk.END)
            
        except ValueError as e:
            messagebox.showerror("錯誤", str(e))
    
    def delete_segment(self):
        """刪除選中的區間"""
        selected = self.segment_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "請先選擇要刪除的區間!")
            return
        
        # 取得選中項目的索引
        indices = [self.segment_tree.index(item) for item in selected]
        
        # 從後往前刪除
        for idx in sorted(indices, reverse=True):
            del self.segments[idx]
            self.segment_tree.delete(selected[indices.index(idx)])
        
        # 重新編號
        self.refresh_segment_list()
    
    def clear_segments(self):
        """清空所有區間"""
        if messagebox.askyesno("確認", "確定要清空所有區間嗎?"):
            self.segments.clear()
            for item in self.segment_tree.get_children():
                self.segment_tree.delete(item)
    
    def refresh_segment_list(self):
        """重新整理區間列表"""
        # 清空表格
        for item in self.segment_tree.get_children():
            self.segment_tree.delete(item)
        
        # 重新插入
        for idx, (start, end, name) in enumerate(self.segments, 1):
            duration = end - start
            self.segment_tree.insert('', tk.END, values=(
                idx,
                self.format_time(start),
                self.format_time(end),
                self.format_time(duration),
                name
            ))
    
    def save_segments(self):
        """儲存區間設定到檔案"""
        if not self.segments:
            messagebox.showwarning("警告", "沒有區間可儲存!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="儲存區間設定",
            defaultextension=".json",
            filetypes=[("JSON檔案", "*.json"), ("所有檔案", "*.*")]
        )
        
        if file_path:
            data = {
                'video_path': self.video_path,
                'segments': [{'start': s, 'end': e, 'name': n} for s, e, n in self.segments]
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", "區間設定已儲存!")
    
    def load_segments(self):
        """從檔案載入區間設定"""
        file_path = filedialog.askopenfilename(
            title="載入區間設定",
            filetypes=[("JSON檔案", "*.json"), ("所有檔案", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.segments = [(s['start'], s['end'], s['name']) for s in data['segments']]
                self.refresh_segment_list()
                messagebox.showinfo("成功", f"已載入 {len(self.segments)} 個區間!")
            except Exception as e:
                messagebox.showerror("錯誤", f"載入失敗:\n{str(e)}")
    
    def import_timecodes(self):
        """匯入時間碼格式的區間設定"""
        if not self.video_path:
            messagebox.showwarning("警告", "請先載入影片!")
            return
        
        # 創建輸入對話框
        dialog = tk.Toplevel(self.root)
        dialog.title("📋 匯入時間碼")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 說明文字
        info_frame = ttk.Frame(dialog, padding="10")
        info_frame.pack(fill=tk.X)
        
        info_text = """請貼上時間碼列表，格式範例：
("00:00:16:36", "00:00:17:29"), ("00:00:20:12", "00:00:20:59")
或
00:00:16:36 - 00:00:17:29
00:00:20:12 - 00:00:20:59

支援格式：HH:MM:SS:FF 或 HH:MM:SS"""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # 文字輸入框
        text_frame = ttk.Frame(dialog, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_input = tk.Text(text_frame, height=15, width=70)
        text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_input.yview)
        text_input.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按鈕
        btn_frame = ttk.Frame(dialog, padding="10")
        btn_frame.pack(fill=tk.X)
        
        def parse_and_import():
            try:
                content = text_input.get(1.0, tk.END).strip()
                if not content:
                    messagebox.showwarning("警告", "請輸入時間碼!")
                    return
                
                imported_count = 0
                
                # 解析多種格式
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 嘗試解析不同格式
                    # 格式1: ("00:00:16:36", "00:00:17:29")
                    if '("' in line or "('" in line:
                        import re
                        matches = re.findall(r'["\']([0-9:]+)["\']', line)
                        if len(matches) >= 2:
                            start_tc = matches[0]
                            end_tc = matches[1]
                    # 格式2: 00:00:16:36 - 00:00:17:29
                    elif ' - ' in line or '-' in line:
                        parts = line.replace(' ', '').split('-')
                        if len(parts) >= 2:
                            start_tc = parts[0]
                            end_tc = parts[1]
                    else:
                        continue
                    
                    # 轉換為秒數
                    start_sec = self.timecode_to_seconds(start_tc)
                    end_sec = self.timecode_to_seconds(end_tc)
                    
                    # 生成名稱
                    name = f"segment_{len(self.segments) + imported_count + 1:03d}"
                    
                    # 加入列表
                    self.segments.append((start_sec, end_sec, name))
                    imported_count += 1
                
                if imported_count > 0:
                    self.refresh_segment_list()
                    messagebox.showinfo("成功", f"已匯入 {imported_count} 個時間區間!")
                    dialog.destroy()
                else:
                    messagebox.showwarning("警告", "未找到有效的時間碼格式!")
                    
            except Exception as e:
                messagebox.showerror("錯誤", f"匯入失敗:\n{str(e)}")
        
        ttk.Button(btn_frame, text="✅ 匯入", command=parse_and_import).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="❌ 取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 範例按鈕
        def insert_example():
            example = '''time_intervals = [
    ("00:00:16:36", "00:00:17:29"),
    ("00:00:20:12", "00:00:20:59"),
    ("00:00:24:06", "00:00:24:48")
]'''
            text_input.delete(1.0, tk.END)
            text_input.insert(1.0, example)
        
        ttk.Button(btn_frame, text="📝 範例", command=insert_example).pack(side=tk.LEFT, padx=5)
    
    def export_timecodes(self):
        """匯出時間碼格式"""
        if not self.segments:
            messagebox.showwarning("警告", "沒有區間可匯出!")
            return
        
        if not self.video_path:
            messagebox.showwarning("警告", "需要載入影片以取得FPS資訊!")
            return
        
        # 創建顯示對話框
        dialog = tk.Toplevel(self.root)
        dialog.title("📋 匯出時間碼")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 文字顯示框
        text_frame = ttk.Frame(dialog, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_output = tk.Text(text_frame, height=20, width=70)
        text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_output.yview)
        text_output.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 生成時間碼列表
        output = "time_intervals = [\n"
        for start, end, name in self.segments:
            start_tc = self.seconds_to_timecode(start)
            end_tc = self.seconds_to_timecode(end)
            output += f'    ("{start_tc}", "{end_tc}"),  # {name}\n'
        output += "]\n"
        
        text_output.insert(1.0, output)
        text_output.config(state='disabled')
        
        # 按鈕
        btn_frame = ttk.Frame(dialog, padding="10")
        btn_frame.pack(fill=tk.X)
        
        def copy_to_clipboard():
            self.root.clipboard_clear()
            self.root.clipboard_append(output)
            messagebox.showinfo("成功", "已複製到剪貼簿!")
        
        ttk.Button(btn_frame, text="📋 複製", command=copy_to_clipboard).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="關閉", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def select_output_dir(self):
        """選擇輸出資料夾"""
        dir_path = filedialog.askdirectory(title="選擇輸出資料夾")
        if dir_path:
            self.output_dir = dir_path
            self.output_label.config(text=dir_path, foreground="black")
    
    def start_splitting(self):
        """開始分割影片"""
        # 驗證
        if not self.video_path:
            messagebox.showwarning("警告", "請先載入影片!")
            return
        
        if not self.segments:
            messagebox.showwarning("警告", "請至少新增一個分割區間!")
            return
        
        if not hasattr(self, 'output_dir'):
            messagebox.showwarning("警告", "請選擇輸出資料夾!")
            return
        
        if not self.has_ffmpeg:
            messagebox.showerror("錯誤", "未找到 FFmpeg，請先安裝!")
            return
        
        # 在背景執行緒執行分割
        thread = threading.Thread(target=self.split_video_thread, daemon=True)
        thread.start()
    
    def split_video_thread(self):
        """背景執行緒:分割影片"""
        try:
            total = len(self.segments)
            encoder = self.encoder_var.get()
            
            self.progress_bar['maximum'] = total
            
            for idx, (start, end, name) in enumerate(self.segments, 1):
                # 更新進度
                self.progress_label.config(text=f"正在處理: {name} ({idx}/{total})")
                self.progress_bar['value'] = idx - 1
                self.root.update()
                
                # 建構 FFmpeg 命令
                output_file = Path(self.output_dir) / f"{name}.mp4"
                duration = end - start
                
                if encoder == "copy":
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(start),
                        '-i', self.video_path,
                        '-t', str(duration),
                        '-c', 'copy',
                        str(output_file)
                    ]
                elif encoder == "h264_nvenc":
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(start),
                        '-i', self.video_path,
                        '-t', str(duration),
                        '-c:v', 'h264_nvenc',
                        '-preset', 'fast',
                        '-b:v', '5M',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        str(output_file)
                    ]
                else:  # libx264
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(start),
                        '-i', self.video_path,
                        '-t', str(duration),
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        str(output_file)
                    ]
                
                # 執行 FFmpeg
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # 完成
            self.progress_bar['value'] = total
            self.progress_label.config(text=f"✅ 完成! 已輸出 {total} 個片段", foreground="green")
            messagebox.showinfo("完成", f"影片分割完成!\n共輸出 {total} 個片段至:\n{self.output_dir}")
            
        except Exception as e:
            self.progress_label.config(text=f"❌ 錯誤: {str(e)}", foreground="red")
            messagebox.showerror("錯誤", f"分割失敗:\n{str(e)}")


def main():
    """主程式入口"""
    try:
        root = tk.Tk()
        app = VideoSplitterGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\n✅ 程式已正常退出")
    except Exception as e:
        print(f"\n❌ 程式錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
