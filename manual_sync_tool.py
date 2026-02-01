#!/usr/bin/env python3
import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np

class ManualSyncTool:
    def __init__(self, root):
        self.root = root
        self.root.title("手動影片同步工具")
        self.root.geometry("1000x700")
        
        self.video_45 = None
        self.video_side = None
        self.current_frame_45 = 0
        self.current_frame_side = 0
        self.offset = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="載入45度影片", command=self.load_video_45).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="載入側面影片", command=self.load_video_side).pack(side=tk.LEFT, padx=5)
        
        # 偏移控制
        offset_frame = ttk.Frame(control_frame)
        offset_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(offset_frame, text="偏移:").pack(side=tk.LEFT)
        self.offset_var = tk.IntVar(value=0)
        offset_spinbox = ttk.Spinbox(offset_frame, from_=-100, to=100, textvariable=self.offset_var, width=10)
        offset_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Button(offset_frame, text="套用", command=self.apply_offset).pack(side=tk.LEFT, padx=5)
        
        # 播放控制
        play_frame = ttk.Frame(self.root)
        play_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(play_frame, text="播放", command=self.play_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(play_frame, text="暫停", command=self.pause_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(play_frame, text="重置", command=self.reset_videos).pack(side=tk.LEFT, padx=5)
        
        # 影片顯示區域
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 狀態標籤
        self.status_label = ttk.Label(self.root, text="請載入影片")
        self.status_label.pack(pady=5)
    
    def load_video_45(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(title="選擇45度影片", filetypes=[("影片檔案", "*.mp4 *.avi")])
        if filename:
            self.video_45 = cv2.VideoCapture(filename)
            self.status_label.config(text=f"45度影片已載入: {filename}")
    
    def load_video_side(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(title="選擇側面影片", filetypes=[("影片檔案", "*.mp4 *.avi")])
        if filename:
            self.video_side = cv2.VideoCapture(filename)
            self.status_label.config(text=f"側面影片已載入: {filename}")
    
    def apply_offset(self):
        self.offset = self.offset_var.get()
        self.status_label.config(text=f"偏移已設定: {self.offset} 幀")
    
    def play_videos(self):
        if self.video_45 and self.video_side:
            self.status_label.config(text="播放中...")
            # 這裡可以實現實際的播放邏輯
    
    def pause_videos(self):
        self.status_label.config(text="已暫停")
    
    def reset_videos(self):
        if self.video_45:
            self.video_45.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.video_side:
            self.video_side.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.status_label.config(text="已重置")

if __name__ == "__main__":
    root = tk.Tk()
    app = ManualSyncTool(root)
    root.mainloop()
