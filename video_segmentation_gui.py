import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
import time
from pathlib import Path

# 確保可以匯入 root 目錄的模組
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from trajector_processing_unified import detect_ball_entries_optimized, segment_video_dynamic
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ 匯入失敗: {e}")
    print("請確保在專案根目錄執行此程式，且已安裝 ultralytics")

class VideoSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 網球自動分割工具 - Powered by GitHub Copilot")
        self.root.geometry("700x600")
        
        # 設定變數
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar(value="model/tennisball_OD_v1.pt")
        self.ball_direction = tk.StringVar(value="right")
        self.user_name = tk.StringVar(value="Exhibition")
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 標題
        ttk.Label(main_frame, text="網球影片自動分割工具", font=("Helvetica", 16, "bold")).pack(pady=(0, 20))
        
        # 檔案選取區域
        file_frame = ttk.LabelFrame(main_frame, text="設定參數", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        # 影片選取
        ttk.Label(file_frame, text="選擇影片:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="瀏覽...", command=self.browse_video).grid(row=0, column=2)
        
        # 輸出路徑
        ttk.Label(file_frame, text="存放目錄:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="瀏覽...", command=self.browse_output).grid(row=1, column=2)
        
        # 模型選取
        ttk.Label(file_frame, text="偵測模型:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.model_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="瀏覽...", command=self.browse_model).grid(row=2, column=2)

        # 進階設定
        opt_frame = ttk.Frame(file_frame)
        opt_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=10)
        
        ttk.Label(opt_frame, text="進場方向:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(opt_frame, text="右邊進場", variable=self.ball_direction, value="right").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(opt_frame, text="左邊進場", variable=self.ball_direction, value="left").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(opt_frame, text="  命名標記:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(opt_frame, textvariable=self.user_name, width=15).pack(side=tk.LEFT)
        
        # 按鈕區域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="🚀 開始分割影片", command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # 日誌區域
        log_frame = ttk.LabelFrame(main_frame, text="處理日誌", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, state=tk.DISABLED, bg="#f0f0f0")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 進度條
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()

    def browse_video(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.MP4 *.avi *.mov")])
        if filename:
            self.video_path.set(filename)
            # 自動推測輸出目錄
            if not self.output_dir.get():
                self.output_dir.set(str(Path(filename).parent / "segmented_output"))

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def browse_model(self):
        filename = filedialog.askopenfilename(filetypes=[("YOLO models", "*.pt")])
        if filename:
            self.model_path.set(filename)

    def start_processing(self):
        v_path = self.video_path.get()
        o_dir = self.output_dir.get()
        m_path = self.model_path.get()
        
        if not v_path or not os.path.exists(v_path):
            messagebox.showerror("錯誤", "請選擇有效的影片檔案")
            return
        
        if not m_path or not os.path.exists(m_path):
            messagebox.showerror("錯誤", "模型檔案不存在，請檢查路徑")
            return

        # 鎖定 UI
        self.start_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        # 啟動執行緒
        thread = threading.Thread(target=self.run_process, args=(v_path, o_dir, m_path))
        thread.daemon = True
        thread.start()

    def run_process(self, video_path, output_dir, model_path):
        try:
            self.log("🚀 初始化模型中...")
            model = YOLO(model_path)
            
            self.log(f"🔍 開始分析影片: {os.path.basename(video_path)}")
            self.log(f"📌 進場方向設定: {self.ball_direction.get()}")
            
            # 建立輸出目錄
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用 trajector_processing_unified 中的偵測邏輯
            # 重新導向 stdout 以擷取 print 訊息 (可選，但這裡我們手動 log 重點)
            entries, exits = detect_ball_entries_optimized(
                video_path, 
                model, 
                confidence_threshold=0.5,
                ball_entry_direction=self.ball_direction.get(),
                enable_exit_detection=True,
                exit_timeout=1.5
            )
            
            self.log(f"📊 偵測完成！找到 {len(entries)} 顆球。")
            
            if not entries:
                self.log("⚠️ 偵測不到任何球，請檢查進場方向或影片內容。")
            else:
                self.log("✂️ 開始執行實體分割...")
                # 執行分割
                segments = segment_video_dynamic(
                    video_path, 
                    entries, 
                    exits, 
                    output_dir, 
                    self.user_name.get(), 
                    "angle", # 固定標記
                    preview_start_time=-0.2
                )
                self.log(f"✅ 分割完成！成功建立 {len(segments)} 個片段檔案。")
                self.log(f"路徑: {output_dir}")
                
        except Exception as e:
            self.log(f"❌ 發生錯誤: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.start_btn.config(state=tk.NORMAL)
            self.progress.stop()
            messagebox.showinfo("完成", "處理流程已結束")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSegmentationGUI(root)
    root.mainloop()
