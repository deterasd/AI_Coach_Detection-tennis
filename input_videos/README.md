# 影片檔案放置說明

請將你的網球影片檔案放在這個資料夾中：

## 檔案命名規則

### 側面影片 (必須)
- 檔名包含: `side`, `側面`, `lateral` 等關鍵字
- 例如: `tennis_side.mp4`, `網球_側面.avi`, `lateral_view.mov`

### 45度影片 (必須)  
- 檔名包含: `45`, `角度`, `angle` 等關鍵字
- 例如: `tennis_45.mp4`, `網球_45度.avi`, `angle_view.mov`

## 支援的影片格式
- .mp4 (建議)
- .avi
- .mov  
- .mkv

## 檔案要求
- 兩個角度的影片需要同時錄製
- 確保網球在影片中清晰可見
- 建議使用穩定的攝影設備
- 影片解析度建議 1080p 以上

## 範例檔案結構
```
input_videos/
├── tennis_side.mp4      ← 側面角度影片
├── tennis_45.mp4        ← 45度角度影片
└── README.md           ← 此說明檔案
```

## 如果沒有關鍵字
如果檔名沒有包含關鍵字，程式會按字母順序自動分配：
- 第一個檔案 → 側面影片
- 第二個檔案 → 45度影片

準備好影片後，執行 `quick_test.bat` 或 `python trajector_processing_simple_test.py` 開始分析！