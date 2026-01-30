# 軌跡處理模組整理清單

## 📋 目前保留的檔案

### 核心處理引擎
- **`trajector_processing_unified.py`** (680 行)
  - **用途**: 核心軌跡分析引擎（重構版）
  - **功能**: 完整的11步驟軌跡分析流程
  - **特點**: 
    - 模塊化設計 - 分割邏輯已提取到獨立模組
    - 支援多球分析，每球獨立資料夾
    - 有 `if __name__ == "__main__"` 可直接執行
  - **使用**: 
    ```bash
    python trajector_processing_unified.py --video_side 側面.mp4 --video_45 45度.mp4 --name 使用者
    # 或在程式中 import
    from trajector_processing_unified import processing_trajectory_unified
    ```

### 分割模組
- **`trajectory_video_segmentation.py`** (1026 行)
  - **用途**: 影片自動分割模組（新增）
  - **功能**: 
    - 智能偵測球進入/出場時間點
    - 動態分割影片片段
    - 多執行緒讀取優化 (ThreadedVideoCapture)
    - 批次推理 + 稀疏掃描 + FP16加速
    - GPU/CPU/軟體編碼三重回退
  - **使用**:
    ```python
    from trajectory_video_segmentation import process_video_segmentation
    results = process_video_segmentation(video_side, video_45, model, name, output_folder)
    ```

### 測試工具
- **`trajector_processing_simple_test.py`** (541 行)
  - **用途**: 互動式測試入口點
  - **功能**: 
    - 使用者資訊輸入
    - YOLO 模型載入
    - 完整流程呼叫
    - 循環處理多個使用者
  - **使用**:
    ```bash
    python trajector_processing_simple_test.py
    ```

---

## 🗂️ 已歸檔的舊版本

所有過期/重複的版本已移至: `_archive/deprecated_processing/`

| 檔案 | 行數 | 原因 |
|------|------|------|
| `trajector_processing_streaming.py` | 1792 | 被 unified.py 取代 |
| `trajector_processing_unified_pipeline.py` | 773 | 被 unified.py 取代 |
| `trajector_processing_with_segmentation.py` | 349 | 分割功能已獨立 |
| `trajector_processing_pipeline_test.py` | 115 | 功能被 simple_test.py 取代 |
| `trajector_processing_unified_backup_v1941lines.py` | 1878 | 重構前的備份 |

---

## 🔄 關鍵檔案關係

```
trajector_processing_simple_test.py (測試入口)
    ↓
    └─→ 呼叫 processing_trajectory_unified()
        
trajector_processing_unified.py (核心引擎)
    ├─→ 匯入 trajectory_video_segmentation
    │   └─→ process_video_segmentation()
    ├─→ 匯入 trajector_2D_smoothing, video_detection 等
    └─→ 11步驟處理流程
        ├─→ process_multiple_balls() (多球)
        └─→ process_single_video_set() (單球)
```

---

## 📊 程式碼行數對比

| 階段 | unified.py | 分割模組 | 總和 |
|------|-----------|--------|------|
| 重構前 | 1941行 | - | 1941行 |
| 重構後 | 680行 | 1026行 | **1706行** |
| **精簡比例** | **-65%** | - | **-12%** |

✅ 重構帶來：
- unified.py 代碼量減少 65% (更容易維護)
- 分割邏輯完全獨立 (可單獨測試/更新)
- 層級結構清晰 (入口→引擎→模組)

---

## 🚀 後續優化方向

### 待檢查
- [ ] 確認所有舊版本都已不需要
- [ ] 測試 simple_test.py 完整流程
- [ ] 驗證模組匯入沒有循環依賴

### 未來改進
- [ ] 考慮將更多模組化 (如 trajectory_knn.py → knn_module/)
- [ ] 統一錯誤處理和日誌系統
- [ ] 建立完整的模組文件

