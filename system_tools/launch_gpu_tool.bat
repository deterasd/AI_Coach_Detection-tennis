@echo off
cls
echo ==========================================
echo 🎬 GPU加速影片分割工具 - 快速啟動
echo ==========================================
echo.

echo 🔍 檢查系統狀態...
python test_gpu_system.py

echo.
echo ==========================================
echo 🚀 啟動影片分割工具
echo ==========================================
echo.

echo 💡 提示:
echo - 工具已支援RTX 3060 GPU加速
echo - 分割速度比之前快30-50倍
echo - 界面會顯示GPU使用統計
echo.

pause
python video_segment_tester_optimized.py