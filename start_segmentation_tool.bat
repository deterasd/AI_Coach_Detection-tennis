@echo off
echo.
echo ==========================================
echo   AI Tennis Video Segmentation Tool
echo ==========================================
echo.

:: 檢查 Python 環境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 找不到 Python，請安裝 Python。
    pause
    exit /b
)

:: 執行圖形化工具
python video_segmentation_gui.py

pause
