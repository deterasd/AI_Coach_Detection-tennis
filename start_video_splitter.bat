@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   🎬 影片分割工具 (毫秒精度版本)
echo ========================================
echo.
echo 正在啟動程式...
echo.

python video_splitter_gui.py

if errorlevel 1 (
    echo.
    echo ❌ 程式執行時發生錯誤
    echo.
    pause
) else (
    echo.
    echo ✅ 程式已正常關閉
    echo.
)
