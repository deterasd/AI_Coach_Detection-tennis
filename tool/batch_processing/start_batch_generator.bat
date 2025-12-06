@echo off
chcp 65001 >nul
echo ========================================
echo 啟動批量 3D 軌跡產生器
echo ========================================
echo.

cd /d "%~dp0"
python batch_3d_trajectory_generator.py

if errorlevel 1 (
    echo.
    echo 執行失敗！請檢查：
    echo 1. Python 是否已安裝
    echo 2. 必要的套件是否已安裝 ^(numpy, opencv-python, ultralytics, tkinter^)
    echo.
    pause
)
