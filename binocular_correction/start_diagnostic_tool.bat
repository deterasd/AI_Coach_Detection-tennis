@echo off
chcp 65001 >nul
echo ========================================
echo 啟動 3D 軌跡數據診斷工具
echo ========================================
echo.

cd /d "%~dp0"
python trajectory_3d_diagnostic_tool.py

if errorlevel 1 (
    echo.
    echo 執行失敗！請檢查：
    echo 1. Python 是否已安裝
    echo 2. 必要的套件是否已安裝 ^(numpy, matplotlib, tkinter^)
    echo.
    pause
)
