@echo off
REM 3D軌跡重建誤差分析工具啟動器
echo ========================================
echo 🎾 3D軌跡重建誤差分析工具
echo ========================================
echo.

REM 檢查Python環境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 找不到Python，請確保已安裝Python
    pause
    exit /b 1
)

REM 檢查虛擬環境
if exist ".venv\Scripts\activate.bat" (
    echo ✅ 找到虛擬環境，正在啟動...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  找不到虛擬環境，將使用系統Python
)

echo.
echo 🚀 啟動GUI工具...
python binocular_correction/trajectory_analysis_gui.py

echo.
echo 工具已關閉
pause