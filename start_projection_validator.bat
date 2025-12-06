@echo off
chcp 65001 > nul
echo ========================================
echo 投影矩陣驗算工具
echo ========================================
echo.
echo 正在啟動圖形化介面...
echo.

python binocular_correction\projection_matrix_validator_gui.py

if errorlevel 1 (
    echo.
    echo ❌ 啟動失敗！
    echo.
    echo 可能原因：
    echo 1. Python 未安裝或不在 PATH 中
    echo 2. 缺少必要套件
    echo.
    echo 請執行以下命令安裝套件：
    echo pip install numpy pandas matplotlib
    echo.
    pause
) else (
    echo.
    echo ✅ 程式已正常結束
)
