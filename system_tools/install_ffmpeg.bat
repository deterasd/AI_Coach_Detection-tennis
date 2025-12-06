@echo off
echo 🛠️ FFmpeg 安裝腳本
echo ============================================================

echo 🔍 檢查是否已安裝 Chocolatey...
where choco >nul 2>nul
if %errorlevel% == 0 (
    echo ✅ Chocolatey 已安裝
    echo 📦 開始安裝 FFmpeg...
    choco install ffmpeg -y
    if %errorlevel% == 0 (
        echo ✅ FFmpeg 安裝成功！
    ) else (
        echo ❌ FFmpeg 安裝失敗
    )
) else (
    echo ❌ Chocolatey 未安裝
    echo 📋 請選擇安裝方式：
    echo    1. 自動安裝 Chocolatey + FFmpeg
    echo    2. 手動下載 FFmpeg
    echo.
    set /p choice="請輸入選擇 (1/2): "
    
    if "%choice%"=="1" (
        echo 🔧 正在安裝 Chocolatey...
        powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
        
        echo 📦 正在安裝 FFmpeg...
        choco install ffmpeg -y
        if %errorlevel% == 0 (
            echo ✅ FFmpeg 安裝成功！
        ) else (
            echo ❌ FFmpeg 安裝失敗
        )
    ) else (
        echo 📖 手動安裝指南：
        echo    1. 訪問: https://ffmpeg.org/download.html
        echo    2. 下載 Windows 版本
        echo    3. 解壓縮到 C:\ffmpeg
        echo    4. 將 C:\ffmpeg\bin 添加到系統 PATH
        echo.
        echo 💡 或者使用 winget (Windows 10/11):
        echo    winget install ffmpeg
    )
)

echo.
echo 🧪 測試 FFmpeg 安裝...
ffmpeg -version >nul 2>nul
if %errorlevel% == 0 (
    echo ✅ FFmpeg 可正常使用！
    ffmpeg -version | findstr "ffmpeg version"
) else (
    echo ❌ FFmpeg 仍無法使用，可能需要重啟命令行或檢查 PATH 設定
)

echo.
pause