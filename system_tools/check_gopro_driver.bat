@echo off
echo 檢查 GoPro 裝置驅動程式狀態...
echo.

echo === 檢查連接的 USB 裝置 ===
wmic path Win32_USBControllerDevice get Dependent | findstr /i "gopro"
if %errorlevel% equ 0 (
    echo 找到 GoPro 裝置
) else (
    echo 未找到 GoPro 裝置，請確認：
    echo 1. GoPro 已開機並連接 USB 線
    echo 2. GoPro 設定為 USB 模式
)

echo.
echo === 檢查裝置管理員中的 GoPro ===
devmgmt.msc

echo.
echo === 檢查 USB 驅動程式 ===
pnputil /enum-drivers | findstr /i "gopro"

echo.
echo 請檢查裝置管理員中是否有：
echo - "便攜式裝置" 下的 GoPro
echo - "通用序列匯流排控制器" 下的相關驅動
echo - 任何帶有黃色驚嘆號的裝置
echo.
pause