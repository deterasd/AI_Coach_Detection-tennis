@echo off
echo 正在建立新的 Python 3.11 虛擬環境...

REM 刪除舊的虛擬環境
if exist .venv_old (
    rmdir /s /q .venv_old
)

REM 備份現有虛擬環境
if exist .venv (
    echo 備份現有虛擬環境到 .venv_old...
    move .venv .venv_old
)

REM 檢查是否有 Python 3.11
python3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.11 未安裝，請先安裝 Python 3.11.9
    echo 下載地址: https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

REM 使用 Python 3.11 建立新虛擬環境
echo 使用 Python 3.11 建立新虛擬環境...
python3.11 -m venv .venv

REM 啟動新虛擬環境
echo 啟動虛擬環境...
call .venv\Scripts\activate.bat

REM 升級 pip
echo 升級 pip...
python -m pip install --upgrade pip

REM 安裝依賴
echo 安裝專案依賴...
pip install -r requirements.txt

echo.
echo 完成！新的 Python 3.11 虛擬環境已建立。
echo 請執行 .venv\Scripts\activate 來啟動虛擬環境。
pause