@echo off
start cmd /k ".\.venv\Scripts\python.exe camera_control(FASTAPI)_3253.py"
timeout /t 2 >nul
start cmd /k ".\.venv\Scripts\python.exe camera_control(FASTAPI)_9436.py"
timeout /t 2 >nul
start cmd /k ".\.venv\Scripts\python.exe main.py"
timeout /t 10 >nul
start cmd /k "node server.js"
timeout /t 5 >nul
start "" "http://localhost:3001/drawing_2D_chart_js.html"
start "" "http://localhost:3001/3d"
REM FastAPI Swagger docs 已在 main.py 啟動於 port 8000
REM 如需靜態文件服務器，可改用其他端口如 8001
REM start cmd /k "python -m http.server 8001"
start "" "http://localhost:8000/docs#/"