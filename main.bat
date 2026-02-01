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
start "" "http://localhost:8000/docs#/"