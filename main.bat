@echo off
start cmd /k ".\.venv\Scripts\activate && python camera_control(FASTAPI)_3253.py"
timeout /t 2 >nul
start cmd /k ".\.venv\Scripts\activate && python camera_control(FASTAPI)_9436.py"
timeout /t 2 >nul
start cmd /k ".\.venv\Scripts\activate && python main.py"
timeout /t 2 >nul
start cmd /k "node server.js"
timeout /t 2 >nul
start "" "http://localhost:3000/drawing_2D_chart_js.html"
start "" "http://localhost:3000/drawing_3D_three_js.html"
start "" "http://localhost:8000/docs#/"