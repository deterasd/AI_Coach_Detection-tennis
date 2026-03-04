#!/bin/bash

# 1. 啟動第一台攝影機的 FastAPI (背景執行)
echo "Starting Camera 1 FastAPI..."
../.venv/bin/python "camera_control(FASTAPI)_3253.py" &
sleep 2

# 2. 啟動第二台攝影機的 FastAPI (背景執行)
echo "Starting Camera 2 FastAPI..."
../.venv/bin/python "camera_control(FASTAPI)_9436.py" &
sleep 2

# 3. 啟動主要的 FastAPI 伺服器 (背景執行)
echo "Starting Main FastAPI server..."
../.venv/bin/python "main.py" &
sleep 10

# 4. 啟動 Node.js 伺服器 (背景執行)
echo "Starting Node.js server..."
node server.js &
sleep 5

# 5. 開啟瀏覽器網頁
echo "Opening web pages..."
open "http://localhost:3001/drawing_2D_chart_js.html"
open "http://localhost:3001/3d"
open "http://localhost:8000/docs#/"

echo "All services started! Press Ctrl+C to stop all background processes."

# 讓腳本等待，以防立刻結束。捕捉 Ctrl+C 來一次關閉所有背景程序
trap "kill 0" SIGINT SIGTERM EXIT
wait
