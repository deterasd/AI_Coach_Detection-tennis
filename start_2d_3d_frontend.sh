#!/bin/bash
# 一鍵啟動 2D / 3D 前端：啟動 Node 靜態伺服器（port 3001）並用瀏覽器開啟兩頁。
# 使用方式：在專案目錄執行 ./start_2d_3d_frontend.sh 或 bash start_2d_3d_frontend.sh

cd "$(dirname "$0")"
PORT=3001

# 若 3001 已有程式在跑，只開瀏覽器
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/" 2>/dev/null | grep -q "200"; then
    echo "偵測到 port ${PORT} 已有伺服器，直接開啟 2D / 3D 頁面..."
    open "http://localhost:${PORT}/"
    open "http://localhost:${PORT}/3d"
    echo "已開啟 2D、3D 前端。"
    exit 0
fi

# 啟動 Node 伺服器（背景）
echo "正在啟動 Node 伺服器（port ${PORT}）..."
PORT=$PORT nohup node server.js > /tmp/pickleball_2d_3d_server.log 2>&1 &
NODE_PID=$!
sleep 2

if ! kill -0 $NODE_PID 2>/dev/null; then
    echo "錯誤：Node 伺服器啟動失敗，請查看 /tmp/pickleball_2d_3d_server.log"
    exit 1
fi

echo "伺服器已啟動（PID $NODE_PID），正在開啟瀏覽器..."
open "http://localhost:${PORT}/"
open "http://localhost:${PORT}/3d"

echo ""
echo "2D 前端: http://localhost:${PORT}/ 或 http://localhost:${PORT}/drawing_2D_chart_js.html"
echo "3D 前端: http://localhost:${PORT}/3d"
echo "伺服器於背景執行，日誌: /tmp/pickleball_2d_3d_server.log"
echo "關閉伺服器: kill $NODE_PID"
