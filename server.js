const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// 新增：簡單的 CORS 中間件
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});

app.use(express.static(path.join(__dirname)));

// 靜音 favicon.ico 404 錯誤 (開發環境常用)
app.get('/favicon.ico', (req, res) => res.status(204).end());

app.listen(port, () => {
    console.log(`伺服器運行於 http://localhost:${port}`);
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'drawing_2D_chart_js.html'));
});

app.get('/getFolders', (req, res) => {
    const assetsDir = path.join(__dirname, 'trajectory');
    // console.log(`[API] 正在讀取資料夾列表: ${assetsDir}`);
    fs.readdir(assetsDir, (err, files) => {
        if (err) {
            // console.error("讀取資料夾失敗：", err);
            return res.status(500).json({ error: '無法讀取資料夾' });
        }
        // 只保留目錄（資料夾）
        const folders = files.filter(file => {
            const filePath = path.join(assetsDir, file);
            return fs.statSync(filePath).isDirectory();
        });
        res.json(folders);
    });
});

app.get('/getVideos', (req, res) => {
    const folder = req.query.folder;
    if (!folder) {
        return res.status(400).json({ error: '請提供 folder 參數' });
    }
    const videoDir = path.join(__dirname, 'trajectory', folder);
    
    // 先檢查目錄是否存在
    if (!fs.existsSync(videoDir)) {
        return res.json([]); // 目錄不存在，回傳空列表而不是報錯 500
    }

    fs.readdir(videoDir, (err, files) => {
        if (err) {
            return res.status(500).json({ error: `無法讀取 ${folder} 資料夾` });
        }
        // 過濾出副檔名為 .mp4 的檔案，或標記檔案 ready.txt
        const resultFiles = files.filter(file => 
            file.toLowerCase().endsWith('.mp4') || file === 'ready.txt'
        );
        res.json(resultFiles);
    });
});


app.get("/getjson", (req, res) => {
    const dir = req.query.dir;
    if (!dir) {
        return res.status(400).json({ error: "缺少 dir 參數" });
    }

    const directoryPath = path.join(__dirname, dir);

    fs.readdir(directoryPath, (err, files) => {
        if (err) {
            return res.status(500).json({ error: "無法讀取目錄", details: err.message });
        }

        // 僅回傳 .json 檔案
        const jsonFiles = files.filter(file => file.endsWith(".json"));
        res.json(jsonFiles);
    });
});