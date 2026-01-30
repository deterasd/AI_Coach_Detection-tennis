const videoPlayer = document.getElementById('videoPlayer');
// --- Speed ---------------------------------------------
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');

function updateSpeed() {
    const speed = Number(speedControl.value);
    videoPlayer.playbackRate = speed;
    speedValue.textContent = `${speed.toFixed(2)}x`;

    const min = Number(speedControl.min);
    const max = Number(speedControl.max);
    const percent = ((speed - min) / (max - min)) * 100;
    speedControl.style.setProperty('--val', percent + '%');
}

speedControl.addEventListener('input', updateSpeed);
updateSpeed();


// ---Video to Json---------------------------------------------
const folderSelect = document.getElementById('folderSelect');
const videoSelect = document.getElementById('videoSelect');
const basePath = "./trajectory/";

async function fetchFolderList() {
    try {
        const response = await fetch('/getFolders');
        const folders_full = await response.json();
        const folders = folders_full.map(folder_full => folder_full.split('__')[0]);
        folderSelect.innerHTML = '<option value="">Player Name</option>' + folders.map(folder => `<option value="${folder}">${folder}</option>`).join('');
    } catch (error) {
        console.error("Unable to fetch folder list:", error);
    }
}

async function fetchVideoList(folder, autoSelectFirst = false) {
    // 新增預設選項
    videoSelect.innerHTML = `<option value="">select trajectory</option>`;
    for (let i = 1; i <= 100; i++) {
        try {
            const currentTrajectory = `trajectory_${i}`;
            const response = await fetch(`/getVideos?folder=${folder}__trajectory/${currentTrajectory}`);
            if (!response.ok) continue;

            const videos_all = await response.json();
            const videos = videos_all.filter(v => v.includes('full_video'));
            
            if (videos.length > 0) {
                const videoFile = videos[0];
                const optionValue = `${basePath}${folder}/${currentTrajectory}/${videoFile}`;
                const option = document.createElement('option');
                option.value = optionValue;
                option.textContent = videoFile;
                videoSelect.appendChild(option);

                // 如果是自動選擇模式且是第一球
                if (autoSelectFirst && i === 1) {
                    videoSelect.value = optionValue;
                    // 手動觸發 change 事件以載入影片和 JSON
                    videoSelect.dispatchEvent(new Event('change'));
                    highlightSelection();
                }
            }
        } catch (error) {
            continue;
        }
    }
}

function highlightSelection() {
    const containers = [folderSelect, videoSelect];
    containers.forEach(el => {
        el.classList.add('highlight-pulse');
        setTimeout(() => el.classList.remove('highlight-pulse'), 5000);
    });
}

folderSelect.addEventListener('change', e => {
    const selectedFolder = e.target.value;
    selectedFolder ? fetchVideoList(selectedFolder) : videoSelect.innerHTML = '<option value="">Choose Video</option>';
});

// --- Polling for First Ball Ready ---
let lastCheckedFolder = null;
let isFirstBallNotified = false;

async function checkFirstBallReady() {
    try {
        const response = await fetch('/getFolders');
        if (!response.ok) return;
        const folders = await response.json();
        if (folders.length === 0) return;

        // 排序取得最新的一個
        folders.sort();
        const latestFolderFull = folders[folders.length - 1]; // 例如 "John__trajectory"
        const cleanName = latestFolderFull.split('__')[0];

        // 如果換了新資料夾（新客戶），重設通知狀態
        if (latestFolderFull !== lastCheckedFolder) {
            lastCheckedFolder = latestFolderFull;
            isFirstBallNotified = false;
        }

        if (isFirstBallNotified) return;

        // 檢查 trajectory_1 資料夾下是否有 ready.txt
        const videoResponse = await fetch(`/getVideos?folder=${latestFolderFull}/trajectory_1`);
        if (videoResponse.ok) {
            const files = await videoResponse.json();
            if (files.includes('ready.txt')) {
                notifyFirstBall(cleanName);
                isFirstBallNotified = true;
            }
        }
    } catch (error) {
        console.error("Polling error:", error);
    }
}

function notifyFirstBall(playerName) {
    // 視覺通知 (升級版)
    const notification = document.createElement('div');
    notification.id = 'readyNotification';
    notification.style.cssText = `
        position: fixed;
        top: 25px;
        right: 25px;
        background: rgba(30, 30, 30, 0.95);
        color: white;
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        z-index: 10000;
        min-width: 300px;
        animation: slideIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    `;
    notification.innerHTML = `
        <div style="margin-bottom: 8px; font-size: 20px; color: #4CAF50;"><strong>🔔 分析完成！</strong></div>
        <div style="margin-bottom: 18px; color: #eee; font-size: 16px;">客戶 <strong>${playerName}</strong> 的第一球結果已產出。</div>
        <div style="display: flex; gap: 10px;">
            <button id="viewResultBtn" style="background: #4CAF50; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-size: 16px; flex: 2; font-weight: bold;">立即查看結果</button>
            <button id="closeNotifyBtn" style="background: transparent; color: #999; border: 1px solid #444; padding: 10px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; flex: 1;">忽略</button>
        </div>
    `;
    document.body.appendChild(notification);

    // 語音通知
    const msg = new SpeechSynthesisUtterance(`${playerName}的第一顆球結果已產出，請至大螢幕查看。`);
    msg.lang = "zh-TW";
    window.speechSynthesis.speak(msg);

    // 點擊「立即查看」
    document.getElementById('viewResultBtn').onclick = () => {
        // 自動選擇下拉選單
        const options = Array.from(folderSelect.options);
        const targetOption = options.find(opt => opt.value === playerName);
        if (targetOption) {
            folderSelect.value = playerName;
            fetchVideoList(playerName, true); // true 表示自動選擇第一球
        }
        notification.remove();
    };

    document.getElementById('closeNotifyBtn').onclick = () => notification.remove();

    // 20秒後自動消失
    setTimeout(() => {
        if (document.getElementById('readyNotification')) {
            notification.style.opacity = '0';
            notification.style.transition = 'opacity 1s ease';
            setTimeout(() => notification.remove(), 1000);
        }
    }, 20000);
}

// 每 3 秒檢查一次
setInterval(checkFirstBallReady, 3000);

document.addEventListener('DOMContentLoaded', () => {
    fetchFolderList();
    
    // 加入動畫樣式
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes slideIn {
            from { transform: translateX(120%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .highlight-pulse {
            animation: pulse-border 1.5s infinite;
            border: 2px solid #4CAF50 !important;
        }
        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(76, 175, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }
    `;
    document.head.appendChild(style);
});

videoSelect.addEventListener('change', e => {
    const selected_path = e.target.value;
    let pathParts = selected_path.split('/');
    pathParts[2] = pathParts[2] + '__trajectory';
    // pathParts[3] = pathParts[3].replace('trajectory_', 'trajectory__');
    const selectedVideo = pathParts.join('/');
    console.log(selectedVideo);

    if (selectedVideo) {
        videoPlayer.src = selectedVideo;
        console.log("TARGET DEBUG", selectedVideo)

        const playPromise = videoPlayer.play();
        if (playPromise !== undefined) {
            playPromise.catch(error => { error });
        }

        const pathParts = selectedVideo.split('/');
        if (pathParts.length >= 4) {
            const folderName = pathParts[2];
            const fileName = pathParts[3];
            const trajectory = pathParts[4];
            const prefix = trajectory.replace('_full_video.mp4', '');
            console.log("資料夾名稱：", folderName, "檔案名稱：", fileName, "tra：", trajectory, "prefix：", prefix, "basePath：", basePath);

            const Json_45_Path = `${basePath}${folderName}/${fileName}/${prefix}_45(2D_trajectory_smoothed).json`;
            const Json_side_Path = `${basePath}${folderName}/${fileName}/${prefix}_side(2D_trajectory_smoothed).json`;

            handleFileSelection(Json_45_Path, Json_side_Path);
        }
    }
});


async function handleFileSelection(filePath45, filePathSide) {
    try {
        const response45 = await fetch(filePath45);
        if (!response45.ok) {
            console.error("45 degree file not found, please check if path is correct:", filePath45);
            throw new Error(`HTTP error! Status: ${response45.status}`);
        }
        const data45 = await response45.json();
        const filename45 = filePath45.split('/').pop();
        console.log("45 degree JSON file loaded successfully, filename:", filename45);
        document.getElementById('filename2').textContent = filename45;
        createChart('trajectoryChart2', data45);
    } catch (error) {
        console.error("Failed to load 45 degree JSON file:", error);
    }
    try {
        const responseSide = await fetch(filePathSide);
        if (!responseSide.ok) {
            console.error("Side file not found, please check if path is correct:", filePathSide);
            throw new Error(`HTTP error! Status: ${responseSide.status}`);
        }
        const dataSide = await responseSide.json();
        const filenameSide = filePathSide.split('/').pop();
        console.log("Side JSON file loaded successfully, filename:", filenameSide);
        document.getElementById('filename1').textContent = filenameSide;
        createChart('trajectoryChart1', dataSide);
    } catch (error) {
        console.error("Failed to load Side JSON file:", error);
    }
}

function createChart(canvasId, data) {
    console.log("createChart executed", canvasId, data);
    const points = data
        .map(frame => ({
            x: frame.right_wrist?.x,
            y: frame.right_wrist?.y,
            frame: frame.frame
        }))
        .filter(point => point.x != null && point.y != null);
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    charts[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Wrist Trajectory',
                data: points,
                borderColor: '#4DC4C0',
                backgroundColor: 'rgba(77, 196, 192, 0.5)',
                showLine: true,
                pointRadius: 3,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'X Position' },
                    grid: { color: '#E5E5E5' }
                },
                y: {
                    type: 'linear',
                    reverse: true,
                    title: { display: true, text: 'Y Position' },
                    grid: { color: '#E5E5E5' }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Frame: ${context.raw.frame}, X: ${context.raw.x.toFixed(2)}, Y: ${context.raw.y.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}

let charts = {
    chart1: null,
    chart2: null
};