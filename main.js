const videoPlayer = document.getElementById("videoPlayer");
// --- Speed ---------------------------------------------
const speedControl = document.getElementById("speedControl");
const speedValue = document.getElementById("speedValue");

function updateSpeed() {
  const speed = Number(speedControl.value);
  videoPlayer.playbackRate = speed;
  speedValue.textContent = `${speed.toFixed(2)}x`;

  const min = Number(speedControl.min);
  const max = Number(speedControl.max);
  const percent = ((speed - min) / (max - min)) * 100;
  speedControl.style.setProperty("--val", percent + "%");
}

speedControl.addEventListener("input", updateSpeed);
updateSpeed();

// --- Frame Control ---（僅在 frameSlider/frameInfo 存在時啟用，drawing_2D_chart_js 無此元素）
const frameInfo = document.getElementById("frameInfo");
const frameSlider = document.getElementById("frameSlider");

function updateFrameDisplay() {
  if (frameInfo && frameSlider && videoPlayer.duration) {
    const fps = 30;
    const currentFrame = Math.floor(videoPlayer.currentTime * fps);
    const totalFrames = Math.floor(videoPlayer.duration * fps);
    frameInfo.textContent = `frame: ${currentFrame}`;
    frameSlider.max = totalFrames;
    frameSlider.value = currentFrame;
  }
}

videoPlayer.addEventListener("timeupdate", updateFrameDisplay);
videoPlayer.addEventListener("loadedmetadata", updateFrameDisplay);

if (frameSlider) {
  frameSlider.addEventListener("input", (e) => {
    const fps = 60;
    const frameNumber = parseInt(e.target.value);
    const timeInSeconds = frameNumber / fps;
    videoPlayer.currentTime = timeInSeconds;
    updateFrameDisplay();
  });
}

// ---Video to Json---------------------------------------------
const folderSelect = document.getElementById("folderSelect");
const videoSelect = document.getElementById("videoSelect");
const basePath = "./trajectory/";

async function fetchFolderList() {
  try {
    const response = await fetch("/getFolders");
    const folderInfos = await response.json();
    console.log("獲取到的文件夾信息:", folderInfos);
    // 按照修改時間遞減排序（最新的在前）
    folderInfos.sort((a, b) => b.mtime - a.mtime);
    console.log("清潔後的文件夾列表:", folderInfos.map((f) => f.name));
    folderSelect.innerHTML =
      '<option value="">Player Name</option>' +
      folderInfos
        .map((info) => `<option value="${info.name}">${info.name}</option>`)
        .join("");
    console.log("文件夾下拉菜單已更新");
  } catch (error) {
    console.error("Unable to fetch folder list:", error);
  }
}

async function fetchVideoList(folder, autoSelectFirst = false) {
  console.log("fetchVideoList called with folder:", folder);
  videoSelect.innerHTML = `<option value="">⏳ 載入中...</option>`;
  videoSelect.disabled = true;

  let firstOptionValue = null;

  try {
    // 步驟1：動態取得子目錄清單（支援任意命名：trajectory_N、player6_N 等）
    const subRes = await fetch(`/getSubfolders?folder=${folder}`);
    const subfolders = subRes.ok ? await subRes.json() : [];
    console.log(`${folder} 的子目錄:`, subfolders);

    videoSelect.innerHTML = `<option value="">select trajectory</option>`;

    if (subfolders.length > 0) {
      // 步驟2：並行取得每個子目錄的影片
      const requests = subfolders.map((sub) => {
        const folderPath = `${folder}/${sub}`;
        return fetch(`/getVideos?folder=${folderPath}`)
          .then((res) => (res.ok ? res.json() : []))
          .then((videos_all) => ({
            sub,
            folderPath,
            videos: pickBestVideos(videos_all),
          }))
          .catch(() => ({ sub, folderPath, videos: [] }));
      });

      const results = await Promise.all(requests);

      results
        .filter((r) => r.videos.length > 0)
        .sort((a, b) => a.sub.localeCompare(b.sub, undefined, { numeric: true }))
        .forEach(({ sub, folderPath, videos }) => {
          const videoFile = videos[0];
          const optionValue = `${basePath}${folderPath}/${videoFile}`;
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = `[${sub}] ${videoFile}`;
          videoSelect.appendChild(option);
          if (firstOptionValue === null) firstOptionValue = optionValue;
          console.log(`已添加選項 [${sub}]:`, videoFile);
        });

    } else {
      // Fallback：flat 結構（如 0129test），直接讀根目錄
      console.log(`${folder} 沒有子目錄，嘗試讀根目錄`);
      const res = await fetch(`/getVideos?folder=${folder}`);
      const videos_all = res.ok ? await res.json() : [];
      const videos = pickBestVideos(videos_all);
      videos.forEach((videoFile) => {
        const optionValue = `${basePath}${folder}/${videoFile}`;
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = videoFile;
        videoSelect.appendChild(option);
        if (firstOptionValue === null) firstOptionValue = optionValue;
        console.log(`已添加根目錄選項:`, videoFile);
      });
    }
  } catch (e) {
    console.error("fetchVideoList 錯誤:", e);
    videoSelect.innerHTML = `<option value="">載入失敗，請重試</option>`;
  }

  videoSelect.disabled = false;

  if (autoSelectFirst && firstOptionValue) {
    videoSelect.value = firstOptionValue;
    videoSelect.dispatchEvent(new Event("change"));
    highlightSelection();
  }

  console.log("fetchVideoList 完成");
}

// 從影片列表中挑選最合適的影片：優先 processed_full_video > full_video > 其他 mp4
function pickBestVideos(videos_all) {
  const preferred = videos_all.filter((v) =>
    v.toLowerCase().endsWith("_processed_full_video.mp4"),
  );
  if (preferred.length > 0) return preferred;
  const withFull = videos_all.filter((v) =>
    v.toLowerCase().endsWith("_full_video.mp4"),
  );
  if (withFull.length > 0) return withFull;
  return videos_all.filter((v) => v.toLowerCase().endsWith(".mp4"));
}

function highlightSelection() {
  const containers = [folderSelect, videoSelect];
  containers.forEach((el) => {
    el.classList.add("highlight-pulse");
    setTimeout(() => el.classList.remove("highlight-pulse"), 5000);
  });
}

folderSelect.addEventListener("change", (e) => {
  const selectedFolder = e.target.value;
  console.log("folderSelect 變化，選擇的文件夾:", selectedFolder);
  if (selectedFolder) {
    fetchVideoList(selectedFolder);
  } else {
    videoSelect.innerHTML = '<option value="">Choose Video</option>';
  }
});

// --- Polling for All Balls Ready ---
let lastCheckedFolder = null;
let notifiedBalls = new Set(); // 記錄已通知過的 "folder-ballNumber" 組合

async function checkFirstBallReady() {
  try {
    const response = await fetch("/getFolders");
    if (!response.ok) return;
    const folderInfos = await response.json();
    if (folderInfos.length === 0) return;

    // 按修改時間排序，取得最新的一個
    folderInfos.sort((a, b) => b.mtime - a.mtime);
    const latestFolderInfo = folderInfos[0];
    const latestFolderFull = latestFolderInfo.name; // 例如 "最8_trajectory"
    const cleanName = latestFolderFull
      .split("_trajectory")[0]
      .replace(/_+$/, ""); // 移除末尾的下劃線

    // 如果換了新資料夾（新客戶），重設通知狀態
    if (latestFolderFull !== lastCheckedFolder) {
      lastCheckedFolder = latestFolderFull;
      notifiedBalls.clear(); // 清空已通知記錄，這樣新客戶的球會重新通知
    }

    // 檢查所有 trajectory_N 資料夾
    for (let ballNum = 1; ballNum <= 50; ballNum++) {
      const ballKey = `${latestFolderFull}-ball${ballNum}`;

      // 如果已經通知過，跳過
      if (notifiedBalls.has(ballKey)) continue;

      try {
        const videoResponse = await fetch(
          `/getVideos?folder=${latestFolderFull}/trajectory_${ballNum}`,
        );
        if (videoResponse.ok) {
          const files = await videoResponse.json();
          if (files.includes("ready.txt")) {
            notifyBallReady(cleanName, ballNum);
            notifiedBalls.add(ballKey); // 標記為已通知
          }
        }
      } catch (e) {
        // 如果找不到該球的資料夾，繼續檢查下一個
      }
    }
  } catch (error) {
    console.error("Polling error:", error);
  }
}

function notifyBallReady(playerName, ballNumber) {
  // 視覺通知 (升級版)
  const notification = document.createElement("div");
  notification.id = "readyNotification";
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
        <div style="margin-bottom: 8px; font-size: 20px; color: #4CAF50;"><strong>🔔 Analysis Complete!</strong></div>
        <div style="margin-bottom: 18px; color: #eee; font-size: 16px;">Player <strong>${playerName}</strong> - Ball <strong>${ballNumber}</strong> result is ready.</div>
        <div style="display: flex; gap: 10px;">
            <button id="viewResultBtn" style="background: #4CAF50; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-size: 16px; flex: 2; font-weight: bold;">View Result</button>
            <button id="closeNotifyBtn" style="background: transparent; color: #999; border: 1px solid #444; padding: 10px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; flex: 1;">Dismiss</button>
        </div>
    `;
  document.body.appendChild(notification);

  // 語音通知 - 修複語音問題
  window.speechSynthesis.cancel(); // 停止任何正在進行的語音
  const msg = new SpeechSynthesisUtterance(
    `Ball ${ballNumber} result for ${playerName} is ready. Please check the display.`,
  );
  msg.lang = "en-US";
  msg.rate = 0.9; // 調整說話速度
  msg.pitch = 1.0;
  msg.volume = 1.0; // 確保音量最大
  window.speechSynthesis.speak(msg);

  // 點擊「立即查看」
  document.getElementById("viewResultBtn").onclick = () => {
    // 自動選擇下拉選單
    const options = Array.from(folderSelect.options);
    const targetOption = options.find((opt) => opt.value === playerName);
    if (targetOption) {
      folderSelect.value = playerName;
      fetchVideoList(playerName, true); // true 表示自動選擇第一球
    }
    notification.remove();
  };

  document.getElementById("closeNotifyBtn").onclick = () =>
    notification.remove();

  // 20秒後自動消失
  setTimeout(() => {
    if (document.getElementById("readyNotification")) {
      notification.style.opacity = "0";
      notification.style.transition = "opacity 1s ease";
      setTimeout(() => notification.remove(), 1000);
    }
  }, 20000);
}

// 每 3 秒檢查一次
setInterval(checkFirstBallReady, 3000);

document.addEventListener("DOMContentLoaded", () => {
  fetchFolderList();

  // 加入動畫樣式
  const style = document.createElement("style");
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

videoSelect.addEventListener("change", (e) => {
  const selectedVideo = e.target.value;
  console.log("Selected video path:", selectedVideo);

  if (selectedVideo) {
    videoPlayer.src = selectedVideo;
    console.log("Loading video:", selectedVideo);

    const playPromise = videoPlayer.play();
    if (playPromise !== undefined) {
      playPromise.catch((error) => {
        console.error("Play error:", error);
      });
    }

    const pathParts = selectedVideo.split("/");
    // pathParts: [".", "trajectory", folderName, (sub?), videoFile]
    if (pathParts.length >= 4) {
      const videoFile = pathParts[pathParts.length - 1];
      // 還原 prefix：去掉常見後綴
      const prefix = videoFile
        .replace(/_processed_full_video\.mp4$/i, "")
        .replace(/_full_video\.mp4$/i, "")
        .replace(/\.mp4$/i, "")
        .replace(/_processed$/i, "");

      // 建構 JSON 所在的目錄路徑（去掉 video 檔名，保留目錄部分）
      const jsonDir = pathParts.slice(0, pathParts.length - 1).join("/");

      // 從 prefix 互推 45 和 side 的 JSON 路徑
      let fortyFivePrefix, sidePrefix;
      if (/_45_segment/i.test(prefix)) {
        fortyFivePrefix = prefix;
        sidePrefix = prefix.replace(/_45_segment/i, "_side_segment");
      } else if (/_side_segment/i.test(prefix)) {
        sidePrefix = prefix;
        fortyFivePrefix = prefix.replace(/_side_segment/i, "_45_segment");
      } else {
        // 無法判斷角度（如 tim401__1）→ 補後綴
        fortyFivePrefix = `${prefix}_45_segment`;
        sidePrefix = `${prefix}_side_segment`;
      }

      const Json_45_Path = `${jsonDir}/${fortyFivePrefix}(2D_trajectory_smoothed).json`;
      const Json_side_Path = `${jsonDir}/${sidePrefix}(2D_trajectory_smoothed).json`;

      console.log("JSON 目錄:", jsonDir, "prefix:", prefix);
      console.log("Loading JSON files:", Json_45_Path, Json_side_Path);
      handleFileSelection(Json_45_Path, Json_side_Path);
    }
  }
});

async function handleFileSelection(filePath45, filePathSide) {
  try {
    const response45 = await fetch(filePath45);
    if (!response45.ok) {
      console.error(
        "45 degree file not found, please check if path is correct:",
        filePath45,
      );
      throw new Error(`HTTP error! Status: ${response45.status}`);
    }
    const data45 = await response45.json();
    const filename45 = filePath45.split("/").pop();
    console.log(
      "45 degree JSON file loaded successfully, filename:",
      filename45,
    );
    document.getElementById("filename2").textContent = filename45;
    createChart("trajectoryChart2", data45);
  } catch (error) {
    console.error("Failed to load 45 degree JSON file:", error);
  }
  try {
    const responseSide = await fetch(filePathSide);
    if (!responseSide.ok) {
      console.error(
        "Side file not found, please check if path is correct:",
        filePathSide,
      );
      throw new Error(`HTTP error! Status: ${responseSide.status}`);
    }
    const dataSide = await responseSide.json();
    const filenameSide = filePathSide.split("/").pop();
    console.log("Side JSON file loaded successfully, filename:", filenameSide);
    document.getElementById("filename1").textContent = filenameSide;
    createChart("trajectoryChart1", dataSide);

    // 載入分析結果（使用 45 度角檔案路徑來提取正確的 prefix）
    // 同時傳入當前播放影片的 src，以便儲入 localStorage
    const currentVideoSrc = videoPlayer ? videoPlayer.src : "";
    await loadAnalysisResults(filePath45, filePathSide, currentVideoSrc);
  } catch (error) {
    console.error("Failed to load Side JSON file:", error);
  }
}

// 載入分析結果
async function loadAnalysisResults(filePath45, filePathSide, videoSrc) {
  try {
    // 優先使用 45 度角檔案來提取 prefix，因為分析檔案通常基於 45 度角檔案名稱
    const pathParts = filePath45
      .replace(/^\.\//, "")
      .split("/")
      .filter(Boolean);
    let folderName, fileName, jsonFile;
    if (pathParts.length >= 3 && pathParts[0] === "trajectory") {
      folderName = pathParts[1];
      if (pathParts.length === 3) {
        fileName = "";
        jsonFile = pathParts[2];
      } else {
        fileName = pathParts[2];
        jsonFile = pathParts[3];
      }
    } else {
      console.error("無法解析檔案路徑:", filePath45);
      return;
    }

    // 從 45 度角檔案名稱提取 prefix
    // 先移除 (2D_trajectory_smoothed).json 後綴，保留前面的所有內容作為 prefix
    let prefix = jsonFile.replace(/\(2D_trajectory_smoothed\)\.json$/i, "");

    // 如果還是沒有匹配到，嘗試從 side 檔案提取
    if (!prefix || prefix === jsonFile) {
      const sidePathParts = filePathSide
        .replace(/^\.\//, "")
        .split("/")
        .filter(Boolean);
      if (sidePathParts.length >= 3 && sidePathParts[0] === "trajectory") {
        const sideJsonFile =
          sidePathParts.length === 3 ? sidePathParts[2] : sidePathParts[3];
        prefix = sideJsonFile
          .replace(/_side\(2D_trajectory_smoothed\)\.json$/i, "")
          .replace(/_side_segment\(2D_trajectory_smoothed\)\.json$/i, "");
      }
    }

    console.log("Loading analysis results for prefix:", prefix);
    console.log("Folder:", folderName, "File:", fileName);

    await loadKNNAnalysis(folderName, fileName, prefix, videoSrc);
    await loadGPTAnalysis(folderName, fileName, prefix);
    document.getElementById("analysisPanel").style.display = "block";
  } catch (error) {
    console.error("Failed to load analysis results:", error);
  }
}

// 格式化建議文字，將 A. B. C. D. 等項目分行顯示
function formatAdviceText(text) {
  if (!text) return "";

  // 使用正則表達式匹配 A. B. C. D. 等項目
  // 匹配模式：大寫字母後跟點號或冒號（A. A: A：）
  // 需要在項目前插入換行，但第一個項目前不需要
  let formatted = text;

  // 匹配 A. B. C. D. 等項目（大寫字母 + 點號/冒號）
  const pattern = /([A-Z][\.:：])/g;

  // 先找到所有匹配位置
  const matches = [];
  let match;
  while ((match = pattern.exec(text)) !== null) {
    matches.push(match.index);
  }

  // 從後往前插入 <br>，避免索引偏移問題
  if (matches.length > 0) {
    let result = text;
    for (let i = matches.length - 1; i > 0; i--) {
      // 只在非第一個項目前插入換行
      const pos = matches[i];
      result = result.slice(0, pos) + "<br>" + result.slice(pos);
    }
    formatted = result;
  } else {
    // 如果沒有匹配到 A. B. C. D. 模式，嘗試按句號、分號分割
    formatted = text.replace(/([。；;])/g, "$1<br>");
    formatted = formatted.replace(/^<br>/, "");
  }

  return formatted;
}

// 分析檔名使用 _segment 後綴（與 pipeline 輸出一致），2D 軌跡檔名可能是 _45_segment / _side_segment
function analysisPrefix(prefix) {
  if (!prefix) return prefix;
  return prefix.replace(/_45_segment$|_side_segment$/i, "_segment");
}

// 載入整合分析結果（取代原本的 KNN 分析）
async function loadKNNAnalysis(folderName, fileName, prefix, videoSrc) {
  try {
    const seg = fileName ? `${fileName}/` : "";
    const prefixForAnalysis = analysisPrefix(prefix);
    let integratedPath = `./trajectory/${folderName}/${seg}${prefix}_integrated_analysis.json`;
    let response = await fetch(integratedPath);
    if (!response.ok && prefixForAnalysis !== prefix) {
      integratedPath = `./trajectory/${folderName}/${seg}${prefixForAnalysis}_integrated_analysis.json`;
      response = await fetch(integratedPath);
    }

    if (response.ok) {
      const integratedData = await response.json();
      const container = document.getElementById("knnSuggestionText");

      // 根據不同分析點分開列點顯示
      if (integratedData.analyses) {
        const analysisItems = [];

        // 定義分析點標題和對應的建議
        const analysisMapping = [
          { key: "backswing_advice", title: "📐 拉拍分析" },
          { key: "forwardswing_advice", title: "⚡ 前段出拍分析" },
          { key: "hitballswing_advice", title: "🎾 擊球出拍轉身分析" },
          { key: "followthrough_advice", title: "🔄 收拍分析" },
          { key: "contact_zone_advice", title: "📍 擊球點區域分析" },
          { key: "head_stability_advice", title: "👁️ 頭部穩定度分析" },
        ];

        analysisMapping.forEach(({ key, title }) => {
          const advice = integratedData.analyses[key];
          if (advice && advice.trim()) {
            const formattedAdvice = formatAdviceText(advice);
            analysisItems.push(
              `<div class="analysis-point"><strong>${title}</strong>${formattedAdvice}</div>`,
            );
          }
        });

        if (analysisItems.length > 0) {
          container.innerHTML = analysisItems.join("<br>");
        } else {
          // 如果沒有 analyses，嘗試使用 combined_advice
          if (integratedData.combined_advice) {
            // 將 combined_advice 按段落分割並格式化
            const paragraphs = integratedData.combined_advice
              .split(/\n\n+/)
              .filter((p) => p.trim());
            container.innerHTML = paragraphs
              .map((p) => {
                const formatted = formatAdviceText(p);
                return `<div class="analysis-point">${formatted}</div>`;
              })
              .join("<br>");
          } else {
            container.textContent = "整合分析結果格式異常";
          }
        }
      } else if (integratedData.combined_advice) {
        // 如果只有 combined_advice，按段落分割並格式化
        const paragraphs = integratedData.combined_advice
          .split(/\n\n+/)
          .filter((p) => p.trim());
        container.innerHTML = paragraphs
          .map((p) => {
            const formatted = formatAdviceText(p);
            return `<div class="analysis-point">${formatted}</div>`;
          })
          .join("<br>");
      } else {
        container.textContent = "整合分析結果格式異常";
      }

      // 解析動作類型：優先使用整合分析產出的 action_type（正拍/反拍），否則從 KNN suggestion 或 nearest_expert 判斷
      const actionTypeEl = document.getElementById("actionTypeValue");
      if (integratedData.action_type) {
        actionTypeEl.textContent = integratedData.action_type;
      } else {
        const knnSuggestion = integratedData.analyses?.knn_suggestion || "";
        if (
          knnSuggestion.includes("正手") ||
          integratedData.nearest_expert?.includes("正手")
        ) {
          actionTypeEl.textContent = "正手擊球";
        } else if (
          knnSuggestion.includes("反手") ||
          integratedData.nearest_expert?.includes("反手")
        ) {
          actionTypeEl.textContent = "反手擊球";
        } else if (
          knnSuggestion.includes("發球") ||
          integratedData.nearest_expert?.includes("發球")
        ) {
          actionTypeEl.textContent = "發球";
        } else {
          actionTypeEl.textContent = "未知動作";
        }
      }

      // 顯示相似度（如果有 expert_distance）
      if (integratedData.expert_distance !== undefined) {
        const similarity = (
          (1 / (1 + integratedData.expert_distance)) *
          100
        ).toFixed(1);
        document.getElementById("similarityValue").textContent =
          `${similarity}%`;
      }
      // Processing Stats：整合分析有資料時先顯示
      if (
        integratedData.statistics &&
        Object.keys(integratedData.statistics).length > 0
      ) {
        const parts = [];
        if (integratedData.statistics.backswing_confidence != null) {
          parts.push(
            `<div><strong>拉拍信心度:</strong> ${(integratedData.statistics.backswing_confidence * 100).toFixed(0)}%</div>`,
          );
        }
        if (integratedData.statistics.head_stability_confidence != null) {
          parts.push(
            `<div><strong>頭部穩定度信心度:</strong> ${(integratedData.statistics.head_stability_confidence * 100).toFixed(0)}%</div>`,
          );
        }
        if (integratedData.analysis_timestamp) {
          parts.push(
            `<div><strong>分析時間:</strong> ${new Date(integratedData.analysis_timestamp).toLocaleString("zh-TW")}</div>`,
          );
        }
        if (parts.length > 0) {
          document.getElementById("statsContent").innerHTML = parts.join("");
        }
      }

      // ===== 將 integrated 資料完整存入 localStorage，供 v3 頁面讀取 =====
      try {
        // 讀取現有 localStorage，保留 CSV 分數欄位（如有）
        let stored = {};
        try { stored = JSON.parse(localStorage.getItem('tennisAnalysisData') || '{}'); } catch(e) { stored = {}; }

        const similarity = integratedData.expert_distance !== undefined
          ? ((1 / (1 + integratedData.expert_distance)) * 100).toFixed(1) + "%"
          : null;

        const stats = integratedData.statistics || {};
        const cz = stats.contact_zone?.flags || {};
        const czOk = [cz.lateral_in_range, cz.height_in_range, cz.depth_in_range].filter(Boolean).length;
        const ContactZone = (czOk / 3) * 10;
        const toSave = Object.assign(stored, {
          id: "ANALYSIS_RESULT",
          hand: "right",
          video: 1,
          Head: (stats.head_stability_confidence || 0) * 10,
          Backswing: (stats.backswing_confidence || 0) * 10,
          ForwardSwing: (stats.forwardswing_confidence || 0) * 10,
          HitballSwing: (stats.hitballswing_confidence || 0) * 10,
          FollowThrough: (stats.followthrough_confidence || 0) * 10,
          ContactZone: ContactZone,
          BodyWeight: 0,
          HitTiming: 0,
          RacketFace: 0,
          overall: 0,
          action_type: integratedData.action_type || null,
          similarity: similarity,
          nearest_expert: integratedData.nearest_expert || null,
          analyses: integratedData.analyses || null,
          statistics: integratedData.statistics || null,
          combined_advice: integratedData.combined_advice || null,
          advice: integratedData.combined_advice || null,
          priority_improvement: integratedData.priority_improvement || null,
          video_url: videoSrc || null,
        });

        localStorage.setItem('tennisAnalysisData', JSON.stringify(toSave));
        console.log('[localStorage] 已將 integrated_analysis 存入 tennisAnalysisData');
      } catch(e) {
        console.warn('將分析資料存入 localStorage 失敗:', e);
      }
    } else {
      // Fallback: 嘗試載入原本的 KNN feedback
      const knnPath = `./trajectory/${folderName}/${seg}${prefix}_knn_feedback.txt`;
      const knnResponse = await fetch(knnPath);
      if (knnResponse.ok) {
        const knnText = await knnResponse.text();
        document.getElementById("knnSuggestionText").innerHTML =
          `<div class="analysis-point">${knnText}</div>`;

        // 解析動作類型
        if (knnText.includes("正手")) {
          document.getElementById("actionTypeValue").textContent = "正手擊球";
        } else if (knnText.includes("反手")) {
          document.getElementById("actionTypeValue").textContent = "反手擊球";
        } else if (knnText.includes("發球")) {
          document.getElementById("actionTypeValue").textContent = "發球";
        } else {
          document.getElementById("actionTypeValue").textContent = "未知動作";
        }
      } else {
        document.getElementById("knnSuggestionText").textContent =
          "整合分析結果未找到";
      }
    }
  } catch (error) {
    console.error("Failed to load integrated analysis:", error);
    document.getElementById("knnSuggestionText").textContent =
      "載入整合分析失敗";
  }
}

// 載入 GPT 分析結果
async function loadGPTAnalysis(folderName, fileName, prefix) {
  try {
    const seg = fileName ? `${fileName}/` : "";
    const prefixForAnalysis = analysisPrefix(prefix);
    let gptPath = `./trajectory/${folderName}/${seg}${prefix}_gpt_feedback.json`;
    let response = await fetch(gptPath);
    if (!response.ok && prefixForAnalysis !== prefix) {
      gptPath = `./trajectory/${folderName}/${seg}${prefixForAnalysis}_gpt_feedback.json`;
      response = await fetch(gptPath);
    }
    console.log("Loading GPT feedback from:", gptPath);

    if (response.ok) {
      const gptData = await response.json();
      console.log("GPT data loaded:", gptData);

      // 顯示 GPT 反饋
      if (gptData.suggestion) {
        document.getElementById("gptFeedbackText").textContent =
          gptData.suggestion;
      } else if (gptData.problem_frame) {
        document.getElementById("gptFeedbackText").textContent =
          "GPT 分析結果格式異常（缺少 suggestion 欄位）";
      } else {
        document.getElementById("gptFeedbackText").textContent =
          "GPT 分析結果格式異常";
      }
      // Processing Stats：問題幀、分析時間
      if (
        gptData.problem_frame ||
        gptData.analysis_timestamp ||
        gptData.suggestion
      ) {
        const statsParts = [];
        if (gptData.problem_frame) {
          statsParts.push(
            `<div><strong>問題幀範圍:</strong> ${gptData.problem_frame}</div>`,
          );
        }
        if (gptData.analysis_timestamp) {
          statsParts.push(
            `<div><strong>分析時間:</strong> ${new Date(gptData.analysis_timestamp).toLocaleString("zh-TW")}</div>`,
          );
        } else if (gptData.suggestion) {
          statsParts.push(
            `<div><strong>分析時間:</strong> ${new Date().toLocaleString("zh-TW")}</div>`,
          );
        }
        if (statsParts.length > 0) {
          const statsEl = document.getElementById("statsContent");
          const existing = statsEl.innerHTML.trim();
          statsEl.innerHTML =
            existing && !existing.startsWith("-")
              ? existing + "<br>" + statsParts.join("")
              : statsParts.join("");
        }
      }
    } else {
      console.error("GPT feedback file not found, status:", response.status);
      document.getElementById("gptFeedbackText").textContent =
        "GPT 分析結果未找到";
    }
  } catch (error) {
    console.error("Failed to load GPT analysis:", error);
    document.getElementById("gptFeedbackText").textContent =
      "載入 GPT 分析失敗: " + error.message;
  }
}

function createChart(canvasId, data) {
  console.log("createChart executed", canvasId, data);
  const points = data
    .map((frame) => ({
      x: frame.right_wrist?.x,
      y: frame.right_wrist?.y,
      frame: frame.frame,
    }))
    .filter((point) => point.x != null && point.y != null);
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (charts[canvasId]) {
    charts[canvasId].destroy();
  }
  charts[canvasId] = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Wrist Trajectory",
          data: points,
          borderColor: "#4DC4C0",
          backgroundColor: "rgba(77, 196, 192, 0.5)",
          showLine: true,
          pointRadius: 3,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: { display: true, text: "X Position" },
          grid: { color: "#E5E5E5" },
        },
        y: {
          type: "linear",
          reverse: true,
          title: { display: true, text: "Y Position" },
          grid: { color: "#E5E5E5" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              return `Frame: ${context.raw.frame}, X: ${context.raw.x.toFixed(2)}, Y: ${context.raw.y.toFixed(2)}`;
            },
          },
        },
      },
    },
  });
}

let charts = {
  chart1: null,
  chart2: null,
};

// 開啟 3D 視覺化並傳遞選中的資料夾與軌跡
function open3DVisualization() {
  const selectedFolder = document.getElementById("folderSelect").value;
  const idx = parseInt(document.getElementById("videoSelect").value, 10);
  const opt = currentTrajectoryOptions[idx];

  if (!selectedFolder) {
    alert("請先選擇一個資料夾！");
    return;
  }

  let url = "/3d";
  const params = new URLSearchParams();
  params.append("folder", selectedFolder);
  if (opt && (opt.prefix || opt.label)) {
    params.append("video", opt.prefix || opt.label);
  }
  if (params.toString()) url += "?" + params.toString();
  window.open(url, "_blank");
}
