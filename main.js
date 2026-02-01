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

// ---Video to Json---------------------------------------------
const folderSelect = document.getElementById("folderSelect");
const videoSelect = document.getElementById("videoSelect");
const basePath = "./trajectory/";

let currentTrajectoryOptions = [];

async function fetchFolderList() {
  try {
    const response = await fetch("/getFolders");
    const folders = await response.json();
    folderSelect.innerHTML =
      '<option value="">— 選擇資料夾 —</option>' +
      folders
        .map(
          (f) =>
            `<option value="${f.replace(/"/g, "&quot;")}">${f}</option>`,
        )
        .join("");
  } catch (error) {
    console.error("Unable to fetch folder list:", error);
  }
}

async function fetchTrajectoryOptions(folder) {
  videoSelect.innerHTML = '<option value="">— 選擇影片 —</option>';
  currentTrajectoryOptions = [];
  if (!folder) return;
  try {
    const response = await fetch(
      `/getTrajectoryOptions?folder=${encodeURIComponent(folder)}`,
    );
    if (!response.ok) {
      console.warn("getTrajectoryOptions failed:", response.status);
      return;
    }
    const options = await response.json();
    currentTrajectoryOptions = options;
    options.forEach((opt, i) => {
      const label = opt.label + (opt.videoPath ? " (含影片)" : "");
      videoSelect.innerHTML += `<option value="${i}">${label}</option>`;
    });
  } catch (error) {
    console.error("fetchTrajectoryOptions:", error);
  }
}

folderSelect.addEventListener("change", (e) => {
  const folder = e.target.value;
  if (folder) fetchTrajectoryOptions(folder);
  else {
    videoSelect.innerHTML = '<option value="">— 選擇影片 —</option>';
    currentTrajectoryOptions = [];
  }
});

document.addEventListener("DOMContentLoaded", fetchFolderList);

videoSelect.addEventListener("change", (e) => {
  const idx = parseInt(e.target.value, 10);
  const opt = currentTrajectoryOptions[idx];
  if (!opt) return;

  if (opt.videoPath) {
    videoPlayer.src = opt.videoPath;
    videoPlayer.play().catch(() => {});
  } else {
    videoPlayer.removeAttribute("src");
    videoPlayer.load();
  }

  const json45 =
    opt.json45 ||
    (opt.fileName
      ? `${basePath}${opt.folderName}/${opt.fileName}/${opt.prefix}_45(2D_trajectory_smoothed).json`
      : `${basePath}${opt.folderName}/${opt.prefix}_45(2D_trajectory_smoothed).json`);
  const jsonSide =
    opt.jsonSide ||
    (opt.fileName
      ? `${basePath}${opt.folderName}/${opt.fileName}/${opt.prefix}_side(2D_trajectory_smoothed).json`
      : `${basePath}${opt.folderName}/${opt.prefix}_side(2D_trajectory_smoothed).json`);

  handleFileSelection(json45, jsonSide);
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
    await loadAnalysisResults(filePath45, filePathSide);
  } catch (error) {
    console.error("Failed to load Side JSON file:", error);
  }
}

// 載入分析結果
async function loadAnalysisResults(filePath45, filePathSide) {
  try {
    // 優先使用 45 度角檔案來提取 prefix，因為分析檔案通常基於 45 度角檔案名稱
    const pathParts = filePath45.replace(/^\.\//, "").split("/").filter(Boolean);
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
      const sidePathParts = filePathSide.replace(/^\.\//, "").split("/").filter(Boolean);
      if (sidePathParts.length >= 3 && sidePathParts[0] === "trajectory") {
        const sideJsonFile = sidePathParts.length === 3 ? sidePathParts[2] : sidePathParts[3];
        prefix = sideJsonFile
          .replace(/_side\(2D_trajectory_smoothed\)\.json$/i, "")
          .replace(/_side_segment\(2D_trajectory_smoothed\)\.json$/i, "");
      }
    }

    console.log("Loading analysis results for prefix:", prefix);
    console.log("Folder:", folderName, "File:", fileName);

    await loadKNNAnalysis(folderName, fileName, prefix);
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
async function loadKNNAnalysis(folderName, fileName, prefix) {
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
          { key: "head_stability_advice", title: "👁️ 頭部穩定度分析" }
        ];
        
        analysisMapping.forEach(({ key, title }) => {
          const advice = integratedData.analyses[key];
          if (advice && advice.trim()) {
            const formattedAdvice = formatAdviceText(advice);
            analysisItems.push(`<div class="analysis-point"><strong>${title}</strong>${formattedAdvice}</div>`);
          }
        });
        
        if (analysisItems.length > 0) {
          container.innerHTML = analysisItems.join("<br>");
        } else {
          // 如果沒有 analyses，嘗試使用 combined_advice
          if (integratedData.combined_advice) {
            // 將 combined_advice 按段落分割並格式化
            const paragraphs = integratedData.combined_advice.split(/\n\n+/).filter(p => p.trim());
            container.innerHTML = paragraphs.map(p => {
              const formatted = formatAdviceText(p);
              return `<div class="analysis-point">${formatted}</div>`;
            }).join("<br>");
          } else {
            container.textContent = "整合分析結果格式異常";
          }
        }
      } else if (integratedData.combined_advice) {
        // 如果只有 combined_advice，按段落分割並格式化
        const paragraphs = integratedData.combined_advice.split(/\n\n+/).filter(p => p.trim());
        container.innerHTML = paragraphs.map(p => {
          const formatted = formatAdviceText(p);
          return `<div class="analysis-point">${formatted}</div>`;
        }).join("<br>");
      } else {
        container.textContent = "整合分析結果格式異常";
      }

      // 解析動作類型：優先使用整合分析產出的 action_type（正拍/反拍），否則從 KNN suggestion 或 nearest_expert 判斷
      const actionTypeEl = document.getElementById("actionTypeValue");
      if (integratedData.action_type) {
        actionTypeEl.textContent = integratedData.action_type;
      } else {
        const knnSuggestion = integratedData.analyses?.knn_suggestion || "";
        if (knnSuggestion.includes("正手") || integratedData.nearest_expert?.includes("正手")) {
          actionTypeEl.textContent = "正手擊球";
        } else if (knnSuggestion.includes("反手") || integratedData.nearest_expert?.includes("反手")) {
          actionTypeEl.textContent = "反手擊球";
        } else if (knnSuggestion.includes("發球") || integratedData.nearest_expert?.includes("發球")) {
          actionTypeEl.textContent = "發球";
        } else {
          actionTypeEl.textContent = "未知動作";
        }
      }

      // 顯示相似度（如果有 expert_distance）
      if (integratedData.expert_distance !== undefined) {
        const similarity = (1 / (1 + integratedData.expert_distance) * 100).toFixed(1);
        document.getElementById("similarityValue").textContent = `${similarity}%`;
      }
      // Processing Stats：整合分析有資料時先顯示
      if (integratedData.statistics && Object.keys(integratedData.statistics).length > 0) {
        const parts = [];
        if (integratedData.statistics.backswing_confidence != null) {
          parts.push(`<div><strong>拉拍信心度:</strong> ${(integratedData.statistics.backswing_confidence * 100).toFixed(0)}%</div>`);
        }
        if (integratedData.statistics.head_stability_confidence != null) {
          parts.push(`<div><strong>頭部穩定度信心度:</strong> ${(integratedData.statistics.head_stability_confidence * 100).toFixed(0)}%</div>`);
        }
        if (integratedData.analysis_timestamp) {
          parts.push(`<div><strong>分析時間:</strong> ${new Date(integratedData.analysis_timestamp).toLocaleString("zh-TW")}</div>`);
        }
        if (parts.length > 0) {
          document.getElementById("statsContent").innerHTML = parts.join("");
        }
      }
    } else {
      // Fallback: 嘗試載入原本的 KNN feedback
      const knnPath = `./trajectory/${folderName}/${seg}${prefix}_knn_feedback.txt`;
      const knnResponse = await fetch(knnPath);
      if (knnResponse.ok) {
        const knnText = await knnResponse.text();
        document.getElementById("knnSuggestionText").innerHTML = `<div class="analysis-point">${knnText}</div>`;
        
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
        document.getElementById("knnSuggestionText").textContent = "整合分析結果未找到";
      }
    }
  } catch (error) {
    console.error("Failed to load integrated analysis:", error);
    document.getElementById("knnSuggestionText").textContent = "載入整合分析失敗";
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
        document.getElementById("gptFeedbackText").textContent = gptData.suggestion;
      } else if (gptData.problem_frame) {
        document.getElementById("gptFeedbackText").textContent = "GPT 分析結果格式異常（缺少 suggestion 欄位）";
      } else {
        document.getElementById("gptFeedbackText").textContent = "GPT 分析結果格式異常";
      }
      // Processing Stats：問題幀、分析時間
      if (gptData.problem_frame || gptData.analysis_timestamp || gptData.suggestion) {
        const statsParts = [];
        if (gptData.problem_frame) {
          statsParts.push(`<div><strong>問題幀範圍:</strong> ${gptData.problem_frame}</div>`);
        }
        if (gptData.analysis_timestamp) {
          statsParts.push(`<div><strong>分析時間:</strong> ${new Date(gptData.analysis_timestamp).toLocaleString("zh-TW")}</div>`);
        } else if (gptData.suggestion) {
          statsParts.push(`<div><strong>分析時間:</strong> ${new Date().toLocaleString("zh-TW")}</div>`);
        }
        if (statsParts.length > 0) {
          const statsEl = document.getElementById("statsContent");
          const existing = statsEl.innerHTML.trim();
          statsEl.innerHTML = existing && !existing.startsWith("-") ? existing + "<br>" + statsParts.join("") : statsParts.join("");
        }
      }
    } else {
      console.error("GPT feedback file not found, status:", response.status);
      document.getElementById("gptFeedbackText").textContent = "GPT 分析結果未找到";
    }
  } catch (error) {
    console.error("Failed to load GPT analysis:", error);
    document.getElementById("gptFeedbackText").textContent = "載入 GPT 分析失敗: " + error.message;
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
