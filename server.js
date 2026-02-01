const express = require("express");
const path = require("path");
const fs = require("fs");

const app = express();
const port = parseInt(process.env.PORT || "3001", 10);

app.use(express.static(path.join(__dirname)));

app.listen(port, () => {
  console.log(`伺服器運行於 http://localhost:${port}`);
  console.log(`  2D 前端: http://localhost:${port}/ 或 http://localhost:${port}/drawing_2D_chart_js.html`);
  console.log(`  3D 前端: http://localhost:${port}/3d`);
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "drawing_2D_chart_js.html"));
});

app.get("/3d", (req, res) => {
  res.sendFile(path.join(__dirname, "drawing_3D_three_js.html"));
});

app.get("/getFolders", (req, res) => {
  const assetsDir = path.join(__dirname, "trajectory");
  fs.readdir(assetsDir, (err, files) => {
    if (err) {
      // console.error("讀取資料夾失敗：", err);
      return res.status(500).json({ error: "無法讀取資料夾" });
    }
    // 只保留目錄（資料夾）
    const folders = files.filter((file) => {
      const filePath = path.join(assetsDir, file);
      return fs.statSync(filePath).isDirectory();
    });
    res.json(folders);
  });
});

app.get("/getVideos", (req, res) => {
  const folder = req.query.folder;
  if (!folder) {
    return res.status(400).json({ error: "請提供 folder 參數" });
  }
  const videoDir = path.join(__dirname, "trajectory", folder);
  fs.readdir(videoDir, (err, files) => {
    if (err) {
      return res.status(500).json({ error: `無法讀取 ${folder} 資料夾` });
    }
    const mp4Files = files.filter((file) =>
      file.toLowerCase().endsWith(".mp4"),
    );
    res.json(mp4Files);
  });
});

/** 列出某資料夾內可選的「軌跡」選項，支援兩種結構：
 *  1) 巢狀：trajectory_1, trajectory_2, ... 子資料夾，內有 *_(45|side)*(2D_trajectory_smoothed).json
 *  2) 扁平：資料夾內直接有 *_(45|side)*(2D_trajectory_smoothed).json
 * 回傳 [{ label, folderName, fileName, prefix, videoPath?, json45, jsonSide }]
 */
app.get("/getTrajectoryOptions", (req, res) => {
  const folder = req.query.folder;
  if (!folder) {
    return res.status(400).json({ error: "請提供 folder 參數" });
  }
  const baseDir = path.join(__dirname, "trajectory", folder);
  if (!fs.existsSync(baseDir) || !fs.statSync(baseDir).isDirectory()) {
    return res.status(404).json({ error: "資料夾不存在" });
  }

  const options = [];
  const smoothed = (f) =>
    /(2D_trajectory_smoothed)\.json$/i.test(f) ||
    /_segment\(2D_trajectory_smoothed\)\.json$/i.test(f);

  function findPrefixAndPair(files) {
    const j45 = files.find(
      (f) =>
        smoothed(f) &&
        (f.includes("_45") || f.includes("45(")) &&
        !f.includes("_side"),
    );
    const jSide = files.find(
      (f) =>
        smoothed(f) &&
        (f.includes("_side") || (f.includes("side") && !f.includes("45"))),
    );
    if (!j45 && !jSide) return null;
    const base = (j45 || jSide)
      .replace(/_45.*$|_side.*$|\(2D.*$|_segment.*$/i, "")
      .replace(/_+$/, "");
    return {
      json45: j45 || null,
      jsonSide: jSide || null,
      prefix: base,
    };
  }

  /** 優先使用 _full_video.mp4，其次 _processed_full_video.mp4 */
  function pickMp4(files, prefix, json45FileName) {
    if (!prefix) {
      const full = files.find((f) => f.toLowerCase().endsWith("_full_video.mp4"));
      if (full) return full;
      return files.find((f) => f.toLowerCase().endsWith("_processed_full_video.mp4")) || null;
    }

    const cleanPrefix = prefix.replace(/_segment$/, "");
    const lowerPrefix = cleanPrefix.toLowerCase();

    let angleType = null;
    if (json45FileName) {
      const jsonLower = json45FileName.toLowerCase();
      if (jsonLower.includes("_45") || jsonLower.includes("45(")) angleType = "_45";
      else if (jsonLower.includes("_side") || jsonLower.includes("side")) angleType = "_side";
    }

    // 優先 _full_video.mp4（用戶指定使用此檔）
    const fullVideoMatch = files.find((f) => {
      const lf = f.toLowerCase();
      if (!lf.endsWith("_full_video.mp4")) return false;
      if (!lf.includes(lowerPrefix)) return false;
      if (angleType && !lf.includes(angleType.toLowerCase())) return false;
      return true;
    });
    if (fullVideoMatch) return fullVideoMatch;

    const fullVideoFallback = files.find((f) => {
      const lf = f.toLowerCase();
      return lf.endsWith("_full_video.mp4") && lf.includes(lowerPrefix);
    });
    if (fullVideoFallback) return fullVideoFallback;

    // 其次 _processed_full_video.mp4
    const processedMatch = files.find((f) => {
      const lf = f.toLowerCase();
      return lf.endsWith("_processed_full_video.mp4") && lf.includes(lowerPrefix);
    });
    if (processedMatch) return processedMatch;

    const v = files.find(
      (f) =>
        f.toLowerCase().endsWith(".mp4") &&
        (f.includes(prefix) || f.toLowerCase().includes("full_video") || f.includes(cleanPrefix)),
    );
    return v || files.find((f) => f.toLowerCase().endsWith(".mp4")) || null;
  }

  const topFiles = fs.readdirSync(baseDir);
  const subdirs = topFiles
    .filter((f) => {
      const p = path.join(baseDir, f);
      return fs.statSync(p).isDirectory() && /^trajectory_\d+$/.test(f);
    })
    .sort((a, b) => {
      const n = (x) => parseInt(x.replace("trajectory_", ""), 10);
      return n(a) - n(b);
    });

  if (subdirs.length > 0) {
    for (const sub of subdirs) {
      const subPath = path.join(baseDir, sub);
      const files = fs.readdirSync(subPath);
      const pair = findPrefixAndPair(files);
      if (!pair) continue;
      // 傳遞 json45 檔名給 pickMp4，以便正確匹配角度（_45 或 _side）
      const videoPath = pickMp4(files, pair.prefix, pair.json45);
      options.push({
        label: pair.prefix || sub,
        folderName: folder,
        fileName: sub,
        prefix: pair.prefix,
        videoPath: videoPath
          ? `trajectory/${folder}/${sub}/${videoPath}`
          : null,
        json45: pair.json45
          ? `trajectory/${folder}/${sub}/${pair.json45}`
          : null,
        jsonSide: pair.jsonSide
          ? `trajectory/${folder}/${sub}/${pair.jsonSide}`
          : null,
      });
    }
  } else {
    const pair = findPrefixAndPair(topFiles);
    if (pair) {
      // 傳遞 json45 檔名給 pickMp4，以便正確匹配角度（_45 或 _side）
      const videoPath = pickMp4(topFiles, pair.prefix, pair.json45);
      options.push({
        label: pair.prefix,
        folderName: folder,
        fileName: "",
        prefix: pair.prefix,
        videoPath: videoPath
          ? `trajectory/${folder}/${videoPath}`
          : null,
        json45: pair.json45
          ? `trajectory/${folder}/${pair.json45}`
          : null,
        jsonSide: pair.jsonSide
          ? `trajectory/${folder}/${pair.jsonSide}`
          : null,
      });
    }
    if (options.length === 0) {
      const json45List = topFiles.filter(
        (f) =>
          smoothed(f) &&
          (f.includes("_45") || f.includes("45(")) &&
          !f.includes("_side"),
      );
      for (const j45 of json45List) {
        const base = j45
          .replace(/_45.*$|\(2D.*$|_segment.*$/i, "")
          .replace(/_+$/, "");
        const jSide = topFiles.find(
          (f) =>
            smoothed(f) &&
            (f.includes("_side") || f.includes("side")) &&
            !f.includes("45") &&
            (f.startsWith(base) || f.includes(base)),
        );
        // 傳遞 json45 檔名給 pickMp4，以便正確匹配角度（_45 或 _side）
        const v = pickMp4(topFiles, base, j45);
        options.push({
          label: base || j45,
          folderName: folder,
          fileName: "",
          prefix: base,
          videoPath: v ? `trajectory/${folder}/${v}` : null,
          json45: `trajectory/${folder}/${j45}`,
          jsonSide: jSide ? `trajectory/${folder}/${jSide}` : null,
        });
      }
    }
  }

  res.json(options);
});

app.get("/getjson", (req, res) => {
  const dir = req.query.dir;
  if (!dir) {
    return res.status(400).json({ error: "缺少 dir 參數" });
  }

  const directoryPath = path.join(__dirname, dir);

  fs.readdir(directoryPath, (err, files) => {
    if (err) {
      return res
        .status(500)
        .json({ error: "無法讀取目錄", details: err.message });
    }

    // 僅回傳 .json 檔案
    const jsonFiles = files.filter((file) => file.endsWith(".json"));
    res.json(jsonFiles);
  });
});

// 獲取分析結果的 API 端點
app.get("/getAnalysisResults", (req, res) => {
  const { folder, fileName, prefix } = req.query;

  if (!folder || !fileName || !prefix) {
    return res
      .status(400)
      .json({ error: "請提供 folder, fileName, 和 prefix 參數" });
  }

  const trajectoryDir = path.join(__dirname, "trajectory", folder, fileName);

  // 檢查分析結果檔案是否存在
  const knnFile = path.join(trajectoryDir, `${prefix}_knn_feedback.txt`);
  const gptFile = path.join(trajectoryDir, `${prefix}_gpt_feedback.json`);

  const results = {
    knn: null,
    gpt: null,
    files: {
      knn_exists: fs.existsSync(knnFile),
      gpt_exists: fs.existsSync(gptFile),
    },
  };

  // 讀取 KNN 分析結果
  if (results.files.knn_exists) {
    try {
      results.knn = fs.readFileSync(knnFile, "utf8");
    } catch (error) {
      console.error("Error reading KNN file:", error);
    }
  }

  // 讀取 GPT 分析結果
  if (results.files.gpt_exists) {
    try {
      const gptData = fs.readFileSync(gptFile, "utf8");
      results.gpt = JSON.parse(gptData);
    } catch (error) {
      console.error("Error reading GPT file:", error);
    }
  }

  res.json(results);
});
