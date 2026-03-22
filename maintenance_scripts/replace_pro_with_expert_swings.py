#!/usr/bin/env python3
"""
將 knn_dataset_new.json 中所有 level=="pro" 的資料刪除，
並替換成「專家揮拍紀錄」資料夾中的 19 個揮拍紀錄，格式與現有 dataset 一致。
"""
import json
from pathlib import Path

KNN_DATASET_PATH = Path(__file__).resolve().parent.parent / "knn_dataset_new.json"
EXPERT_FOLDER = Path(__file__).resolve().parent.parent / "專家揮拍紀錄"


def main():
    # 載入現有 dataset
    with open(KNN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    original_count = len(dataset)
    pro_count_before = sum(1 for e in dataset if e.get("level") == "pro")

    # 刪除所有 pro 資料
    dataset = [e for e in dataset if e.get("level") != "pro"]
    non_pro_count = len(dataset)

    # 找出專家揮拍紀錄中的 19 個 3D trajectory 檔（排除 _only_swing）
    expert_pattern = "*segment(3D_trajectory_smoothed).json"
    expert_files = []
    for f in EXPERT_FOLDER.rglob(expert_pattern):
        if "_only_swing" in f.name:
            continue
        expert_files.append(f)

    expert_files.sort(key=lambda p: (p.parent.name, p.name))

    if len(expert_files) != 19:
        print(f"⚠ 預期 19 個專家揮拍檔，實際找到 {len(expert_files)} 個")
        for pf in expert_files:
            print(f"   - {pf.relative_to(EXPERT_FOLDER)}")
    else:
        print(f"✓ 找到 19 個專家揮拍紀錄")

    # 載入並轉成 dataset 格式
    new_pro_entries = []
    for fp in expert_files:
        with open(fp, "r", encoding="utf-8") as f:
            frames = json.load(f)
        # 使用檔名（不含路徑）作為 filename，與現有格式一致
        entry = {
            "filename": fp.name,
            "level": "pro",
            "suggestion": "",
            "data": frames,
        }
        new_pro_entries.append(entry)
        print(f"  + {fp.name} ({len(frames)} frames)")

    dataset.extend(new_pro_entries)

    # 寫回
    with open(KNN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\n完成:")
    print(f"  原本總筆數: {original_count}")
    print(f"  刪除 pro 筆數: {pro_count_before}")
    print(f"  新增 pro 筆數: {len(new_pro_entries)}")
    print(f"  最終總筆數: {len(dataset)}")
    print(f"  已寫入: {KNN_DATASET_PATH}")


if __name__ == "__main__":
    main()
