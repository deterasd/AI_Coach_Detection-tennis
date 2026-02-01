#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接指定 3D 軌跡 JSON 路徑，只執行 trajector_processing 步驟 10～11：
  - 步驟 10：KNN 分析
  - 步驟 11：GPT 反饋生成
（含中間的整合分析）

所有輸出（KNN 建議、整合分析、GPT 反饋）一律寫入**該檔案所在資料夾**。
不需從步驟 1 重跑，節省時間。

使用方式:
  1. 在腳本開頭填寫 DEFAULT_TRAJECTORY_PATH、DEFAULT_KNN_DATASET，然後直接執行：
     python3 analyze_from_file.py

  2. 或從命令列傳入路徑（會覆蓋預設值）：
     python3 analyze_from_file.py "trajectory/.../X(3D_trajectory_smoothed).json"
     python3 analyze_from_file.py "path/to/file.json" --knn-dataset knn_dataset_new1.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from trajectory_knn import analyze_trajectory as analyze_trajectory_knn
from trajectory_integrated_analysis import analyze_integrated_trajectory
from trajectory_gpt_single_feedback import generate_feedback


# -----------------------------------------------------------------------------
# 在這裡填寫預設路徑，直接執行腳本時會使用（不帶命令列參數時）
# -----------------------------------------------------------------------------
DEFAULT_TRAJECTORY_PATH = "trajectory/Cindy__trajectory/player6_1_2/outdoor6__1_45_segment(3D_trajectory_smoothed).json"
DEFAULT_KNN_DATASET = "knn_dataset_new.json"
# -----------------------------------------------------------------------------

SUFFIX_3D = "(3D_trajectory_smoothed).json"
SUFFIX_ONLY_SWING = "(3D_trajectory_smoothed)_only_swing.json"


def _base_and_dir(trajectory_path: str | Path) -> tuple[str, Path]:
    """從 3D 軌跡路徑取得目錄與 base（檔名去掉 (3D_trajectory_smoothed).json）。"""
    p = Path(trajectory_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"找不到檔案: {p}")
    if not p.name.endswith(SUFFIX_3D):
        raise ValueError(
            f"請指定 *{SUFFIX_3D} 的 3D 軌跡檔，當前為: {p.name}"
        )
    base = p.name[: -len(SUFFIX_3D)]  # e.g. 測試者2__1_45
    return base, p.parent


def run_analysis(
    trajectory_path: str | Path,
    knn_dataset_path: str = "knn_dataset_new.json",
    n_neighbors: int = 3,
) -> bool:
    """
    對指定 3D 軌跡執行 KNN → 整合分析 → GPT 反饋，輸出全寫入該檔所在資料夾。
    """
    trajectory_path = Path(trajectory_path).resolve()
    base, out_dir = _base_and_dir(trajectory_path)
    knn_path = out_dir / f"{base}_knn_feedback.txt"
    only_swing_path = out_dir / f"{base}{SUFFIX_ONLY_SWING}"
    gpt_path = out_dir / f"{base}_gpt_feedback.json"

    timing = {}
    t0 = time.perf_counter()

    # ----- 步驟 10: KNN -----
    print("\n[步驟 10] KNN 分析...")
    t = time.perf_counter()
    try:
        results, nearest_expert = analyze_trajectory_knn(
            knn_dataset_path,
            str(trajectory_path),
            n_neighbors=n_neighbors,
        )
        text = results[0] if results else ""
        with open(knn_path, "w", encoding="utf-8") as f:
            f.write(text)
        timing["KNN 分析"] = time.perf_counter() - t
        print(f"  完成，耗時 {timing['KNN 分析']:.2f}s | 最相似專家: {nearest_expert}")
        print(f"  輸出: {knn_path}")
    except Exception as e:
        print(f"  KNN 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ----- 整合分析 -----
    print("\n[整合分析] ...")
    t = time.perf_counter()
    try:
        actual_integrated = analyze_integrated_trajectory(
            str(trajectory_path),
            knn_dataset_path,
            nearest_expert,
            str(knn_path),
        )
        timing["整合分析"] = time.perf_counter() - t
        print(f"  完成，耗時 {timing['整合分析']:.2f}s")
        print(f"  輸出: {actual_integrated}")
    except Exception as e:
        print(f"  整合分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ----- 步驟 11: GPT 反饋 -----
    print("\n[步驟 11] GPT 反饋生成...")
    json_for_gpt = str(only_swing_path) if only_swing_path.is_file() else str(trajectory_path)
    t = time.perf_counter()
    try:
        out_gpt = generate_feedback(
            json_for_gpt,
            str(knn_path),
            integrated_analysis_path=actual_integrated,
            output_path=str(gpt_path),
        )
        timing["GPT 反饋"] = time.perf_counter() - t
        print(f"  完成，耗時 {timing['GPT 反饋']:.2f}s")
        print(f"  輸出: {out_gpt}")
    except Exception as e:
        print(f"  GPT 反饋失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    total = time.perf_counter() - t0
    print("\n" + "=" * 50)
    print("執行時間摘要")
    print("=" * 50)
    for k, v in timing.items():
        print(f"  {k}: {v:.2f}s")
    print(f"  總計: {total:.2f}s")
    print("=" * 50)
    print("所有輸出已寫入該檔案所在資料夾。")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="對指定 3D 軌跡只做 KNN + 整合分析 + GPT 反饋，輸出寫入同資料夾。",
    )
    ap.add_argument(
        "trajectory",
        type=str,
        nargs="?",
        default=None,
        help="3D 軌跡 JSON 路徑；若不填則使用腳本內 DEFAULT_TRAJECTORY_PATH",
    )
    ap.add_argument(
        "--knn-dataset",
        type=str,
        default=None,
        help="KNN 資料集路徑；若不填則使用腳本內 DEFAULT_KNN_DATASET",
    )
    ap.add_argument(
        "-k", "--n-neighbors",
        type=int,
        default=3,
        help="KNN 鄰居數（預設: 3）",
    )
    args = ap.parse_args()

    trajectory = args.trajectory or DEFAULT_TRAJECTORY_PATH
    knn_dataset = args.knn_dataset or DEFAULT_KNN_DATASET

    if not Path(knn_dataset).exists():
        print(f"錯誤: 找不到 KNN 資料集 {knn_dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"使用 3D 軌跡: {trajectory}")
    print(f"使用 KNN 資料集: {knn_dataset}\n")

    ok = run_analysis(
        trajectory,
        knn_dataset_path=knn_dataset,
        n_neighbors=args.n_neighbors,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
