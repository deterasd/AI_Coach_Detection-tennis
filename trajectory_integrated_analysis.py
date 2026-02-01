"""
整合分析模組 - 統一管理所有分析點
將所有分析點的結果整合到一個 JSON 檔案中
"""

import json
import numpy as np
from datetime import datetime
from trajectory_analysis.trajectory_center_of_mass_knn import analyze_center_of_mass
from trajectory_analysis.trajectory_backswing_knn import analyze_backswing
from trajectory_analysis.trajectory_forwardswing_knn import analyze_forwardswing
from trajectory_analysis.trajectory_hitballswing_knn import analyze_hitballswing
from trajectory_analysis.trajectory_followthrough_knn import analyze_followthrough
from trajectory_analysis.trajectory_contact_zone_eval import analyze_contact_zone
from trajectory_analysis.trajectory_head_stability import analyze_head_stability


def load_json(file_path):
    """載入 JSON 檔案"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def combine_all_advice(analyses_dict):
    """
    合併所有分析點的建議
    參數: analyses_dict - 包含所有分析點建議的字典
    返回: 合併後的建議字串
    """
    combined_parts = []
    
    # 按順序合併各個分析點的建議（不含 knn_suggestion）
    analysis_order = [
        "backswing_advice",
        "forwardswing_advice",
        "hitballswing_advice",
        "followthrough_advice",
        "contact_zone_advice",
    ]
    
    for analysis_key in analysis_order:
        if analysis_key in analyses_dict and analyses_dict[analysis_key]:
            combined_parts.append(analyses_dict[analysis_key])
    
    return "\n\n".join(combined_parts)


def analyze_all_features(trajectory_data, knn_dataset, expert_filename, knn_suggestion=None):
    """
    執行所有分析點的整合分析
    參數:
    - trajectory_data: 3D 軌跡資料路徑
    - knn_dataset: KNN 資料庫路徑
    - expert_filename: 最相近的專家檔案名稱
    返回: 整合分析結果字典
    """
    if expert_filename is None:
        expert_filename = "unknown"
    print(f"開始整合分析，使用專家: {expert_filename}")
    
    # 初始化整合結果結構
    result = {
        "analysis_timestamp": datetime.now().isoformat(),
        "trajectory_file": trajectory_data,
        "nearest_expert": expert_filename,
        "expert_distance": 0.0,
        "analyses": {},
        "combined_advice": "",
        "statistics": {}
    }
    
    # 0. KNN 建議不再寫入 analyses（依需求省略）
    # 1. 重心分析（暫停執行與輸出）
    # print("執行重心分析...")
    # 暫停：不執行 analyze_center_of_mass，也不輸出 center_of_mass_advice 與統計
    # try:
    #     center_of_mass_data = analyze_center_of_mass(
    #         knn_dataset, trajectory_data, expert_filename
    #     )
    #     result["analyses"]["center_of_mass_advice"] = center_of_mass_data.get("height_advice", "")
    #     result["expert_distance"] = center_of_mass_data.get("distance", 0.0)
    #     result["statistics"]["center_of_mass"] = center_of_mass_data.get("statistics", {})
    #     print(f"重心分析完成: {center_of_mass_data.get('height_assessment', '未知')}")
    # except Exception as e:
    #     print(f"重心分析失敗: {e}")
    #     result["analyses"]["center_of_mass_advice"] = "重心分析失敗"
    
    # 2. 拉拍分析
    print("執行拉拍分析...")
    try:
        trajectory_json = load_json(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
        backswing_suggestion, backswing_confidence = analyze_backswing(
            trajectory_json, knn_dataset, expert_filename
        )
        result["analyses"]["backswing_advice"] = backswing_suggestion
        result["statistics"]["backswing_confidence"] = backswing_confidence
        print(f"拉拍分析完成: 信心度 {backswing_confidence:.2f}")
    except Exception as e:
        print(f"拉拍分析失敗: {e}")
        result["analyses"]["backswing_advice"] = "拉拍分析失敗"

    # 3. 前段出拍分析
    print("執行前段出拍分析...")
    try:
        trajectory_json = load_json(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
        forwardswing_suggestion, forwardswing_confidence = analyze_forwardswing(
            trajectory_json, knn_dataset, expert_filename
        )
        result["analyses"]["forwardswing_advice"] = forwardswing_suggestion
        result["statistics"]["forwardswing_confidence"] = forwardswing_confidence
        print(f"前段出拍分析完成: 信心度 {forwardswing_confidence:.2f}")
    except Exception as e:
        print(f"前段出拍分析失敗: {e}")
        result["analyses"]["forwardswing_advice"] = "前段出拍分析失敗"

    # 4. 擊球出拍轉身分析
    print("執行擊球出拍轉身分析...")
    try:
        trajectory_json = load_json(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
        hitballswing_suggestion, hitballswing_confidence = analyze_hitballswing(
            trajectory_json, knn_dataset, expert_filename
        )
        result["analyses"]["hitballswing_advice"] = hitballswing_suggestion
        result["statistics"]["hitballswing_confidence"] = hitballswing_confidence
        print(f"擊球出拍轉身分析完成: 信心度 {hitballswing_confidence:.2f}")
    except Exception as e:
        print(f"擊球出拍轉身分析失敗: {e}")
        result["analyses"]["hitballswing_advice"] = "擊球出拍轉身分析失敗"

    # 5. 收拍分析
    print("執行收拍分析...")
    try:
        trajectory_json = load_json(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
        followthrough_suggestion, followthrough_confidence = analyze_followthrough(
            trajectory_json, knn_dataset, expert_filename
        )
        result["analyses"]["followthrough_advice"] = followthrough_suggestion
        result["statistics"]["followthrough_confidence"] = followthrough_confidence
        print(f"收拍分析完成: 信心度 {followthrough_confidence:.2f}")
    except Exception as e:
        print(f"收拍分析失敗: {e}")
        result["analyses"]["followthrough_advice"] = "收拍分析失敗"

    # 6. 擊球點區域分析（球 vs 專業分佈）
    print("執行擊球點區域分析...")
    try:
        contact_zone = analyze_contact_zone(knn_dataset, trajectory_data)
        if isinstance(contact_zone, str):
            result["analyses"]["contact_zone_advice"] = contact_zone
        else:
            result["analyses"]["contact_zone_advice"] = contact_zone.get("advice", "")
            result["statistics"]["contact_zone"] = {
                "user_values": contact_zone.get("user_values", {}),
                "pro_ranges": contact_zone.get("pro_ranges", {}),
                "flags": contact_zone.get("flags", {}),
            }
        print("擊球點區域分析完成")
    except Exception as e:
        print(f"擊球點區域分析失敗: {e}")
        result["analyses"]["contact_zone_advice"] = "擊球點區域分析失敗"

    # 7. 頭部穩定度分析（眼睛盯球）
    print("執行頭部穩定度分析...")
    try:
        trajectory_json = load_json(trajectory_data) if isinstance(trajectory_data, str) else trajectory_data
        head_stability_suggestion, head_stability_confidence = analyze_head_stability(
            trajectory_json, knn_dataset, expert_filename
        )
        result["analyses"]["head_stability_advice"] = head_stability_suggestion
        result["statistics"]["head_stability_confidence"] = head_stability_confidence
        print(f"頭部穩定度分析完成: 信心度 {head_stability_confidence:.2f}")
    except Exception as e:
        print(f"頭部穩定度分析失敗: {e}")
        result["analyses"]["head_stability_advice"] = "頭部穩定度分析失敗"

    # 合併所有建議
    result["combined_advice"] = combine_all_advice(result["analyses"])

    # 依擊球點左右位置判斷動作類型：正拍 / 反拍
    # lateral_n > 0 → 正拍，lateral_n < 0 → 反拍
    cz = result.get("statistics", {}).get("contact_zone", {}).get("user_values", {})
    lateral_n = cz.get("lateral_n")
    if lateral_n is not None:
        if lateral_n > 0.1:
            result["action_type"] = "正拍"
        elif lateral_n < -0.1:
            result["action_type"] = "反拍"
        else:
            result["action_type"] = "未知動作"  # 擊球點接近身體中線時保留未知
    # 若 contact_zone 未產出則不寫入 action_type，前端會沿用舊邏輯或顯示未知

    print("整合分析完成")
    return result


def save_integrated_analysis(integrated_result, trajectory_data_path):
    """
    保存整合分析結果到檔案
    參數:
    - integrated_result: 整合分析結果字典
    - trajectory_data_path: 軌跡資料路徑（用於生成輸出檔名）
    返回: 輸出檔案路徑
    """
    # 生成輸出檔案路徑
    output_path = trajectory_data_path.replace(
        '(3D_trajectory_smoothed).json', 
        '_integrated_analysis.json'
    )
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_result, f, ensure_ascii=False, indent=2)
        print(f"整合分析結果已保存至: {output_path}")
        return output_path
    except Exception as e:
        print(f"保存整合分析結果失敗: {e}")
        # 備用檔案名
        backup_path = trajectory_data_path + '.integrated_analysis.json'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_result, f, ensure_ascii=False, indent=2)
        return backup_path


def analyze_integrated_trajectory(trajectory_data, knn_dataset, expert_filename, knn_suggestion=None):
    """
    主要的整合分析函式
    執行所有分析點並保存結果
    參數:
    - trajectory_data: 3D 軌跡資料路徑
    - knn_dataset: KNN 資料庫路徑  
    - expert_filename: 最相近的專家檔案名稱
    返回: 輸出檔案路徑
    """
    # 執行整合分析
    integrated_result = analyze_all_features(trajectory_data, knn_dataset, expert_filename, knn_suggestion)
    
    # 保存結果
    output_path = save_integrated_analysis(integrated_result, trajectory_data)
    
    return output_path


if __name__ == "__main__":
    # 測試用
    trajectory_path = "trajectory/testing_123/testing_(3D_trajectory_smoothed).json"
    knn_dataset_path = "knn_dataset.json"
    expert_name = "pro_2_4(3D_trajectory_smoothed).json"
    
    result_path = analyze_integrated_trajectory(
        trajectory_path, knn_dataset_path, expert_name
    )
    print(f"整合分析完成，結果保存至: {result_path}")
