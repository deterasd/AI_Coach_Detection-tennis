import json
import numpy as np


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_center_of_mass_feature(frames, normalize_by_height=True):
    """
    計算每幀的垂直重心代理指標：
    - 預設使用 (髖 - 踝) 的 y 軸差值。
    - 若 normalize_by_height=True，則會以頭頂–踝部距離或近似身高比例進行正規化。
      可減少不同身高或攝影比例對結果的影響。
    返回與幀數相等長度的浮點數列表（跳過沒有所需數據的幀）。
    """
    feature_values = []
    for frame in frames:
        hips, ankles, heads = [], [], []

        # 收集各部位 y 座標
        if 'left_hip' in frame and frame['left_hip'] and frame['left_hip'].get('y') is not None:
            hips.append(frame['left_hip']['y'])
        if 'right_hip' in frame and frame['right_hip'] and frame['right_hip'].get('y') is not None:
            hips.append(frame['right_hip']['y'])
        if 'left_ankle' in frame and frame['left_ankle'] and frame['left_ankle'].get('y') is not None:
            ankles.append(frame['left_ankle']['y'])
        if 'right_ankle' in frame and frame['right_ankle'] and frame['right_ankle'].get('y') is not None:
            ankles.append(frame['right_ankle']['y'])
        if 'nose' in frame and frame['nose'] and frame['nose'].get('y') is not None:
            heads.append(frame['nose']['y'])  # 使用 nose 或 head 作為頭頂點近似

        # 若關鍵點不足則跳過
        if len(hips) == 0 or len(ankles) == 0:
            continue

        mean_hip_y = float(np.mean(hips))
        mean_ankle_y = float(np.mean(ankles))

        # 若啟用 normalize_by_height，使用相對身高比例進行修正
        if normalize_by_height:
            # 使用鼻子到腳踝距離作為身高基準
            if len(ankles) > 0 and len(heads) > 0:
                mean_head_y = float(np.mean(heads))   # nose 的 y 值
                mean_ankle_y = float(np.mean(ankles)) # ankle 的 y 值
                height = abs(mean_head_y - mean_ankle_y)  # 以鼻子到腳踝距離近似身高
            else:
                continue  # 若無法偵測到 nose 或 ankle，略過該幀

            if height > 0:
                relative_com = abs(mean_hip_y - mean_ankle_y) / height
                feature_values.append(relative_com)
        else:
            feature_values.append(mean_hip_y - mean_ankle_y)

    return feature_values


def _minmax_normalize_sequences(sequences):
    """
    使用全域最小-最大值一起標準化多個一維序列。
    參數: sequences: list[list[float]]
    返回: list[list[float]] 具有相同形狀。
    """
    if not sequences:
        return sequences
    non_empty_arrays = [np.array(seq, dtype=float) for seq in sequences if len(seq) > 0]
    if len(non_empty_arrays) == 0:
        return sequences
    concat = np.concatenate(non_empty_arrays)
    min_v = float(np.min(concat))
    max_v = float(np.max(concat))
    if max_v - min_v == 0:
        return sequences
    normalized = []
    for seq in sequences:
        arr = np.array(seq, dtype=float)
        if arr.size == 0:
            normalized.append(seq)
        else:
            normalized.append(((arr - min_v) / (max_v - min_v)).tolist())
    return normalized


def _analyze_center_of_mass_height(current_seq, expert_features):
    """
    分析重心高低：比較當前重心與專家重心的平均高度
    回傳重心高低判斷和對應建議
    """
    if not current_seq or not expert_features:
        return "", ""

    # 計算當前重心的平均高度
    current_avg = np.mean(current_seq)

    # 計算所有專家重心的平均高度
    expert_avgs = [np.mean(seq) for seq in expert_features if len(seq) > 0]
    if not expert_avgs:
        return "", ""

    expert_mean = np.mean(expert_avgs)
    expert_std = np.std(expert_avgs)

    height_diff = current_avg - expert_mean

    # 若只有一位專家（標準差=0），改用相對比例比較
    if expert_std == 0:
        if expert_mean != 0:
            diff_ratio = height_diff / expert_mean
            if diff_ratio > 0.05:
                height_assessment = "重心偏高"
                height_advice = "建議降低重心，多屈膝以穩定下盤。"
            elif diff_ratio < -0.05:
                height_assessment = "重心偏低"
                height_advice = "建議適度提高重心，保持身體直立。"
            else:
                height_assessment = "重心適中"
                height_advice = "重心位置良好，請持續保持。"
        else:
            height_assessment = "資料不足"
            height_advice = ""
    else:
        if height_diff > expert_std * 0.5:
            height_assessment = "重心偏高"
            height_advice = "建議降低重心，多屈膝以穩定下盤，有助於擊球穩定性和力量傳遞。"
        elif height_diff < -expert_std * 0.5:
            height_assessment = "重心偏低"
            height_advice = "建議適度提高重心，保持身體直立，避免過度彎曲影響擊球流暢性。"
        else:
            height_assessment = "重心適中"
            height_advice = "重心位置良好，請繼續保持。"

    return height_assessment, height_advice


def analyze_center_of_mass(merged_dataset_path, dynamic_filename, knn_expert_filename=None, normalize_by_height=True):
    """
    基於原有 KNN 找到的專家進行重心分析，確保前後一致。
    - merged_dataset_path: knn_dataset.json 的路徑（專家資料庫）
    - dynamic_filename: 當前 3D 平滑軌跡 json 的路徑
    - knn_expert_filename: 原有 KNN 找到的專家檔案名稱（可選）
    - normalize_by_height: 是否啟用以身高比例修正的重心計算
    產生 JSON 結果並保存到動態檔案旁邊，返回其路徑。
    """
    merged_dataset = load_json(merged_dataset_path)
    trajectory_data = load_json(dynamic_filename)
    current_seq = _extract_center_of_mass_feature(trajectory_data, normalize_by_height=normalize_by_height)

    if knn_expert_filename:
        # 使用指定專家進行重心分析
        target_expert = next((e for e in merged_dataset if e.get("filename") == knn_expert_filename), None)

        if target_expert is None:
            result = {
                'feature': 'center_of_mass_vertical_diff',
                'status': 'expert_not_found',
                'message': f'專家 {knn_expert_filename} 未找到'
            }
        else:
            expert_seq = _extract_center_of_mass_feature(target_expert['data'], normalize_by_height=normalize_by_height)
            height_assessment, height_advice = _analyze_center_of_mass_height(current_seq, [expert_seq])

            if len(expert_seq) > 0 and len(current_seq) > 0:
                min_len = min(len(expert_seq), len(current_seq))
                distance = float(np.mean(np.abs(np.array(expert_seq[:min_len]) - np.array(current_seq[:min_len]))))
            else:
                distance = float('inf')

            expert_advice = target_expert.get('suggestion', 'None')
            combined_advice = (
                f"與專家「{knn_expert_filename}」的動作相似度距離為 {distance:.3f}。\n"
                f"【重心分析】{height_assessment}：{height_advice}\n"
                f"【專家建議】{expert_advice}"
            )

            result = {
                'feature': 'center_of_mass_vertical_diff',
                'nearest_expert': knn_expert_filename,
                'distance': distance,
                'advice': combined_advice,
                'height_assessment': height_assessment,
                'height_advice': height_advice,
                'expert_advice': expert_advice,
                'statistics': {
                    'sequence_len_test': len(current_seq),
                    'sequence_len_expert': len(expert_seq),
                    'current_avg_height': float(np.mean(current_seq)) if current_seq else 0,
                    'expert_avg_height': float(np.mean(expert_seq)) if expert_seq else 0
                }
            }

    else:
        # 多專家 KNN 模式
        expert_entries, expert_features, expert_suggestions = [], [], []
        for entry in merged_dataset:
            if not isinstance(entry, dict) or 'data' not in entry:
                continue
            if "是否擊球:否" in entry.get("suggestion", ""):
                continue
            seq = _extract_center_of_mass_feature(entry['data'], normalize_by_height=normalize_by_height)
            expert_entries.append(entry)
            expert_features.append(seq)
            expert_suggestions.append(entry.get('suggestion', 'None'))

        height_assessment, height_advice = _analyze_center_of_mass_height(current_seq, expert_features)

        normalized = _minmax_normalize_sequences(expert_features + [current_seq])
        normalized_experts, normalized_current = normalized[:-1], normalized[-1]

        distances = []
        for idx, seq in enumerate(normalized_experts):
            if len(seq) == 0 or len(normalized_current) == 0:
                distances.append((idx, float('inf')))
                continue
            min_len = min(len(seq), len(normalized_current))
            dist = float(np.mean(np.abs(np.array(seq[:min_len]) - np.array(normalized_current[:min_len]))))
            distances.append((idx, dist))

        if not distances:
            result = {
                'feature': 'center_of_mass_vertical_diff',
                'status': 'no_data',
                'message': 'No valid expert or current sequence.'
            }
        else:
            best_idx, best_dist = min(distances, key=lambda x: x[1])
            best_entry = expert_entries[best_idx]
            expert_advice = expert_suggestions[best_idx]
            combined_advice = f"{expert_advice}\n\n【重心分析】\n{height_assessment}：{height_advice}"

            result = {
                'feature': 'center_of_mass_vertical_diff',
                'nearest_expert': best_entry.get('filename', 'unknown'),
                'distance': best_dist,
                'advice': combined_advice,
                'height_assessment': height_assessment,
                'height_advice': height_advice,
                'expert_advice': expert_advice,
                'statistics': {
                    'sequence_len_test': len(normalized_current),
                    'sequence_len_expert': len(normalized_experts[best_idx]),
                    'current_avg_height': float(np.mean(current_seq)) if current_seq else 0,
                    'expert_avg_height': float(np.mean([np.mean(seq) for seq in expert_features if seq])) if expert_features else 0
                }
            }

        out_path = dynamic_filename.replace('(3D_trajectory_smoothed).json', '_center_of_mass_knn.json')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            out_path = dynamic_filename + '.center_of_mass_knn.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return out_path

    print(f"重心分析完成: {result.get('height_assessment', '未知')}")
    return result
