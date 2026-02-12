import os
import pandas as pd
import json
import time
import single_feedback.prompt as prompt
import single_feedback.model_config as model_config
from openai import OpenAI
from open_ai_key import api_key
import os
# api_key = os.getenv("OPENAI_API_KEY")

# --- 設定 API 參數與載入 Prompt 與模型設定 ---
client = OpenAI(api_key=api_key)
MODEL = model_config.MODEL
TEMPERATURE = model_config.TEMPERATURE
MAX_TOKENS = model_config.MAX_TOKENS
FREQUENCY_PENALTY = model_config.FREQUENCY_PENALTY
PRESENCE_PENALTY = model_config.PRESENCE_PENALTY
TOP_P = model_config.TOP_P

INSTRUCTIONS = prompt.INSTRUCTIONS
DATADESCIRBE = prompt.DATADESCIRBE

def create_chat_completion(messages):
    """
    以給定的 messages 呼叫 OpenAI ChatCompletion
    回傳產生的 completion 結果
    """
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion

def generate_feedback(json_filepath, txt_filepath, integrated_analysis_path=None, output_path=None):
    """
    讀取 JSON (運動軌跡) 與 KNN 結果(txt)，並綜合兩者資訊產出 GPT 回饋
    最後將結果輸出為 _gpt_feedback.json 檔。

    output_path: 可選；若提供則寫入此路徑，否則依 json_filepath 推導（only_swing → _gpt_feedback.json）。
    """
    # 讀取運動軌跡資料
    my_motion = pd.read_json(json_filepath)

    # 讀取或解讀 KNN 回饋來源：
    # - 若是 list：取第一個當作文字內容
    # - 若是存在的檔案路徑：用 csv 讀取（處理空檔案情況）
    # - 其他情況：直接轉成字串
    if isinstance(txt_filepath, list):
        knn_feedback = str(txt_filepath[0]) if txt_filepath else ""
    elif isinstance(txt_filepath, str) and os.path.exists(txt_filepath):
        try:
            # 檢查檔案是否為空
            if os.path.getsize(txt_filepath) == 0:
                knn_feedback = ""
            else:
                df = pd.read_csv(txt_filepath, header=None)
                if df.empty or len(df) == 0:
                    knn_feedback = ""
                else:
                    knn_feedback = df.iloc[0, 0] if len(df.columns) > 0 else ""
        except (pd.errors.EmptyDataError, IndexError, Exception) as e:
            print(f"讀取 KNN feedback 檔案失敗（可能為空檔案）: {e}")
            knn_feedback = ""
    else:
        knn_feedback = str(txt_filepath) if txt_filepath is not None else ""

    # 初始化 messages 列表
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "system", "content": DATADESCIRBE},
    ]
    
    # 優先使用整合分析結果
    combined_advice = None
    if integrated_analysis_path:
        try:
            with open(integrated_analysis_path, 'r', encoding='utf-8') as f:
                integrated_data = json.load(f)
            combined_advice = integrated_data.get("combined_advice", "")
            # 若缺少合併字串，嘗試以 analyses 的建議組裝（確保 contact zone 也被整合）
            if not combined_advice:
                analyses = integrated_data.get("analyses", {}) or {}
                order = [
                    "backswing_advice",
                    "forwardswing_advice",
                    "hitballswing_advice",
                    "followthrough_advice",
                    "contact_zone_advice",
                    "head_stability_advice",
                ]
                combined_parts = [analyses[k] for k in order if analyses.get(k)]
                combined_advice = "\n\n".join(combined_parts)
            print(f"使用整合分析結果: {integrated_analysis_path}")
        except Exception as e:
            print(f"讀取整合分析結果失敗: {e}")
            combined_advice = None

    # 使用整合分析結果或原有邏輯
    if combined_advice:
        # 嘗試使用 GPT 重新描述；若失敗則優雅降級為直接輸出 combined_advice
        try:
            messages.append({
                "role": "user",
                "content": f"""
                    以下是網球揮拍的分析結果：
                    
                    {combined_advice}
                    
                    請用口語化、自然的方式重新描述這些建議，就像你是一位教練站在球場上跟學員面對面說話一樣。
                    
                    重要要求：
                    1. 不要使用任何列點、編號或結構化格式（如 A., B., C.）
                    2. 只用 2-3 句簡短、完整的句子
                    3. 語氣要親切、鼓勵，像真人對話
                    4. 先肯定做得好的地方，再簡短提出改進建議
                    5. 使用繁體中文
                    
                    直接開始你的建議，保持簡短。
                """
            })
            # 讓 GPT 重新描述建議
            knn_completion = create_chat_completion(messages)
            knn_response = knn_completion.choices[0].message.content
            
            # 讓 GPT 推測問題幀範圍
            messages.append({
                "role": "user",
                "content": f"""
                    Based on this {my_motion}, 
                    Speculate in which frame section the issue described in the feedback occurs. 
                    Please provide a broader frame range covering more frames (e.g., a range of at least 15 frames), 
                    and You MUST respond with a numeric range only, in the format "number-number" (e.g., "13-24"), 
                    containing only digits and a hyphen, with no additional text or formatting.
                """
            })
            frame_completion = create_chat_completion(messages)
            frame_response = frame_completion.choices[0].message.content
            
        except Exception as e:
            print(f"GPT 重寫失敗，改用整合建議原文：{e}")
            knn_response = combined_advice
            frame_response = "90-112"
        
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})
        
    elif knn_feedback and knn_feedback == "頭:沒問題!、肩膀:沒問題!、手碗:沒問題!、手肘:沒問題!、膝蓋:沒問題!、是否擊球:是、其他:無":
        # 原有邏輯：特定正向回饋訊息
        knn_response = "沒有觀察到顯著問題，請繼續保持！"
        frame_response = "0-0"

        # 將 frame 與建議回饋一起附加到 messages 中
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})

    elif knn_feedback:
        # 如果沒有整合分析結果，但有 KNN 回饋，使用原有邏輯
        # 第一次讓 GPT 根據 KNN Feedback 產生中文敘述
        messages.append({
            "role": "user",
            "content": f"""
                observe analysis results: {knn_feedback}, 
                Rephrase the analysis results of each body part in 1 sentence
            """
        })
        knn_completion = create_chat_completion(messages)
        knn_response = knn_completion.choices[0].message.content

        # 讓 GPT 根據 json 內容推測大致在第幾幀區間會出現問題
        messages.append({
            "role": "user",
            "content": f"""
                Based on this {my_motion}, 
                Speculate in which frame section the issue described in the feedback occurs. 
                Please provide a broader frame range covering more frames (e.g., a range of at least 15 frames), 
                and You MUST respond with a numeric range only, in the format "number-number" (e.g., "13-24"), 
                containing only digits and a hyphen, with no additional text or formatting.
            """
        })
        frame_completion = create_chat_completion(messages)
        frame_response = frame_completion.choices[0].message.content

        # 將數字範圍與 knn_response 加入到 messages (可以用於後續檢視或除錯)
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})
    
    else:
        # 如果既沒有整合分析結果，也沒有 KNN 回饋，使用預設訊息
        print("警告: 沒有可用的分析結果，使用預設回饋")
        knn_response = "無法取得分析結果，請檢查輸入資料。"
        frame_response = "0-0"
        messages.append({"role": "assistant", "content": frame_response})
        messages.append({"role": "assistant", "content": knn_response})

    # 處理換行符號
    frame_response = frame_response.replace("\n", "")
    knn_response = knn_response.replace("\n", "")

    
    # 構造 JSON 格式回傳結果
    ai_feedback = {
        "problem_frame": frame_response,
        "suggestion": knn_response,
    }

    print(ai_feedback)

    # 輸出檔案路徑 (以原檔案名稱 + "_gpt_feedback.json"，或使用 output_path)
    if output_path is not None:
        output_filepath = output_path
    else:
        output_filepath = json_filepath.replace('(3D_trajectory_smoothed)_only_swing.json', '_gpt_feedback.json')
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(ai_feedback, f, ensure_ascii=False, indent=2)

    return output_filepath


def generate_feedback_data_only(json_filepath, txt_filepath):
    """
    讀取 JSON (運動軌跡) 與 KNN 結果(txt)，並綜合兩者資訊產出 GPT 回饋
    回傳字典格式資料，不寫入檔案
    """
    try:
        # 讀取運動軌跡資料與 KNN 回饋
        my_motion = pd.read_json(json_filepath)
        knn_feedback = pd.read_csv(txt_filepath, header=None).iloc[0, 0]

        # 初始化 messages 列表
        messages = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "system", "content": DATADESCIRBE},
        ]

        # 如果 knn_feedback 為特定正向回饋訊息
        if knn_feedback == "頭:沒問題!、肩膀:沒問題!、手碗:沒問題!、手肘:沒問題!、膝蓋:沒問題!、是否擊球:是、其他:無":
            knn_response = "沒有觀察到顯著問題，請繼續保持！"
            frame_response = "0-0"

            # 將 frame 與建議回饋一起附加到 messages 中
            messages.append({"role": "assistant", "content": frame_response})
            messages.append({"role": "assistant", "content": knn_response})

        else:
            # 第一次讓 GPT 根據 KNN Feedback 產生中文敘述
            messages.append({
                "role": "user",
                "content": f"""
                    observe analysis results: {knn_feedback}, 
                    Rephrase the analysis results of each body part in 1 sentence
                """
            })
            knn_completion = create_chat_completion(messages)
            knn_response = knn_completion.choices[0].message.content

            # 讓 GPT 根據 json 內容推測大致在第幾幀區間會出現問題
            messages.append({
                "role": "user",
                "content": f"""
                    Based on this {my_motion}, 
                    Speculate in which frame section the issue described in the feedback occurs. 
                    Please provide a broader frame range covering more frames (e.g., a range of at least 15 frames), 
                    and You MUST respond with a numeric range only, in the format "number-number" (e.g., "13-24"), 
                    containing only digits and a hyphen, with no additional text or formatting.
                """
            })
            frame_completion = create_chat_completion(messages)
            frame_response = frame_completion.choices[0].message.content

            # 將數字範圍與 knn_response 加入到 messages (可以用於後續檢視或除錯)
            messages.append({"role": "assistant", "content": frame_response})
            messages.append({"role": "assistant", "content": knn_response})

        # 處理換行符號
        frame_response = frame_response.replace("\n", "")
        knn_response = knn_response.replace("\n", "")

        # 構造 JSON 格式回傳結果
        ai_feedback = {
            "problem_frame": frame_response,
            "suggestion": knn_response,
        }

        return ai_feedback
        
    except Exception as e:
        # 如果發生錯誤，回傳錯誤訊息
        print(f"⚠️ GPT 反饋生成失敗: {e}")
        return {
            "problem_frame": "N/A",
            "suggestion": "GPT功能暫時無法使用，請參考KNN分析結果",
            "error": True,
            "error_type": "processing_error"
        }


if __name__ == "__main__":
    json_path = "嘉洋__3(3D_trajectory_smoothed).json"
    txt_path = "嘉洋__3_knn_feedback.txt"

    # 開始計時
    start_time = time.time()

    # 產生並輸出回饋
    output_filepath = generate_feedback(json_path, txt_path)

    # 結束計時
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("AI Feedback:")
    print(f"Processing time: {elapsed_time:.2f} seconds")