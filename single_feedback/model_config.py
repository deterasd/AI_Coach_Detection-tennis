# LM Studio 模型設定
# 如果使用 LM Studio，請將 MODEL 設為您在 LM Studio 中載入的模型名稱
# 例如: "llama-2-7b-chat", "mistral-7b-instruct", "local-model" 等
# 如果使用 OpenAI API，請設為 "gpt-4o", "gpt-4o-mini" 等（不可用 Hugging Face 格式如 google/gemma-3n-e4b）

MODEL = "gpt-4o-mini"  

# 優化參數設定以提升速度
TEMPERATURE = 0.5  # 降低溫度以加快生成速度
MAX_TOKENS = 100   # 降低 token 數量，2 句建議不需要太多
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
MAX_CONTEXT_QUESTIONS = 10
TOP_P = 0.9