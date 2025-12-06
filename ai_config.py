# AI 模型配置管理
# 支援 LM Studio 本地模型、遠程模型和 OpenAI API 的自動切換

import os
import json
from datetime import datetime
from openai import OpenAI

class AIConfig:
    """AI 模型配置管理器"""

    def __init__(self):
        # 載入 config.json 設定
        self.config = self._load_config()
        
        # 嘗試載入遠程 LM Studio 配置
        self._load_remote_config()

        # 從環境變數或檔案讀取 OpenAI API 金鑰
        self.openai_key = self._get_openai_key()

        # 根據 config.json 決定使用哪個服務
        self.use_lm_studio = self._should_use_lm_studio()
        self.client = None
        self.current_provider = None
        self.current_model = None
        self._initialize_client()

    def _load_remote_config(self):
        """載入遠程 LM Studio 配置"""
        try:
            from remote_lm_studio_config import REMOTE_LM_STUDIO_URL, REMOTE_LM_STUDIO_API_KEY
            self.lm_studio_url = REMOTE_LM_STUDIO_URL
            self.lm_studio_key = REMOTE_LM_STUDIO_API_KEY
            print(f"✅ 已載入遠程 LM Studio 配置: {self.lm_studio_url}")
        except ImportError:
            # 使用本地預設值
            self.lm_studio_url = "http://localhost:1234/v1"
            self.lm_studio_key = "lm-studio"
            print("ℹ️ 使用本地 LM Studio 配置，如需遠程連接請編輯 remote_lm_studio_config.py")

    def _load_config(self):
        """載入 config.json 設定檔案"""
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                print("✅ 已載入 config.json 設定")
                return config
        except FileNotFoundError:
            print("ℹ️ 找不到 config.json，使用預設設定")
            return {"ai_model": {"provider": "auto"}}
        except json.JSONDecodeError as e:
            print(f"⚠️ config.json 格式錯誤: {e}，使用預設設定")
            return {"ai_model": {"provider": "auto"}}

    def _should_use_lm_studio(self):
        """根據 config.json 決定是否使用 LM Studio"""
        ai_model_config = self.config.get("ai_model", {})
        provider = ai_model_config.get("provider", "auto")
        
        print(f"🔧 AI 模型設定: provider={provider}")
        
        if provider == "auto":
            return True  # 保持原有自動偵測邏輯（優先 LM Studio）
        elif provider == "lm-studio":
            return True
        elif provider in ["gpt-4o", "gpt-4o-mini", "openai"]:
            return False
        else:
            print(f"⚠️ 不支援的 provider: {provider}，使用自動偵測")
            return True

    def _get_openai_key(self):
        """獲取 OpenAI API 金鑰"""
        # 優先從環境變數讀取
        key = os.getenv('OPENAI_API_KEY')
        if key:
            return key

        # 從檔案讀取
        try:
            from open_ai_key import api_key
            return api_key
        except ImportError:
            print("⚠️ 無法讀取 OpenAI API 金鑰")
            return None

    def _initialize_client(self):
        """初始化 AI 客戶端"""
        if self.use_lm_studio:
            try:
                self.client = OpenAI(
                    base_url=self.lm_studio_url,
                    api_key=self.lm_studio_key
                )
                # 測試連接
                self.client.models.list()
                self.current_provider = "lm-studio"
                
                # 決定模型名稱
                ai_model_config = self.config.get("ai_model", {})
                model_name = ai_model_config.get("model_name", "auto")
                if model_name == "auto":
                    from single_feedback import model_config
                    self.current_model = model_config.MODEL
                else:
                    self.current_model = model_name
                
                print(f"✅ 已連接到 LM Studio 服務器: {self.current_model}")
            except Exception as e:
                print(f"⚠️ LM Studio 連接失敗 ({e})，切換到 OpenAI API")
                self.use_lm_studio = False
                self._initialize_openai_client()
        else:
            self._initialize_openai_client()

    def _initialize_openai_client(self):
        """初始化 OpenAI 客戶端"""
        if self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
            self.current_provider = "openai"
            
            # 決定模型名稱
            ai_model_config = self.config.get("ai_model", {})
            provider = ai_model_config.get("provider", "auto")
            model_name = ai_model_config.get("model_name", "auto")
            
            if model_name != "auto":
                self.current_model = model_name
            elif provider == "gpt-4o":
                self.current_model = "gpt-4o"
            elif provider == "gpt-4o-mini":
                self.current_model = "gpt-4o-mini"
            else:
                self.current_model = "gpt-4o-mini"  # 預設
            
            print(f"✅ 已連接到 OpenAI API: {self.current_model}")
        else:
            print("❌ 無法初始化任何 AI 客戶端")
            self.client = None

    def is_lm_studio(self):
        """檢查是否使用 LM Studio"""
        return self.use_lm_studio and self.client is not None

    def get_client(self):
        """獲取 AI 客戶端"""
        return self.client

    def switch_to_openai(self):
        """手動切換到 OpenAI API"""
        self.use_lm_studio = False
        self._initialize_openai_client()

    def switch_to_lm_studio(self):
        """手動切換到 LM Studio"""
        self.use_lm_studio = True
        self._initialize_client()

    def get_model_name(self):
        """獲取當前模型名稱"""
        return self.current_model

    def get_model_config(self):
        """獲取當前模型的配置參數"""
        ai_model_config = self.config.get("ai_model", {})
        
        # 從 config.json 獲取覆寫參數，否則使用 model_config.py 預設值
        from single_feedback import model_config
        
        return {
            "temperature": ai_model_config.get("temperature", model_config.TEMPERATURE),
            "max_tokens": ai_model_config.get("max_tokens", model_config.MAX_TOKENS),
            "frequency_penalty": getattr(model_config, "FREQUENCY_PENALTY", 0.0),
            "presence_penalty": getattr(model_config, "PRESENCE_PENALTY", 0.0),
            "top_p": getattr(model_config, "TOP_P", 0.9)
        }

    def reload_config(self):
        """重新載入設定檔案"""
        self.config = self._load_config()
        self.use_lm_studio = self._should_use_lm_studio()
        self._initialize_client()

# 全域配置實例
ai_config = AIConfig()