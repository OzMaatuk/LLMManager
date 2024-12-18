# llm_utils/manager.py

from typing import Optional
from llm_utils.google import GoogleLLM
from llm_utils.huggingface import HuggingfaceLLM
from llm_utils.openai import OpenAILLM
import os

class LLMUtils:
    """Manages interactions with Language Models (LLMs)."""

    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        if "gemini" in model_name:
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
            self.model = GoogleLLM(api_key=api_key, model_name=model_name)
        elif "gpt" in model_name:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            self.model = OpenAILLM(api_key=api_key, model_name=model_name)
        else:
            if not api_key:
                api_key = os.getenv("HF_TOKEN")
            self.model = HuggingfaceLLM(api_key=api_key, model_name=model_name)

    def generate_text(self, prompt: str) -> Optional[str]:
        return self.model.generate_content(prompt)