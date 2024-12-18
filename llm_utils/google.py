# src\google.py

import logging
import google.generativeai as genai
from llm_utils.base import LLM


class GoogleLLM(LLM):

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        if not api_key:
            raise ValueError("API key is required for Google LLM.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def generate_content(self, prompt: str) -> str:
        """Generates text using the LLM."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else None
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            return None