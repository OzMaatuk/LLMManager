# llm_utils/openai.py

import logging
import openai
from llm_utils.base import LLM


class OpenAILLM(LLM):
    """Implementation of OpenAI's GPT models."""

    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        super().__init__(model_name)
        self.model_name = model_name
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key is required for OpenAI LLM.")

    def generate_content(self, prompt: str) -> str:
        try:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt,
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].text.strip() if response.choices else None
        except Exception as e:
            logging.error(f"Error generating text with OpenAI model: {e}")
            return None
