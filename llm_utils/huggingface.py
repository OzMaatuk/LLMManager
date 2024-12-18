# llm_utils/huggingface.py

import logging
import requests
from llm_utils.base import LLM


class HuggingfaceLLM(LLM):
    """Implementation of Hugging Face's Inference API."""

    def __init__(self, api_key: str, model_name: str = "microsoft/Phi-3-mini-128k-instruct"):
        super().__init__(model_name)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }
        if not api_key:
            raise ValueError("API key is required for HuggingFace.")

    def generate_content(self, prompt: str) -> str:
        if not self.headers.get("Authorization"):
            logging.error("Authorization header is missing. Cannot generate content.")
            return None
        try:
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                json={"inputs": prompt}
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                else:
                    logging.error("Unexpected response format from Hugging Face API.")
                    return None
            else:
                logging.error(f"Hugging Face API returned an error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error generating text with Hugging Face Inference API: {e}")
            return None
