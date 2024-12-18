import unittest
from llm_utils.google import GoogleLLM
from llm_utils.huggingface import HuggingfaceLLM
from llm_utils.openai import OpenAILLM
from dotenv import load_dotenv
import os

class TestLLMUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Specify the path to the .env file
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path)

    def test_google_llm_generate(self):
        key = os.getenv("GOOGLE_API_KEY")
        llm = GoogleLLM(api_key=key, model_name="gemini-1.5-flash")
        response = llm.generate_content("What is the capital of France?")
        self.assertIsNotNone(response)

    def test_huggingface_llm_generate(self):
        key = os.getenv("HF_TOKEN")
        llm = HuggingfaceLLM(api_key=key, model_name="microsoft/Phi-3-mini-128k-instruct")
        response = llm.generate_content("What is the capital of France?")
        self.assertIsNotNone(response)

    def test_openai_llm_generate(self):
        key = os.getenv("OPENAI_API_KEY")
        llm = OpenAILLM(api_key=key, model_name="gpt-4")
        response = llm.generate_content("What is the capital of France?")
        self.assertIsNotNone(response)

if __name__ == "__main__":
    unittest.main()
