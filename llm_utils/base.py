# src\base.py

import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLM(ABC):

    def __init__(self, model_name: str = None):
        pass
    

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        pass