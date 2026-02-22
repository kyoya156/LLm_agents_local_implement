"""
Base class for all agents.
"""
from typing import Any, Dict
import ollama
class BaseAgent:
    """Base class for all agents."""
    def __init__(self, llm_model: str = "ollama3"):
        self.llm_model = llm_model
        self.agent_name = self.__class__.__name__

    def call_llm(self, message: list, stream: bool = False) -> Dict:
        """Call the LLM with a message and return the response."""
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=message,
                stream=stream
            )
            return response
        except Exception as e:
            print(f"Error occurred while calling LLM: {e}")
            raise

    def log(self, message: str):
        """Simple logging function for agent actions."""
        print(f"[{self.agent_name}] {message}")

    def process(self, state: Dict) -> Dict:
        """Process the given state and return a new state. To be implemented by subclasses or agents."""
        raise NotImplementedError(f"{self.agent_name} must implement the process() method.")