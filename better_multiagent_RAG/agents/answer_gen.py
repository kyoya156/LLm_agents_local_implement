from typing import Any, Dict, List
from .base_agent import BaseAgent
from memory import MemoryManager
import prompts


class AnswerGeneratorAgent(BaseAgent):
    """Agent responsible for generating the final answer based on verified facts and memory context."""
    def __init__(self, llm_model: str, memory_manager: MemoryManager):
        super().__init__(llm_model)
        self.memory = memory_manager

        self.style_mapping = {
            "DEFINITION": "concise and clear",
            "EXPLANATION": "detailed and thorough",
            "COMPARISON": "structured and comparative",
            "FACT": "precise and factual",
            "LIST": "organized and enumerated",
            "PERSONAL": "friendly and personalized"
        }

    def answer_style(self, query_intent: str) -> str:
        """Choose appropriate writing style based on query intent"""
        return self.style_mapping.get(query_intent, "clear and informative")
    
    def adjust_confidence_level(self, quality_score: float) -> str:
        """Determine confidence level based on quality score"""
        if quality_score >= 0.6:
            return "HIGH"
        elif quality_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_insufficient_data_response(self) -> str:
        """Generate response when there's insufficient data"""
        return (
            "I don't have enough information in my knowledge base to answer "
            "that question accurately. The available documents don't contain "
            "relevant details about this topic."
        )
    
    def generate_answer(self, query: str, query_intent: str, verified_facts: list,
                   quality_score: float, writing_style: str, memory_context: str) -> str:
        """Generate the final answer using LLM"""
        
        facts_text = "\n".join(verified_facts)

        answer_prompt = prompts.answer_generator_prompt.format(
            query=query,
            query_intent=query_intent,
            writing_style=writing_style,
            quality_score=quality_score,
            memory_context=memory_context,
            facts_text=facts_text
        )

        response = self.call_llm([
            {"role": "system", "content": f"You are a {writing_style} writer. Create clear, accurate answers."},
            {"role": "user", "content": answer_prompt}
        ])
        
        return response['message']['content']

    def process(self, state: Dict) -> Dict:
        """Generate the final answer."""
        
        query = state["query"]
        query_intent = state["query_intent"]
        writing_style = self.answer_style(query_intent)
        quality_score = state["quality_score"]
        verified_facts = state["verified_facts"]
        confidence = self.adjust_confidence_level(quality_score)

        state["reasoning_steps"].append({
            f"THOUGHT: Using {writing_style} style with {confidence} confidence"
        })

        memory_context = self.memory.get_context_for_query()

        # Generate the final answer
        if verified_facts == ["INSUFFICIENT_DATA"]:
            final_answer = self.generate_insufficient_data_response()
            self.log("Generated insufficient data response", "→")
        else:
            final_answer = self.generate_answer(
                query=query,
                query_intent=query_intent,
                verified_facts=verified_facts,
                quality_score=quality_score,
                writing_style=writing_style,
                memory_context=memory_context
            )

        state["generated_answer"] = final_answer
        state["agent_logs"].append(
            f"Writer: Generated {writing_style} answer "
            f"({len(final_answer)} chars, {confidence} confidence)"
        )
        state["reasoning_steps"].append(
            "OBSERVATION: Answer generated successfully"
        )
        self.log("Generated answer:")
        return state