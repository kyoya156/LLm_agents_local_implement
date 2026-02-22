from typing import Any, Dict, List
from .base_agent import BaseAgent
from memory import MemoryManager
import prompts

class MemoryAgent(BaseAgent):
    """Agent responsible for managing memory and context."""
    def __init__(self, llm_model: str, memory_manager: MemoryManager):
        super().__init__(llm_model)
        self.memory = memory_manager

        # Triggers for long-term memory updates
        self.personal_indicators = [
            "I have", "I like", "I prefer", "my", "I am",
            "I'm worried", "I need", "I want"
        ]
    
    def should_extract_personal_info(self, query: str, query_intent: str) -> bool:
        """Determine if query contains personal information to remember"""
        if query_intent == "PERSONAL":
            return True
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in self.personal_indicators)

    def extract_facts(self, query: str, final_answer: str) -> Dict:
        """Extract relevant facts about user from the conversation."""
        
        extract_prompt = prompts.extract_facts_prompt.format(query=query, final_answer=final_answer)

        response = self.call_llm([
            {"role": "system", "content": "You extract learnable facts about users."},
            {"role": "user", "content": extract_prompt}
        ])

        extraction = response['message']['content']

        if "NONE" in extraction.upper() or not "FACTS" in extraction.upper():
            return None
        
        try:
            fact_part = extraction.split("FACT:")[1].split("|")[0].split("CATEGORY:")[0].strip()
            category_part = extraction.split("CATEGORY:")[1].strip()
            
            return {
                "fact": fact_part,
                "category": category_part
            }
        except (IndexError, AttributeError):
            self.log("Failed to parse fact extraction response:", extraction)
            return None

    def process(self, state: Dict) -> Dict:
        """Process memory-related tasks."""
        query = state.get("query", "")
        final_answer = state["final_answer"]
        # Extract facts if query contains personal info or is explicitly personal
        query_intent = state.get("query_intent", "")

        self.memory.add_to_short_term("user", query)
        self.memory.add_to_short_term("assistant", final_answer)

        message_count = len(self.memory.short_term_memory)
        self.log(f"MemoryAgent Current message count: {message_count}")

        if message_count % 6 and  message_count > 0:  # Every 6 messages, attempt to extract facts
            self.log("Summarizing conversations for short-term memory...")
            summary = self.memory.summarize_conversation()
            self.log("Summary:", summary)
        
        should_extract = self.should_extract_personal_info(query, query_intent)
        self.log(f"Should extract personal info: {should_extract}")

        facts_learned = 0

        if should_extract:
            learned = self.extract_facts(query, final_answer)
            if learned:
                self.memory.add_learned_fact(
                    fact=learned["fact"],
                    category=learned["category"]
                )
                facts_learned += 1
                self.log(f"Learned new fact: {learned['fact']} (Category: {learned['category']})")
            else:
                self.log("No learnable facts extracted from this interaction.")

        # Update state with memory info
        state["conversation_history"] = self.memory.conversation_history
        state["conversation_summary"] = self.memory.conversation_summary
        state["user_preferences"] = self.memory.user_preferences
        state["learned_facts"] = self.memory.learned_facts

        # Update agent logs
        state["agent_logs"].append(
            f"Memory: Updated short-term ({message_count} messages) "
            f"and long-term ({len(self.memory.learned_facts)} total facts, "
            f"+{facts_learned} new)"
        )
        
        # Add to reasoning trace
        state["reasoning_steps"].append(
            f"OBSERVATION: Memory updated - {message_count} messages in history, "
            f"{len(self.memory.learned_facts)} facts learned"
        )

        self.log(f"Memory update complete")
        return state