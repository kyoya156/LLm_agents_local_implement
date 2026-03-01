from typing import Dict
from .base_agent import BaseAgent
from . import prompts


class MemoryAgent(BaseAgent):
    """
    Manages short-term conversation history and long-term incident facts.
    Adapted for cybersecurity: tracks attacker IPs, affected users,
    attack techniques, and recommendations across the session.
    """

    # Personal-style indicators replaced with security incident indicators
    SECURITY_INDICATORS = [
        "failed password", "brute force", "port scan", "privilege escalation",
        "lateral movement", "exfiltration", "malware", "ransomware", "phishing",
        "unauthorized access", "suspicious", "attack", "compromise", "breach",
    ]

    def __init__(self, llm_model: str, memory_manager):
        super().__init__(llm_model)
        self.memory = memory_manager

    def should_extract_facts(self, query: str, query_intent: str) -> bool:
        """Decide whether this interaction contains incident facts worth remembering."""
        if query_intent in ("LOG_ANALYSIS", "INCIDENT_RESPONSE", "ANOMALY_DETECTION"):
            return True
        query_lower = query.lower()
        return any(ind in query_lower for ind in self.SECURITY_INDICATORS)

    def extract_incident_facts(self, query: str, final_answer: str) -> Dict:
        """Extract structured incident facts from the conversation turn."""
        extract_prompt = prompts.extract_incident_facts_prompt.format(
            query=query,
            final_answer=final_answer
        )
        response = self.call_llm([
            {"role": "system", "content": "You extract cybersecurity incident facts."},
            {"role": "user", "content": extract_prompt}
        ])

       
        extraction = response["message"]["content"]

        if "NONE" in extraction.upper() or "FACT:" not in extraction.upper():
            return None

        try:
            fact_part = extraction.split("FACT:")[1].split("CATEGORY:")[0].strip()
            category_part = extraction.split("CATEGORY:")[1].strip().split("\n")[0].strip()
            return {"fact": fact_part, "category": category_part}
        except (IndexError, AttributeError):
            self.log(f"Failed to parse fact extraction: {extraction[:100]}")
            return None

    def process(self, state: Dict) -> Dict:
        query = state.get("query", "")
        
        final_answer = state.get("final_answer", "")
        query_intent = state.get("query_intent", "")

        # Update short-term memory
        self.memory.add_to_short_term("user", query)
        self.memory.add_to_short_term("assistant", final_answer)

        message_count = len(self.memory.short_term_memory.get("conversation_history", []))
        self.log(f"Message count: {message_count}")

        
        if message_count % 6 == 0 and message_count > 0:
            self.log("Summarising conversation for short-term memory...")
            summary = self.memory.summarize_conversation()
            self.log(f"Summary: {summary[:80]}")

        # Extract and store security incident facts
        should_extract = self.should_extract_facts(query, query_intent)
        self.log(f"Should extract incident facts: {should_extract}")

        facts_learned = 0
        if should_extract:
            learned = self.extract_incident_facts(query, final_answer)
            if learned:
                self.memory.add_learned_fact(
                    fact=learned["fact"],
                    category=learned["category"]
                )
                facts_learned += 1
                self.log(f"Learned: [{learned['category']}] {learned['fact']}")
            else:
                self.log("No extractable facts from this interaction.")

        # Sync state with memory
        state["conversation_history"] = self.memory.short_term_memory.get("conversation_history", [])
        state["conversation_summary"] = self.memory.short_term_memory.get("conversation_summary", "")
        state["user_preferences"] = self.memory.long_term_memory.get("user_preferences", {})
        state["learned_facts"] = self.memory.long_term_memory.get("learned_facts", [])

        total_facts = len(self.memory.long_term_memory.get("learned_facts", []))
        state.setdefault("agent_logs", [])
        state["agent_logs"].append(
            f"Memory: {message_count} messages | "
            f"{total_facts} total facts (+{facts_learned} new)"
        )
        state.setdefault("reasoning_steps", [])
        state["reasoning_steps"].append(
            f"OBSERVATION: Memory updated — {message_count} messages, "
            f"{total_facts} incident facts stored."
        )

        self.log("Memory update complete.")
        return state