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

        # FIX: response['message'] not response['messages']
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
        # Note: summarization is triggered automatically inside add_to_short_term
        # when the 10-message window is full — no manual trigger needed here.

        message_count = len(self.memory.short_term_memory.get("conversation_history", []))

        # ── Auto-extract attacker IPs from retrieved docs ──────────────
        ips_stored = 0
        for doc in state.get("retrieved_docs", []):
            ip_str = doc.get("metadata", {}).get("ips", "")
            if ip_str:
                for ip in ip_str.split(", "):
                    ip = ip.strip()
                    if ip:
                        self.memory.add_attacker_ip(ip, context=f"seen in {query_intent} event")
                        ips_stored += 1
        if ips_stored:
            self.log(f"Auto-stored {ips_stored} attacker IPs from retrieved docs.")

        # ── Auto-extract MITRE techniques from assessment ──────────────
        techniques_stored = 0
        mitre_assessment = state.get("mitre_assessment", "")
        if mitre_assessment and "T1" in mitre_assessment:
            import re
            technique_ids = list(set(re.findall(r"T\d{4}(?:\.\d{3})?", mitre_assessment)))
            for tid in technique_ids:
                self.memory.add_attack_technique(tid)
                techniques_stored += 1
            if techniques_stored:
                self.log(f"Auto-stored {techniques_stored} MITRE techniques: {technique_ids}")

        # ── LLM-based fact extraction ──────────────────────────────────
        should_extract = self.should_extract_facts(query, query_intent)
        facts_learned = 0

        if should_extract:
            learned = self.extract_incident_facts(query, final_answer)
            if learned:
                self.memory.add_learned_fact(
                    fact=learned["fact"],
                    category=learned["category"]
                )
                facts_learned += 1
                self.log(f"Extracted: [{learned['category']}] {learned['fact']}")
            else:
                self.log("No extractable facts from this interaction.")

        # ── Sync state with memory ─────────────────────────────────────
        state["conversation_history"] = self.memory.short_term_memory.get("conversation_history", [])
        state["conversation_summary"] = self.memory.short_term_memory.get("conversation_summary", "")
        state["user_preferences"] = self.memory.long_term_memory.get("user_preferences", {})
        state["learned_facts"] = self.memory.long_term_memory.get("learned_facts", [])

        total_facts = len(self.memory.long_term_memory.get("learned_facts", []))
        state.setdefault("agent_logs", [])
        state["agent_logs"].append(
            f"Memory: {message_count} msgs | {total_facts} facts "
            f"(+{facts_learned} extracted, +{ips_stored} IPs, +{techniques_stored} techniques)"
        )
        state.setdefault("reasoning_steps", [])
        state["reasoning_steps"].append(
            f"OBSERVATION: Memory updated — {message_count} messages, "
            f"{total_facts} total facts, {ips_stored} IPs, {techniques_stored} MITRE techniques stored."
        )

        self.log("Memory update complete.")
        return state