from typing import Dict, List
from .base_agent import BaseAgent
from . import prompts


class AnswerGeneratorAgent(BaseAgent):
    """
    Generates a structured cybersecurity incident report based on
    verified facts, MITRE assessment, and memory context.
    """

    STYLE_MAP = {
        "LOG_ANALYSIS":       "technical and precise",
        "THREAT_INTEL":       "detailed and informative",
        "INCIDENT_RESPONSE":  "urgent and action-oriented",
        "ANOMALY_DETECTION":  "analytical and cautious",
        "GENERAL":            "clear and professional",
    }

    def __init__(self, llm_model: str, memory_manager):
        super().__init__(llm_model)
        self.memory = memory_manager

    def writing_style(self, query_intent: str) -> str:
        return self.STYLE_MAP.get(query_intent, "clear and professional")

    def confidence_label(self, quality_score: float) -> str:
        if quality_score >= 0.65:
            return "HIGH"
        elif quality_score >= 0.40:
            return "MEDIUM"
        return "LOW"

    def insufficient_data_response(self) -> str:
        return (
            "SUMMARY: Insufficient evidence to make a determination.\n\n"
            "THREAT ASSESSMENT: The available logs or documents do not contain enough "
            "information to identify a specific threat. This could indicate the event "
            "is benign, or that relevant log sources are not being captured.\n\n"
            "RECOMMENDED ACTIONS:\n"
            "• Expand log collection to cover more systems and services.\n"
            "• Review firewall, DNS, and endpoint logs for additional context.\n"
            "• Re-submit the query with more specific log entries or indicators."
        )

    def generate_report(self, query: str, query_intent: str, verified_facts: List[str],
                        quality_score: float, memory_context: str,
                        mitre_assessment: str, severity: str) -> str:
        """Call the LLM to produce the final incident report."""

        facts_text = "\n".join(f"- {f}" for f in verified_facts)
        style = self.writing_style(query_intent)

        answer_prompt = prompts.answer_generator_prompt.format(
            query=query,
            query_intent=query_intent,
            severity=severity,
            quality_score=quality_score,
            memory_context=memory_context,
            facts_text=facts_text,
            mitre_assessment=mitre_assessment,
        )

        response = self.call_llm([
            {"role": "system", "content": f"You are a {style} cybersecurity incident reporter."},
            {"role": "user", "content": answer_prompt}
        ])

        
        return response["message"]["content"]

    def process(self, state: Dict) -> Dict:
        query = state["query"]
        query_intent = state.get("query_intent", "GENERAL")
        quality_score = state.get("quality_score", 0.0)
        verified_facts = state.get("verified_facts", ["INSUFFICIENT_DATA"])
        mitre_assessment = state.get("mitre_assessment", "No MITRE assessment available.")
        severity = state.get("severity", "UNKNOWN")
        confidence = self.confidence_label(quality_score)

       
        memory_context = self.memory.get_context_for_query(query)

        state.setdefault("reasoning_steps", [])
        
        state["reasoning_steps"].append(
            f"THOUGHT: Generating {self.writing_style(query_intent)} report "
            f"with {confidence} confidence (severity={severity})"
        )

        if verified_facts == ["INSUFFICIENT_DATA"]:
            final_answer = self.insufficient_data_response()
            self.log("Generated insufficient-data response.")
        else:
            final_answer = self.generate_report(
                query=query,
                query_intent=query_intent,
                verified_facts=verified_facts,
                quality_score=quality_score,
                memory_context=memory_context,
                mitre_assessment=mitre_assessment,
                severity=severity,
            )

        # Store under BOTH keys so downstream agents and CLI both work
        state["generated_answer"] = final_answer
        state["final_answer"] = final_answer   

        state.setdefault("agent_logs", [])
        state["agent_logs"].append(
            f"Writer: {len(final_answer)} chars | {confidence} confidence | severity={severity}"
        )
        state["reasoning_steps"].append("OBSERVATION: Incident report generated.")

        self.log("Incident report generated.")
        return state