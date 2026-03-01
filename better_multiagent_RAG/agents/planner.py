from typing import Dict
from .base_agent import BaseAgent
from . import prompts


class PlannerAgent(BaseAgent):
    """
    Plans the investigation strategy using ReAct.
    Uses memory context to avoid re-investigating known threats
    and to connect new queries to prior incidents.
    """

    def __init__(self, llm_model: str, memory_manager):
        super().__init__(llm_model)
        self.memory = memory_manager

    def process(self, state: Dict) -> Dict:
        query = state["query"]

        # Fetch query-relevant memory context
        memory_context = self.memory.get_context_for_query(query)
        has_prior_context = memory_context != "No prior context."

        # System prompt adapts based on whether prior context exists
        if has_prior_context:
            system_msg = (
                "You are a senior SOC analyst with memory of past incidents. "
                "You MUST reference the prior context provided when planning — "
                "connect new observations to known attacker IPs, techniques, or patterns. "
                "Do not repeat investigation steps already completed in prior context."
            )
        else:
            system_msg = (
                "You are a senior SOC analyst. This appears to be a new incident with no prior context. "
                "Plan a thorough investigation from scratch."
            )

        prompt = prompts.planner_prompt.format(
            query=query,
            memory_context=memory_context
        )

        response = self.call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ])

        reasoning = response["message"]["content"]
        reasoning_lower = reasoning.lower()

        # Determine action type
        if "log_analysis" in reasoning_lower:
            action = "LOG_ANALYSIS"
        elif "threat_intel" in reasoning_lower:
            action = "THREAT_INTEL_LOOKUP"
        elif "incident_response" in reasoning_lower:
            action = "INCIDENT_RESPONSE"
        elif "anomaly_detection" in reasoning_lower:
            action = "ANOMALY_DETECTION"
        else:
            action = "GENERAL_QUERY"

        # Determine severity
        severity = "UNKNOWN"
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if level in reasoning.upper():
                severity = level
                break

        state.setdefault("reasoning_steps", [])
        state.setdefault("actions_taken", [])
        state.setdefault("agent_logs", [])

        state["reasoning_steps"].append(reasoning)
        state["planned_action"] = action
        state["severity"] = severity
        state["used_memory"] = has_prior_context

        state["actions_taken"].append({
            "agent": self.agent_name,
            "action": action,
            "severity": severity,
            "used_memory": has_prior_context,
            "reasoning": reasoning[:200],
        })

        state["agent_logs"].append(
            f"Planner: action={action} | severity={severity} | memory={'yes' if has_prior_context else 'no'}"
        )

        self.log(f"Action={action} | Severity={severity} | Memory used={has_prior_context}")
        return state