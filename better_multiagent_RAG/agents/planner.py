from typing import Dict
from .base_agent import BaseAgent
from . import prompts


class PlannerAgent(BaseAgent):
    """
    Plans the investigation strategy for a security query using ReAct pattern.
    Determines the action type and severity before retrieval begins.
    """

    def __init__(self, llm_model: str, memory_manager):
        super().__init__(llm_model)
        self.memory = memory_manager

    def process(self, state: Dict) -> Dict:
        """Plan the investigation steps based on the query and memory context."""

        query = state["query"]

        
        memory_context = self.memory.get_context_for_query(query)

        prompt = prompts.planner_prompt.format(
            query=query,
            memory_context=memory_context
        )

        response = self.call_llm([
            {"role": "system", "content": "You are a senior SOC analyst. Think step-by-step."},
            {"role": "user", "content": prompt}
        ])

        
        reasoning = response["message"]["content"]

        # Parse action and severity from structured response
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

        # Determine severity from planner's assessment
        severity = "UNKNOWN"
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if level in reasoning.upper():
                severity = level
                break

        # Initialise state lists if not present
        state.setdefault("reasoning_steps", [])
        state.setdefault("actions_taken", [])
        state.setdefault("agent_logs", [])

        state["reasoning_steps"].append(reasoning)
        state["planned_action"] = action
        state["severity"] = severity

        state["actions_taken"].append({
            "agent": self.agent_name,
            "action": action,
            "severity": severity,
            "reasoning": reasoning[:200],
        })

        state["agent_logs"].append(
            f"Planner: Action={action} | Severity={severity}"
        )

        self.log(f"Planned action: {action} | Severity: {severity}")
        return state