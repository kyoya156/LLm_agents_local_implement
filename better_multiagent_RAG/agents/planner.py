from typing import Dict
from .base_agent import BaseAgent
from memory import MemoryManager
from . import prompts

class PlannerAgent(BaseAgent):
    """Agent responsible for planning the reasoning process """
    def __init__(self, llm_model: str , memory_manager: MemoryManager):
        super().__init__(llm_model)
        self.memory = memory_manager

    def process(self, state: Dict) -> Dict:
        """Plan the reasoning steps based on the query and memory context."""
        
        query = state["query"]

        # Get memory context for the query
        memory_context = self.memory.get_context_for_query()


       # Create the prompt for the planner
        prompt = prompts.planner_prompt.format(query=query, memory_context=memory_context)

        # Call the LLM with the planner prompt
        response = self.call_llm([
            {"role": "system", "content": "You are a strategic planner. Think step-by-step using ReAct pattern."},
            {"role": "user", "content": prompt}
        ])

        reasoning =  response["message"]["content"]

        #store reasoning steps
        state["reasoning_steps"] = state.get("reasoning_steps", [])
        state["reasoning_steps"].append(reasoning)

        # Determine primary action from reasoning
        reasoning_lower = reasoning.lower()
        if "search_knowledge" in reasoning_lower:
            action = "SEARCH_KNOWLEDGE"
        elif "use_memory" in reasoning_lower:
            action = "USE_MEMORY"
        else:
            action = "SEARCH_KNOWLEDGE"

        # Record action taken
        state["actions_taken"] = state.get("actions_taken", [])
        state["actions_taken"].append({
            "agent": self.agent_name,
            "action": action,
            "reasoning": reasoning
        })

        # Add to agent logs
        state["agent_logs"] = state.get("agent_logs", [])
        state["agent_logs"].append(f"ReAct Planner: Planned approach - {action}")

        self.log(f"Planned action: {action} based on reasoning.")
        self.log(f"Reasoning: {reasoning}")
        return state

if __name__ == "__main__":
    # Quick test
    
    memory = MemoryManager()
    planner = PlannerAgent(
        llm_model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF',
        memory_manager=memory
    )
    
    test_state = {
        "query": "How do cats communicate?",
        "reasoning_steps": [],
        "actions_taken": [],
        "agent_logs": []
    }
    
    result = planner.process(test_state)
    print(f"\n✓ Test complete")
    print(f"Action planned: {result['actions_taken'][0]['action']}")