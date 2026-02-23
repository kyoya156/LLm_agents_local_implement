from typing import Dict
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from vector_db import VectorDBManager
from memory import MemoryManager
from agents import (
    PlannerAgent,
    RetrieverAgent,
    AnalyzerAgent,
    AnswerGeneratorAgent,
    MemoryAgent
)

# Define the state structure
class AgentState(TypedDict):
    # Query
    query: str
    query_intent: str
    
    # Retrieved information
    retrieved_docs: list
    quality_score: float
    verified_facts: list # Verified information
    
    # Output
    final_answer: str
    
    # Memory
    conversation_history: list # Short-term: Recent conversation
    conversation_summary: str # Short-term: Summary of conversation
    user_preferences: dict # Long-term: User preferences
    learned_facts: list # Long-term: Facts about the user

    # ReAct
    reasoning_steps: list # Reasoning steps taken by the planner
    actions_taken: list # Actions tracking
    should_continue: bool # Flag to indicate if the process should continue or stop

    
    # Logging
    agent_logs: list

class MultiAgentRAG:
    """Multi-Agent RAG System"""
    def __init__(self, vector_db: VectorDBManager, user_id: str = "default_user"):
        self.vector_db = vector_db
        # LLM model to use for all agents as this is a test implementation
        self.llm_model = "llama3" 
        self.memory = MemoryManager(user_id)

        # Initialize agents
        self.planner = PlannerAgent(self.llm_model, self.memory)
        self.retriever = RetrieverAgent(self.llm_model, self.memory, self.vector_db)
        self.analyzer = AnalyzerAgent(self.llm_model)
        self.answer_generator = AnswerGeneratorAgent(self.llm_model, self.memory)
        self.memory_agent = MemoryAgent(self.llm_model, self.memory)
        print("Agents initialized successfully.")

        # Build the workflow 
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        workflow.add_node("planner", self._run_planner)
        workflow.add_node("retriever", self._run_retriever)
        workflow.add_node("analyzer", self._run_analyzer)
        workflow.add_node("answer_generator", self._run_answer_generator)
        workflow.add_node("memory_agent", self._run_memory_agent)

        # Define execution flow
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "analyzer")
        workflow.add_edge("analyzer", "answer_generator")
        workflow.add_edge("answer_generator", "memory_agent")
        workflow.add_edge("memory_agent", END)

        return workflow
    
    def _run_planner(self, state: AgentState) -> AgentState:
        """Run the planner agent to determine the plan of action"""
        return self.planner.process(state)

    def _run_retriever(self, state: AgentState) -> AgentState:
        """Run the retriever agent to gather relevant information"""
        return self.retriever.process(state)

    def _run_analyzer(self, state: AgentState) -> AgentState:
        """Run the analyzer agent to assess the information"""
        return self.analyzer.process(state)

    def _run_answer_generator(self, state: AgentState) -> AgentState:
        """Run the answer generator agent to produce the final answer"""
        return self.answer_generator.process(state)

    def _run_memory_agent(self, state: AgentState) -> AgentState:
        """Run the memory agent to manage conversation history and user preferences"""
        return self.memory_agent.process(state)
    
    def process_query(self, query: str) -> str:
        """Process a user query through the multi-agent pipeline"""
        initial_state: AgentState = {
            "query": query,
            "query_intent": "",
            "retrieved_docs": [],
            "quality_score": 0.0,
            "verified_facts": [],
            "final_answer": "",
            "conversation_history": self.memory.short_term_memory["conversation_history"],
            "conversation_summary": self.memory.short_term_memory["conversation_summary"],
            "user_preferences": self.memory.long_term_memory["user_preferences"],
            "learned_facts": self.memory.long_term_memory["learned_facts"],
            "reasoning_steps": [],
            "actions_taken": [],
            "should_continue": True,
            "agent_logs": []
        }
        # Run the workflow with the initial state
        final_state = self.app.invoke(initial_state)

        return final_state
    
if __name__ == "__main__":
    # Test the modular system
    
    db = VectorDBManager()
    
    if db.get_collection_count() == 0:
        print("No documents in database. Please run setup.py first!")
    else:
        print("Testing Modular Multi-Agent RAG System")
        
        # Create system
        rag = MultiAgentRAG(db, user_id="test_user")
        
        # Test query
        result = rag.process_query("How do cats communicate with humans?")
        
        print(f"\n FINAL ANSWER:")
        print(result['final_answer'])
        
        print(f"\n STATISTICS:")
        print(f"   Intent: {result['query_intent']}")
        print(f"   Quality: {result['quality_score']:.2f}")
        print(f"   Docs Retrieved: {len(result['retrieved_docs'])}")
        print(f"   Facts Verified: {len(result['verified_facts'])}")
        
        print(f"\n MEMORY:")
        print(f"   Messages: {len(result['conversation_history'])}")
        print(f"   Learned Facts: {len(result['learned_facts'])}")
        
        print(f"\n AGENT LOGS:")
        for log in result['agent_logs']:
            print(f"   • {log}")