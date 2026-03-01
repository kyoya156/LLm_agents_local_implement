"""
Multi-Agent Cybersecurity RAG System — Main Orchestrator
"""
from typing import Dict
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from vector_db import VectorDBManager
from memory import MemoryManager
from mitre_kb import MitreKnowledgeBase
from log_parser import LogParser
from agents import (
    PlannerAgent,
    RetrieverAgent,
    AnalyzerAgent,
    AnswerGeneratorAgent,
    MemoryAgent,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # Query
    query: str
    query_intent: str
    planned_action: str
    severity: str

    # Retrieved information
    retrieved_docs: list
    quality_score: float
    verified_facts: list
    mitre_assessment: str

    # Output
    generated_answer: str
    final_answer: str       # alias kept for memory_agent compatibility

    # Memory
    conversation_history: list
    conversation_summary: str
    user_preferences: dict
    learned_facts: list
    used_memory: bool

    # ReAct trace
    reasoning_steps: list
    actions_taken: list
    should_continue: bool

    # Logging
    agent_logs: list


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class MultiAgentRAG:
    """
    Cybersecurity Multi-Agent RAG System.

    Pipeline:  Planner → Retriever → Analyzer → AnswerGenerator → MemoryAgent
    """

    def __init__(
        self,
        vector_db: VectorDBManager,
        user_id: str = "analyst",
        llm_model: str = "llama3",
        mitre_stix_path: str = "enterprise-attack.json",
    ):
        self.vector_db = vector_db
        self.llm_model = llm_model
        self.memory = MemoryManager(user_id)
        self.mitre_kb = MitreKnowledgeBase(mitre_stix_path)
        self.log_parser = LogParser()

        # Initialise agents
        self.planner = PlannerAgent(llm_model, self.memory)
        self.retriever = RetrieverAgent(llm_model, self.memory, vector_db, self.log_parser)
        self.analyzer = AnalyzerAgent(llm_model, self.mitre_kb)
        self.answer_generator = AnswerGeneratorAgent(llm_model, self.memory)
        self.memory_agent = MemoryAgent(llm_model, self.memory)

        print(f"[MultiAgentRAG] Agents initialised | user={user_id} | model={llm_model}")
        print(f"[MultiAgentRAG] MITRE KB: {len(self.mitre_kb.techniques)} techniques loaded")
        print(f"[MultiAgentRAG] Vector DB: {vector_db.get_collection_count()} documents")

        self.app = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner",          self._run_planner)
        workflow.add_node("retriever",         self._run_retriever)
        workflow.add_node("analyzer",          self._run_analyzer)
        workflow.add_node("answer_generator",  self._run_answer_generator)
        workflow.add_node("memory_agent",      self._run_memory_agent)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner",         "retriever")
        workflow.add_edge("retriever",        "analyzer")
        workflow.add_edge("analyzer",         "answer_generator")
        workflow.add_edge("answer_generator", "memory_agent")
        workflow.add_edge("memory_agent",     END)

        return workflow

    # ------------------------------------------------------------------
    # Node wrappers
    # ------------------------------------------------------------------
    def _run_planner(self, state):         return self.planner.process(state)
    def _run_retriever(self, state):       return self.retriever.process(state)
    def _run_analyzer(self, state):        return self.analyzer.process(state)
    def _run_answer_generator(self, state): return self.answer_generator.process(state)
    def _run_memory_agent(self, state):    return self.memory_agent.process(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_query(self, query: str) -> Dict:
        """Run a security query through the full agent pipeline."""
        initial_state: AgentState = {
            "query": query,
            "query_intent": "",
            "planned_action": "",
            "severity": "UNKNOWN",
            "retrieved_docs": [],
            "quality_score": 0.0,
            "verified_facts": [],
            "mitre_assessment": "",
            "generated_answer": "",
            "final_answer": "",
            "conversation_history": self.memory.short_term_memory.get("conversation_history", []),
            "conversation_summary": self.memory.short_term_memory.get("conversation_summary", ""),
            "user_preferences": self.memory.long_term_memory.get("user_preferences", {}),
            "learned_facts": self.memory.long_term_memory.get("learned_facts", []),
            "reasoning_steps": [],
            "actions_taken": [],
            "should_continue": True,
            "agent_logs": [],
        }
        return self.app.invoke(initial_state)

    def analyze_log_file(self, filepath: str) -> Dict:
        """
        Convenience method: parse a log file and run a query
        asking for a full threat assessment.
        """
        events = self.log_parser.parse_file(filepath)
        summary = self.log_parser.summarize(events)
        suspicious = self.log_parser.get_suspicious_events(events)

        if not suspicious:
            return {"final_answer": "No suspicious events found in the log file.", "agent_logs": []}

        # Build a query that includes the top suspicious log lines
        top_lines = "\n".join(e["raw"] for e in suspicious[:20])
        query = (
            f"Analyze these suspicious log entries and provide a full threat assessment:\n\n"
            f"{top_lines}\n\n"
            f"Summary: {summary['suspicious_count']}/{summary['total_lines']} lines flagged. "
            f"Top flags: {summary['top_flags'][:3]}. "
            f"Top IPs: {summary['top_source_ips'][:3]}."
        )
        return self.process_query(query)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    db = VectorDBManager()
    if db.get_collection_count() == 0:
        print("No documents in DB — run setup.py first.")
    else:
        rag = MultiAgentRAG(db, user_id="test_analyst")
        result = rag.process_query(
            "Jan  5 12:01:03 server sshd[1234]: Failed password for root from 192.168.1.105 port 22 ssh2\n"
            "Jan  5 12:01:05 server sshd[1234]: Failed password for root from 192.168.1.105 port 22 ssh2\n"
            "Jan  5 12:01:08 server sshd[1234]: Failed password for root from 192.168.1.105 port 22 ssh2"
        )
        print("\n=== INCIDENT REPORT ===")
        print(result["final_answer"])
        print(f"\nMITRE: {result['mitre_assessment'][:300]}")