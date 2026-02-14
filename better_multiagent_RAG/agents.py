"""
Multi-Agent RAG System
Three agents: Retriever -> Analyzer -> Answer Generator
Similar  to the older one but this time i will try to build a proper RAG retrieval system with better prompts and analysis.
"""
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import ollama
from vector_db import VectorDBManager
import prompts

# Define the state flow between agents
class AgentState(TypedDict):
    query: str
    query_intent: str
    retrieved_docs: List[Dict]
    quality_score: float # between 0 and 1
    verified_facts: List[str]  # Verified information
    agent_logs: List[str]
    final_answer: str

class MultiAgentRAG:
    """Multi-Agent RAG System with Retriever, Analyzer, and Answer Generator."""
    def __init__(self, vector_db: VectorDBManager):
        self.vector_db = vector_db
        self.llm_model = 'llama3' # Using a local LLM model i downloaded with Ollama for all agents

        # Define agents
        self.workflow = self._build_graph()
        # Compile the workflow
        self.app = self.workflow.compile()


    def _build_graph(self) -> StateGraph:
        """Builds the state graph for the multi-agent WorkFlow"""
        workflow = StateGraph(AgentState)
        # Define states and transitions

        #Add agents as nodes
        workflow.add_node("Retriever", self.retriever)
        workflow.add_node("Analyzer", self.analyzer)
        workflow.add_node("AnswerGenerator", self.answer_generator)

        # Define Execution flow:
        workflow.set_entry_point("Retriever")
        workflow.add_edge("Retriever", "Analyzer")
        workflow.add_edge("Analyzer", "AnswerGenerator")
        workflow.add_edge("AnswerGenerator", END)

        return workflow
    
    def retriever(self, state: AgentState) -> AgentState:
        """Retriever: Retrieves relevant documents from the vector DB.
           this time a real agent (that analyzes the intent) for retrieval"""
        # Get the query from the state
        query = state["query"]

        # First, determine the intent of the query using LLM
        # pull the prompt from prompts.py
        intent_promt = prompts.retriever_intent_prompt.format(query=query)

        intent_response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": "Your task is to classify query intent"},
                      {"role": "user", "content": intent_promt}],
            stream=False
        )
        # Get the intent from the response
        query_intent = intent_response['message']['content'].strip().upper()
        state["query_intent"] = query_intent
        print(f"Determined Query Intent: {query_intent}")
        # Retrieve documents from the vector DB based on the query and intent
        top_k = 5  if query_intent in ["EXPLANATION", "COMPARISON"] else 3
        results = self.vector_db.search(query, top_k=top_k)

        retrieved_docs = []
        for i , (distance, doc, metadata) in enumerate(results):
            retrieved_docs.append({
                "rank": i + 1,
                "content": doc.strip(),# the retrieved document content
                "similarity": 1 - distance,# the similarity score
                "metadata": metadata,
                "relevance": "HIGH" if (1 - distance) > 0.8 else "MEDIUM" if (1 - distance) > 0.5 else "LOW"
            })
            print(f"Retrieved Doc {i+1}: Similarity={1 - distance:.4f}, Source={metadata.get('source', 'N/A')}")

        state["retrieved_docs"] = retrieved_docs
        state["agent_logs"] = state.get("agent_logs", [])
        state["agent_logs"].append(f"Retriever classified intent as {query_intent} and found {len(retrieved_docs)} documents.")

        return state

    def analyzer(self, state: AgentState) -> AgentState:
        """
        Analyzer Agent: Analyzes the retrieved documents.
        Responsibilities:
        - Evaluate quality and relevance of gathered information
        - Assign quality scores
        """
        query = state["query"]
        query_intent = state["query_intent"]
        retrieved_docs = state["retrieved_docs"]

        # quality analysis
        #calculate a quality score based on relevance
        high_relevance = sum(1 for doc in retrieved_docs if doc["relevance"] == "HIGH")
        total_docs = len(retrieved_docs)
        quality_score = high_relevance / total_docs if total_docs > 0 else 0

        state["quality_score"] = quality_score
        print(f" Quality Score: {quality_score:.2f}({high_relevance}/{total_docs} highly relevant)")

        #facrt verification
        docs_text = "\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(retrieved_docs)])

        analysis_prompt = prompts.analyzer_verification_prompt.format(
            query=query,
            query_intent=query_intent,
            docs_text=docs_text
        )
        
        analysis_response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": "Your task is to be a critical fact-checker. Be thorough and skeptical."},
                      {"role": "user", "content": analysis_prompt}],
            stream=False
        )

        verified_contents = analysis_response['message']['content']

        #parse the verified facts
        if "INSUFFICIENT_DATA" in verified_contents:
            verified_facts = ["Insufficient data to verify facts."]
            print("Analyzer found insufficient data to verify facts.")
        else:
            verified_facts = [line.strip() for line in verified_contents.split('\n') if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•') or line.strip()[0].isdigit())]
            print("Analyzer found the following verified facts:")
            for fact in verified_facts:
                print(f" - {fact}")

        state["verified_facts"] = verified_facts
        state["agent_logs"].append(f"Analyzer assigned quality score of {quality_score:.2f} and verified {len(verified_facts)} facts.")

        return state

    def answer_generator(self, state: AgentState) -> AgentState:
        """
        Answer Generator Agent: Generates the final answer based on analysis.
        Responsibilities:
        - Synthesize information from analysis
        - Provide well-structured responses
        """
        # Get necessary info from state
        query = state["query"]
        query_intent = state["query_intent"]
        verified_facts = state["verified_facts"]
        quality_score = state["quality_score"]
        retrieved_docs = state["retrieved_docs"]

        #style mapping based on intent
        style_mapping = {
            "DEFINITION": "concise and clear",
            "EXPLANATION": "detailed and thorough",
            "COMPARISON": "structured and comparative",
            "FACT": "precise and factual",
            "LIST": "organized and enumerated"
        }

        answer_style = style_mapping.get(query_intent, "clear and informative")

        if verified_facts == ["Insufficient data to verify facts."]:
            final_answer = "I'm sorry, but there is insufficient data..."
            state["final_answer"] = final_answer
            state["agent_logs"].append(f"Answer Generator: Insufficient data (quality score {quality_score:.2f}).")
            print("Generated Final Answer (Insufficient Data).")
            return state
        
        facts_text = "\n".join(verified_facts)
        docs_context = "\n".join([f"- {doc['content']}" for doc in retrieved_docs[:3]])

        writing_prompt = prompts.answer_generator_prompt.format(
            query=query,
            query_intent=query_intent,
            answer_style=answer_style,
            verified_facts=facts_text,
            docs_context=docs_context
        )

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": f"You are a {answer_style} writer. Create clear, accurate answers."},
                      {"role": "user", "content": writing_prompt}],
            stream=False
        )

        final_answer = response['message']['content'].strip()
        state["final_answer"] = final_answer
        state["agent_logs"].append(f"Answer Generator produced the final answer with quality score {quality_score:.2f}.")
        print("Generated Final Answer.")
        return state

    def process_query(self, query: str) -> Dict:
        """Processes a user query through the multi-agent pipeline"""
        initial_state: AgentState = {
        "query": query,
        "query_intent": "",  
        "retrieved_docs": [],
        "quality_score": 0.0, 
        "verified_facts": [],  
        "agent_logs": [],
        "final_answer": ""
    }
        # Execute the workflow
        final_state = self.app.invoke(initial_state)
        print("Pipeline completed.")
        return final_state

# Quick test
if __name__ == "__main__":
    #test the agents system
    db = VectorDBManager()

    if db.get_collection_count() == 0:
        print("No documents found in the database. Try running setup_db.py to add sample documents.")
    else:
        rag = MultiAgentRAG(db)

        #test query
        result = rag.process_query("What is the biggest cat species or breed?")

        print("\n FINAL RESULT:")
        print(f"\nAnswer: {result['final_answer']}")
        print(f"\nAgent Logs:")
        for log in result['agent_logs']:
            print(f"  - {log}")