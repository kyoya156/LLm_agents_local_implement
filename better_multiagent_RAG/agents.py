"""
Multi-Agent RAG System
Three agents: Retriever -> Analyzer -> Answer Generator
"""
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import ollama
from vector_db import VectorDBManager

# Define the state flow between agents
class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Dict]
    analysis: str
    final_answer: str
    agent_logs: List[str]

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
           Not Really an LLM agent just a pipeline step to get relevant docs.:)"""
        # Get the query from the state
        query = state["query"]

        # Search the vector DB
        results = self.vector_db.search(query, top_k=3)

        retrieved_docs = []
        for i , (distance, doc, metadata) in enumerate(results):
            retrieved_docs.append({
                "rank": i + 1,
                "content": doc.strip(),# the retrieved document content
                "similarity": 1 - distance,# the similarity score
                "metadata": metadata,
            })
            print(f"Retrieved Doc {i+1}: Similarity={1 - distance:.4f}, Source={metadata.get('source', 'N/A')}")

        state["retrieved_docs"] = retrieved_docs
        state["agent_logs"] = state.get("agent_logs", [])
        state["agent_logs"].append(f"Retriever found {len(retrieved_docs)} documents.")

        return state
    
    def analyzer(self, state: AgentState) -> AgentState:
        """Analyzer Agent: Analyzes the retrieved documents."""
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]

        # Prepare the analysis prompt
        docs_text = "\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(retrieved_docs)])
        analysis_prompt = f"""Analyze the following documents for relevance to the query: "{query}"
        Documents:
        {docs_text}
        Provide a brief analysis:
        1. Are these documents relevant?
        2. What key information do they contain?
        3. Can they answer the query?

        And overall keep the analysis concise and focused(about 2-3 sentences).
        """

        # Call the LLM for analysis
        respond = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": "Your task is to analyze the retrieved documents for relevance to the query. Provide brief, clear analysis"},
                      {"role": "user", "content": analysis_prompt}],
            #max_tokens=500
            stream=False
        )
        # Get the analysis from the response
        analysis = respond['message']['content']#.strip()
        # Update the state
        state["analysis"] = analysis
        # Log the analysis step
        state["agent_logs"].append("Analyzer completed the analysis.")

        return state
    
    def answer_generator(self, state: AgentState) -> AgentState:
        """Answer Generator Agent: Generates the final answer based on analysis."""
        query = state["query"]
        analysis = state["analysis"]
        retrieved_docs = state["retrieved_docs"]

        # Prepare the answer generation prompt
        docs_text = "\n".join([f"- {doc['content']}" for doc in retrieved_docs])

        answer_prompt = f"""
        Question: {query}

        Analysis:
        {analysis}
        Do not show the analysis in the final answer.

        Information from Documents:
        {docs_text}

        Provide a clear and concise answer based on the most relevant information from the documents.
        If the information is insufficient, state that you cannot provide an answer with the available data.
        Also, if the asked question is outside the scope of the documents, politely decline to answer and do not mention anything else.
        keep the answer concise (about 2-4 sentences).
        """
        # Call the LLM for answer generation
        respond = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": "Your task is to generate a final answer based on the analysis and retrieved documents."},
                      {"role": "user", "content": answer_prompt}],
            stream=False
        )

        # Get the final answer from the response
        final_answer = respond['message']['content']
        state["final_answer"] = final_answer
        state["agent_logs"].append("Answer Generator produced the final answer.")
        print("Answer Generated.")

        return state
    
    def process_query(self, query: str) -> Dict:
        """Processes a user query through the multi-agent pipeline"""
        initial_state: AgentState = {
            "query": query,
            "retrieved_docs": [],
            "analysis": "",
            "final_answer": "",
            "agent_logs": []
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