from typing import Dict
from .base_agent import BaseAgent
from memory import MemoryManager
from . import prompts

class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving information from memory."""
    def __init__(self, llm_model: str, memory_manager: MemoryManager, vector_db):
        super().__init__(llm_model)
        self.memory = memory_manager
        self.vector_db = vector_db

        # Intent-based search strategies
        self.search_strategies = {
            "DEFINITION": 3,      # Need precise info
            "EXPLANATION": 5,     # Need coverage
            "COMPARISON": 5,      # Need multiple perspectives
            "FACT": 3,           # Need specific info
            "LIST": 4,           # Need variety
            "PERSONAL": 3        # Need targeted info
        }

    def classify_intent(self, query: str) -> str:
        """Classify the intent of the query to determine search strategy."""
        intent_prompt = prompts.retriever_intent_prompt.format(query=query)
        response = self.call_llm([
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": intent_prompt}
        ])
        intent = response['message']['content'].strip().upper()
        if intent in self.search_strategies:
            return intent
        else:
            print(f"Warning: Unrecognized intent '{intent}' classified. Defaulting to FACT.")
            return "FACT"  # Default to FACT if unclear

    def should_use_memory(self, query: str, memory_context: str) -> bool:
        """Determine if the query can be answered from memory alone or if a knowledge search is needed."""
        if memory_context == "No prior context.":
            return False
        
        decision_prompt = prompts.retriever_decision_prompt.format(query=query, memory_context=memory_context)
        response = self.call_llm([
            {"role": "system", "content": "You decide information retrieval strategy."},
            {"role": "user", "content": decision_prompt}
        ])

        return response['message']['content'].strip().lower() == "use_memory"

    def document_relevance(self, similarity: float) -> str:
        """Assign relevance level based on similarity score"""
        if similarity >= 0.6:
            return "HIGH"
        elif similarity >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def process(self, state: Dict) -> Dict:
        """Retrieve relevant information from memory based on the query."""
        query = state["query"]

        # Get memory context
        memory_context = self.memory.get_context_for_query()

        # classify intent 
        query_intent = self.classify_intent(query)
        state["query_intent"] = query_intent
        self.log(f"Intent: {query_intent}")

        # Determine if we can use memory
        use_memory = self.should_use_memory(query, memory_context)
        self.log(f"Strategy: {'Use memory' if use_memory else 'Search knowledge base'}")

        # Get search count based on intent
        top_k = self.search_strategies.get(query_intent, 3)  # Default to 3 if intent not recognized
        results = self.vector_db.search(query, top_k=top_k)

        # process results with relevance scoring
        retrieved_docs = []
        for i, (distance, doc, metadata) in enumerate(results):
            similarity = 1 - distance
            relevance = self.document_relevance(similarity)

            retrieved_docs.append({
                "rank": i + 1,
                "content": doc.strip(),
                "similarity": similarity,
                "metadata": metadata,
                "relevance": relevance
            })

            self.log(f"Doc {i+1}: Similarity={similarity:.4f} | Relevance={relevance}")

        state["retrieved_docs"] = retrieved_docs
        state["agent_logs"].append(
            f"Researcher: Found {len(retrieved_docs)} docs "
            f"(Intent: {query_intent}, Memory used: {use_memory})"
        )
        
        # record 
        high_relevance_count = sum(1 for doc in retrieved_docs if doc['relevance'] == 'HIGH')
        state["reasoning_steps"].append(
            f"OBSERVATION: Found {high_relevance_count}/{len(retrieved_docs)} highly relevant documents"
        )
        
        self.log(f"Retrieval complete: {len(retrieved_docs)} sources gathered")
        return state

if __name__ == "__main__":
    # Quick test
    from vector_db import VectorDBManager
    from memory import MemoryManager
    
    db = VectorDBManager()
    memory = MemoryManager()
    
    retriever = RetrieverAgent(
        llm_model='ollama3',
        vector_db=db,
        memory_manager=memory
    )
    
    test_state = {
        "query": "How do cats communicate?",
        "reasoning_steps": [],
        "agent_logs": []
    }
    
    result = retriever.process(test_state)
    print(f"\n Test complete")
    print(f"Intent: {result['query_intent']}")
    print(f"Documents found: {len(result['retrieved_docs'])}")