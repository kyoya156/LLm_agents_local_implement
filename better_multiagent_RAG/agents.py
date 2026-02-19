"""
Multi-Agent RAG System
Three agents: Retriever -> Analyzer -> Answer Generator
This time adds memory and reaction capabilities.
"""
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
import ollama, prompts, json, os
from vector_db import VectorDBManager
from datetime import datetime
from memory import MemoryManager

# Define the state flow between agents
class AgentState(TypedDict):
    # agents fields
    query: str
    query_intent: str
    retrieved_docs: List[Dict]
    quality_score: float # between 0 and 1
    verified_facts: List[str]  # Verified information
    agent_logs: List[str]
    final_answer: str

    # Memory fields
    conversation_history: List[Dict]  # Short-term: Recent conversation
    conversation_summary: str  # Short-term: Summary of conversation
    user_preferences: Dict  # Long-term: User preferences
    learned_facts: List[Dict]  # Long-term: Facts learned about user

    # REACT fields
    reasoning_steps: List[str]  # Reasoning steps 
    actions_taken: List[Dict]  # Actions tracking
    should_continue: bool  # Flag to indicate if the process should continue or stop

class MultiAgentRAG:
    """Multi-Agent RAG System with Retriever, Analyzer, and Answer Generator."""
    def __init__(self, vector_db: VectorDBManager, user_id: str = "default_user"):
        self.vector_db = vector_db
        self.llm_model = 'llama3' # Using a local LLM model i downloaded with Ollama for all agents
        self.memory = MemoryManager(user_id)

        # Define agents
        self.workflow = self._build_graph()
        # Compile the workflow
        self.app = self.workflow.compile()


    def _build_graph(self) -> StateGraph:
        """Builds the state graph for the multi-agent WorkFlow"""
        workflow = StateGraph(AgentState)
        # Define states and transitions

        #Add agents as nodes
        workflow.add_node("planner", self.planner) # entry point
        workflow.add_node("Retriever", self.retriever)
        workflow.add_node("Analyzer", self.analyzer)
        workflow.add_node("AnswerGenerator", self.answer_generator)
        workflow.add_node("memory_agent", self.memory_agent) # for handling memory updates and retrieval

        # Define Execution flow:
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "Retriever")
        workflow.add_edge("Retriever", "Analyzer")
        workflow.add_edge("Analyzer", "AnswerGenerator")
        workflow.add_edge("AnswerGenerator", "memory_agent")
        workflow.add_edge("memory_agent", END) # end of the workflow

        return workflow
    
    def planner(self, state: AgentState) -> AgentState:
        """Planner: Orchestrates the workflow and manages state transitions.
        
        REACT:
        1. THOUGHT: What do I need to do?
        2. ACTION: What action should I take?
        3. OBSERVATION: What did I learn?
        """
        query = state["query"]
        #memory context retrieval
        memory_context = self.memory.get_context_for_query()
        
        planner_prompt = prompts.planner_prompt.format(query=query, memory_context=memory_context)

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": "You are a strategic planner. Think step-by-step to answer this query."},
                      {"role": "user", "content": planner_prompt}],
            stream=False
        )

        reasoning = response['message']['content']

        # Parse the response for thought, action, and plan
        state["reasoning_steps"] = state.get("reasoning_steps", [])
        state["reasoning_steps"].append(reasoning)

        #determine next action based on response
        # some simple action for testing, but could be more complex with multiple actions and conditions
        if "search_knowledge" in reasoning.lower():
            action = "search_knowledge"
        elif "use_memory" in reasoning.lower():
            action = "use_memory"
        else:
            action = "search_knowledge" # default action

        state["actions_taken"] = state.get("actions_taken", [])
        state["actions_taken"].append({
            "agent": "planner",
            "action": action,
            "reasoning": reasoning
        })

        state["agent_logs"] = state.get("agent_logs", [])
        state["agent_logs"].append(f"Planner decided to {action}.")

    def retriever(self, state: AgentState) -> AgentState:
        """Retriever: Retrieves relevant documents from the vector DB.
           this time a real agent (that analyzes the intent) for retrieval"""
        # Get the query from the state
        query = state["query"]

        #check if memory can be used first
        memory_context = self.memory.get_context_for_query()

        #Thought: Should the retriever use memory or search the vector database?

        thought_prompt = prompts.retriever_thought_prompt.format(query=query, memory_context=memory_context)

        thought_response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": thought_prompt}],
            stream=False
        )

        decision = thought_response['message']['content'].strip().lower()
        print(f"Retriever decision: {decision}")

        intent_prompt = prompts.retriever_intent_prompt.format(query=query)

        intent_response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": intent_prompt}],
            stream=False
        )

        query_intent = intent_response['message']['content'].strip().upper()
        state["query_intent"] = query_intent

        #action  search knowledge base
        top_k = 5 if query_intent in ["EXPLANATION", "COMPARISON"] else 3 # adjust number of retrieved docs based on intent
        results = self.vector_db.search(query, top_k=top_k)

        retrieved_docs = []
        for i , (distance, doc, metadata) in enumerate(results):
            retrieved_docs.append({
                "rank": i + 1,
                "content": doc.strip(),# the retrieved document content
                "similarity": 1 - distance,# the similarity score
                "metadata": metadata,
                "relevance": "HIGH" if (1 - distance) > 0.6 else "MEDIUM" if (1 - distance) > 0.4 else "LOW"
            })
            print(f"Retrieved Doc {i+1}: Similarity={1 - distance:.4f}, Source={metadata.get('source', 'N/A')}")

        state["retrieved_docs"] = retrieved_docs
        state["agent_logs"].append(f"Retriever: Found {len(retrieved_docs)} docs (Intent: {query_intent}, Memory used: {memory_context != 'No prior context.'})")
        #recording an observation about the relevance of retrieved documents 
        state["reasoning_steps"].append(
            f"OBSERVATION: Found {len([d for d in retrieved_docs if d['relevance'] == 'HIGH'])} highly relevant documents"
        )
        return state

    def analyzer(self, state: AgentState) -> AgentState:
        """
        Analyzer Agent: Analyzes the retrieved documents.
        Responsibilities:
        - Evaluate quality and relevance of gathered information
        - Assign quality scores
        """
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]

        # quality analysis
        # calculate weighted score and average similarity
        # then combine them into a final quality score between 0 and 1
        if total_docs := len(retrieved_docs) > 0:
            weighted_score = sum((1 if doc["relevance"] == "HIGH" else 0.6 if doc["relevance"] == "MEDIUM" else 0.3) for doc in retrieved_docs) / total_docs
            avg_similarity = sum(doc["similarity"] for doc in retrieved_docs) / total_docs

            # combine relevance and similarity into a quality score
            quality_score = (weighted_score * 0.7) + (avg_similarity * 0.3)
        else:
            quality_score = 0.0
        
        high_relevance =  sum(1 for doc in retrieved_docs if doc['relevance'] == 'HIGH')

        state["quality_score"] = quality_score
        high_relevance = sum(1 for doc in retrieved_docs if doc["relevance"] == "HIGH")
        print(f" Quality Score: {quality_score:.2f} ({high_relevance} HIGH, avg similarity: {avg_similarity:.2f})")

        # fact verification
        docs_text = "\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(retrieved_docs)])

        analysis_prompt = prompts.analyzer_verification_prompt.format(
            query=query,
            docs_text=docs_text
        )
        
        analysis_response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": analysis_prompt}],
            stream=False
        )

        verified_contents = analysis_response['message']['content']

        #parse the verified facts
        if "INSUFFICIENT_DATA" in verified_contents.upper():
            verified_facts = ["Insufficient data to verify facts."]
            print("Analyzer found insufficient data to verify facts.")
        else:
            # Parse facts - look for bullet points or numbered items
            lines = verified_contents.split('\n')
            verified_facts = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line starts with bullet/number markers
                if (line.startswith('-') or 
                    line.startswith('•') or 
                    line.startswith('*') or
                    (len(line) > 0 and line[0].isdigit() and ('. ' in line[:4] or ') ' in line[:4]))):
                    # Clean up the marker
                    cleaned = line.lstrip('-•*0123456789.) ').strip()
                    if cleaned:
                        verified_facts.append(cleaned)
            
            # If no bullet points found, but there's content, treat each non-empty line as a fact
            if not verified_facts and verified_contents.strip():
                verified_facts = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
            
            if verified_facts:
                print(f"Analyzer found {len(verified_facts)} verified facts:")
                for fact in verified_facts:
                    print(f" - {fact}")
            else:
                # Fallback if parsing fails
                verified_facts = ["Insufficient data to verify facts."]
                print("Analyzer could not parse facts from response.")
        
        state["verified_facts"] = verified_facts
        state["agent_logs"].append(f"Analyzer assigned a quality score of {quality_score:.2f} and extracted {len(verified_facts)} verified facts.")
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
        # memory context
        memory_context = self.memory.get_context_for_query()

        #style mapping based on intent
        style_mapping = {
            "DEFINITION": "concise and clear",
            "EXPLANATION": "detailed and thorough",
            "COMPARISON": "structured and comparative",
            "FACT": "precise and factual",
            "LIST": "organized and enumerated",
            "PERSONAL": "friendly and personalized"
        }

        answer_style = style_mapping.get(query_intent, "clear and informative")

        if verified_facts == ["Insufficient data to verify facts."]:
            final_answer = "I don't have sufficient information in my knowledge base to answer that question accurately."
            state["final_answer"] = final_answer
            state["agent_logs"].append(f"Answer Generator: Insufficient data (quality score {quality_score:.2f}).")
            print("Generated Final Answer (Insufficient Data).")
            return state
        
        facts_text = "\n".join(verified_facts)

        writing_prompt = prompts.answer_generator_prompt.format(
            query=query,
            query_intent=query_intent,
            answer_style=answer_style,
            quality_score=quality_score,
            memory_context=memory_context,
            facts_text=facts_text
        )

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": writing_prompt}],
            stream=False
        )

        final_answer = response['message']['content']
        state["final_answer"] = final_answer
        state["agent_logs"].append(f"Answer Generator: Quality score {quality_score:.2f} with {answer_style} answer.")
        print("Generated Final Answer.")
        return state

    def memory_agent(self, state: AgentState) -> AgentState:
        """Memory Agent: Manages memory updates """
        # For this simple implementation, we'll just update the memory with the latest conversation
        query = state["query"]
        final_answer = state["final_answer"]
        query_intent = state["query_intent"]

        #update short-term memory (conversation history and summary)
        self.memory.add_to_short_term("user", query)
        self.memory.add_to_short_term("assistant", final_answer)

        #summarize conversation history every 3 interactions(3 user queries + 3 assistant responses = 6 entries in short-term memory)
        if len(self.memory.short_term_memory["conversation_history"]) % 6 == 0:
            summary = self.memory.summarize_conversation()

        #update long-term memory with any new facts learned about the user
        # added some words to catch more preference-related queries as the ollama 3 model might not be that good at classifying intent
        if query_intent == "PERSONAL" or "i like" in query.lower() or "my preference" in query.lower():
            self.memory.add_learned_fact(fact=query, category="preference")

        extract_facts_prompt = prompts.extract_facts_prompt.format(
            query=query,
            final_answer=final_answer
        )

        respond = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": extract_facts_prompt}],
            stream=False
        )

        extraction = respond['message']['content']

        if "FACT:" in extraction:
                try:
                    fact = extraction.split("FACT:")[1].split("|")[0].strip()
                    category = extraction.split("CATEGORY:")[1].strip()
                    self.memory.add_learned_fact(fact, category)
                    print(f"      Learned: {fact} ({category})")
                except:
                    print("      Failed to parse learned fact from response.")
                    pass
        state["conversation_history"] = self.memory.short_term_memory["conversation_history"]
        state["conversation_summary"] = self.memory.short_term_memory["conversation_summary"]
        state["user_preferences"] = self.memory.user_preferences
        state["learned_facts"] = self.memory.learned_facts
        
        state["agent_logs"].append(
            f"Memory: Updated short-term ({len(self.memory.short_term_memory['conversation_history'])} messages) "
            f"and long-term ({len(self.memory.learned_facts)} facts)"
        )

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
        "final_answer": "",
        "conversation_history": self.memory.short_term_memory["conversation_history"],
        "conversation_summary": self.memory.short_term_memory["conversation_summary"],
        "user_preferences": self.memory.user_preferences,
        "learned_facts": self.memory.learned_facts,
        "reasoning_steps": [],
        "actions_taken": [],
        "should_continue": True
    }
        # Execute the workflow
        final_state = self.app.invoke(initial_state)
        print("Pipeline completed.")
        return final_state

# Quick test
if __name__ == "__main__":
    from vector_db import VectorDBManager
    
    db = VectorDBManager()
    
    if db.get_collection_count() == 0:
        print("No documents in database. Please run setup.py first!")
    else:
        # Create RAG system
        rag = MultiAgentRAG(db, user_id="test_user")
        
        # Test conversation with memory

        print("Testing Memory and ReAct Pattern")

        
        # First query
        result1 = rag.process_query("How many hours do cats sleep?")
        print(f"\n Answer 1: {result1['final_answer']}\n")
        
        # Second query (should use memory from first)
        result2 = rag.process_query("Why do they sleep so much?")
        print(f"\n Answer 2: {result2['final_answer']}\n")
        
        # Show memory

        print("💾 MEMORY STATE")
        print(f"Conversation history: {len(rag.memory.short_term_memory['conversation_history'])} messages")
        print(f"Summary: {rag.memory.short_term_memory['conversation_summary']}")
        print(f"Learned facts: {len(rag.memory.learned_facts)}")
