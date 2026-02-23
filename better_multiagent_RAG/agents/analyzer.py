from typing import Any, Dict, List
from .base_agent import BaseAgent
from . import prompts

class AnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing retrieved information and formulating a response."""
    def __init__(self, llm_model: str):
        super().__init__(llm_model)

    def calculate_quality_score(self, retrieved_docs: List[Dict]) -> float:
        """Calculate a quality score based on relevance"""
        if not retrieved_docs:
            return 0.0

        # total_docs = len(retrieved_docs)
        weighted_score = sum((1 if doc["relevance"] == "HIGH" else 0.6 if doc["relevance"] == "MEDIUM" else 0.3) for doc in retrieved_docs) / len(retrieved_docs)
        avg_similarity = sum(doc["similarity"] for doc in retrieved_docs) / len(retrieved_docs)

        return 0.7 * weighted_score + 0.3 * avg_similarity

    def verify_facts(self, query: str, retrieved_docs: List[Dict]) -> list[str]:
        """Verify the facts in the retrieved documents using LLM"""

        docs_text = "\n".join([f"{i+1}. {doc['content']}" for i, doc in enumerate(retrieved_docs)])

        verification_prompt = prompts.analyzer_verification_prompt.format(query=query, docs_text=docs_text)

        response = self.call_llm([
            {"role": "system", "content": "You are a fact-checker. Be thorough and skeptical."},
            {"role": "user", "content": verification_prompt}
        ])

        verified_contents = response['message']['content']

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

        return verified_facts
    
    def process(self, state: Dict) -> Dict:
        """Analyze retrieved information and prepare for answer generation."""
        query = state["query"]
        retrieved_docs = state["retrieved_docs"]

        # Calculate quality score
        quality_score = self.calculate_quality_score(retrieved_docs)
        state["quality_score"] = quality_score
        self.log(f"Calculated quality score: {quality_score:.2f}")

        # Verify facts
        verified_facts = self.verify_facts(query, retrieved_docs)
        state["verified_facts"] = verified_facts

        if verified_facts == ["INSUFFICIENT_DATA"]:
            self.log("Insufficient data to verify facts")
        
        state["agent_logs"].append(
            f"Analyzer: Quality {quality_score:.2f}, "
            f"verified {len(verified_facts)} facts"
        )
        self.log(f"Analyzer verified facts")
        return state
