from typing import Dict, List, Optional
from .base_agent import BaseAgent
from . import prompts


class AnalyzerAgent(BaseAgent):
    """
    Analyzes retrieved evidence, verifies facts, calculates quality,
    and produces a MITRE ATT&CK assessment.
    """

    def __init__(self, llm_model: str, mitre_kb=None):
        super().__init__(llm_model)
        self.mitre_kb = mitre_kb   # MitreKnowledgeBase instance 

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def calculate_quality_score(self, retrieved_docs: List[Dict]) -> float:
        if not retrieved_docs:
            return 0.0
        weight_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
        weighted = sum(weight_map.get(d["relevance"], 0.3) for d in retrieved_docs)
        avg_sim = sum(d["similarity"] for d in retrieved_docs) / len(retrieved_docs)
        return round(0.7 * (weighted / len(retrieved_docs)) + 0.3 * avg_sim, 3)

    # ------------------------------------------------------------------
    # Fact verification
    # ------------------------------------------------------------------

    def verify_facts(self, query: str, retrieved_docs: List[Dict]) -> List[str]:
        """Use the LLM to extract verified facts from retrieved evidence."""
        docs_text = "\n".join(
            f"{i+1}. [{d.get('source','?')}] {d['content'][:300]}"
            for i, d in enumerate(retrieved_docs)
        )

        verification_prompt = prompts.analyzer_verification_prompt.format(
            query=query,
            docs_text=docs_text
        )

        response = self.call_llm([
            {"role": "system", "content": "You are a cybersecurity threat analyst. Be precise and evidence-based."},
            {"role": "user", "content": verification_prompt}
        ])

        
        content = response["message"]["content"]

        if "INSUFFICIENT_DATA" in content.upper():
            self.log("Insufficient evidence to verify facts.")
            return ["INSUFFICIENT_DATA"]

        # Parse bullet-point facts
        facts = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("-", "•", "*")) or (line[0].isdigit() and line[1:3] in (". ", ") ")):
                cleaned = line.lstrip("-•*0123456789.) ").strip()
                if cleaned:
                    facts.append(cleaned)

        if not facts and content.strip():
            facts = [l.strip() for l in content.splitlines() if len(l.strip()) > 15]

        if not facts:
            return ["INSUFFICIENT_DATA"]

        self.log(f"Verified {len(facts)} facts.")
        return facts

    # ------------------------------------------------------------------
    # MITRE ATT&CK assessment
    # ------------------------------------------------------------------

    def build_mitre_assessment(self, retrieved_docs: List[Dict], query: str) -> str:
        """
        Collect all flags from inline log docs + MITRE technique docs,
        then build a structured ATT&CK assessment string.
        """
        if self.mitre_kb is None:
            return "MITRE ATT&CK analysis unavailable (no knowledge base loaded)."

        # Gather flags from inline log events
        all_flags = []
        for doc in retrieved_docs:
            all_flags.extend(doc.get("flags", []))

        # Also pull flags from vector DB log_event results
        for doc in retrieved_docs:
            if doc.get("metadata", {}).get("type") == "log_event":
                flag_str = doc["metadata"].get("flags", "")
                if flag_str:
                    all_flags.extend(flag_str.split(", "))

        all_flags = list(set(all_flags))  # deduplicate

        if not all_flags:
            # Fall back to keyword search against the query
            keyword_techniques = self.mitre_kb.search_by_keyword(query.split()[0] if query else "attack")
            if keyword_techniques:
                lines = [
                    f"  • {t['id']} - {t['name']} [{', '.join(t['tactics'])}]"
                    for t in keyword_techniques[:3]
                ]
                return "Possible relevant techniques (keyword match):\n" + "\n".join(lines)
            return "No specific MITRE techniques identified from available evidence."

        mitre_summary = self.mitre_kb.get_mitre_summary_for_flags(all_flags)

        # Use LLM to produce a concise analyst interpretation
        llm_response = self.call_llm([
            {"role": "system", "content": "You are a cybersecurity expert specialising in MITRE ATT&CK."},
            {"role": "user", "content": prompts.analyzer_mitre_prompt.format(
                flags=", ".join(all_flags),
                mitre_summary=mitre_summary
            )}
        ])
        llm_interpretation = llm_response["message"]["content"].strip()

        return f"{mitre_summary}\n\nAnalyst Interpretation:\n{llm_interpretation}"

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def process(self, state: Dict) -> Dict:
        query = state["query"]
        retrieved_docs = state.get("retrieved_docs", [])

        # Quality score
        quality_score = self.calculate_quality_score(retrieved_docs)
        state["quality_score"] = quality_score
        self.log(f"Quality score: {quality_score:.3f}")

        # Fact verification
        verified_facts = self.verify_facts(query, retrieved_docs)
        state["verified_facts"] = verified_facts

        # MITRE assessment
        mitre_assessment = self.build_mitre_assessment(retrieved_docs, query)
        state["mitre_assessment"] = mitre_assessment
        self.log("MITRE assessment complete.")

        state.setdefault("agent_logs", [])
        state["agent_logs"].append(
            f"Analyzer: quality={quality_score:.2f} | "
            f"facts={len(verified_facts)} | "
            f"mitre={'yes' if mitre_assessment else 'no'}"
        )
        state.setdefault("reasoning_steps", [])
        state["reasoning_steps"].append(
            f"OBSERVATION: Analysis complete — {len(verified_facts)} facts verified, "
            f"quality={quality_score:.2f}"
        )

        return state