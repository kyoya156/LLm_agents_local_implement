from typing import Dict, List
from .base_agent import BaseAgent
from . import prompts


class RetrieverAgent(BaseAgent):
    """
    Retrieves relevant information from TWO sources:
      1. ChromaDB — contains ingested CTI reports, MITRE techniques, and parsed log events
      2. Inline log text — if the user pastes raw logs in the query, parse them on the fly
    """

    # How many docs to retrieve per query intent
    SEARCH_STRATEGIES = {
        "LOG_ANALYSIS":       6,
        "THREAT_INTEL":       5,
        "INCIDENT_RESPONSE":  5,
        "ANOMALY_DETECTION":  4,
        "GENERAL":            3,
    }

    def __init__(self, llm_model: str, memory_manager, vector_db, log_parser=None):
        super().__init__(llm_model)
        self.memory = memory_manager
        self.vector_db = vector_db
        self.log_parser = log_parser   # optional LogParser instance

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> str:
        intent_prompt = prompts.retriever_intent_prompt.format(query=query)
        response = self.call_llm([
            {"role": "system", "content": "You are a cybersecurity query classifier. Reply with ONE phrase only."},
            {"role": "user", "content": intent_prompt}
        ])
        # FIX: response['message'] not response['messages']
        raw = response["message"]["content"].strip().upper()

        for intent in self.SEARCH_STRATEGIES:
            if intent in raw:
                return intent

        self.log(f"Unrecognised intent '{raw}', defaulting to GENERAL.")
        return "GENERAL"

    # ------------------------------------------------------------------
    # Memory strategy
    # ------------------------------------------------------------------

    def should_use_memory(self, query: str) -> bool:
        """Check if prior context is sufficient to answer without vector search."""
        # FIX: signature now only takes query; memory_context fetched internally
        memory_context = self.memory.get_context_for_query(query)
        if not memory_context or memory_context == "No prior context.":
            return False

        decision_prompt = prompts.retriever_decision_prompt.format(
            query=query,
            memory_context=memory_context
        )
        response = self.call_llm([
            {"role": "system", "content": "You decide information retrieval strategy."},
            {"role": "user", "content": decision_prompt}
        ])
        # FIX: response['message'] not response['messages']
        return response["message"]["content"].strip().lower() == "use_memory"

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def document_relevance(self, similarity: float) -> str:
        if similarity >= 0.65:
            return "HIGH"
        elif similarity >= 0.40:
            return "MEDIUM"
        else:
            return "LOW"

    # ------------------------------------------------------------------
    # Inline log parsing
    # ------------------------------------------------------------------

    def _extract_inline_logs(self, query: str) -> List[Dict]:
        """
        If the query contains pasted log lines, parse them and return
        as retrieved_docs entries so the analyzer can process them.
        """
        if self.log_parser is None:
            return []

        # Heuristic: query contains log-like lines if it has timestamps or known keywords
        log_keywords = ["failed password", "accepted", "sudo", "sshd", "kernel", "iptables",
                        "jan ", "feb ", "mar ", "apr ", "may ", "jun ",
                        "jul ", "aug ", "sep ", "oct ", "nov ", "dec "]
        query_lower = query.lower()
        if not any(kw in query_lower for kw in log_keywords):
            return []

        self.log("Detected inline log content — parsing...")
        events = self.log_parser.parse_lines(query)
        suspicious = self.log_parser.get_suspicious_events(events)

        if not suspicious:
            return []

        docs = []
        for i, event in enumerate(suspicious):
            docs.append({
                "rank": i + 1,
                "content": (
                    f"[INLINE LOG - {event['severity']}] "
                    f"Flags: {', '.join(event['flags'])} | "
                    f"IPs: {', '.join(event['ips']) or 'N/A'} | "
                    f"User: {event['user'] or 'N/A'} | "
                    f"Raw: {event['raw'][:200]}"
                ),
                "similarity": 1.0,          # inline logs are maximally relevant
                "metadata": {
                    "type": "inline_log",
                    "severity": event["severity"],
                    "flags": ", ".join(event["flags"]),
                },
                "relevance": "HIGH",
                "source": "inline_log",
                "flags": event["flags"],
            })
        self.log(f"Extracted {len(docs)} suspicious inline log events.")
        return docs

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def process(self, state: Dict) -> Dict:
        query = state["query"]

        # 1. Classify intent
        query_intent = self.classify_intent(query)
        state["query_intent"] = query_intent
        self.log(f"Intent: {query_intent}")

        # 2. Check if memory alone is sufficient
        use_memory = self.should_use_memory(query)
        self.log(f"Memory sufficient: {use_memory}")

        # 3. Only parse inline logs for log-related intents — not for response/intel queries
        inline_docs = []
        if query_intent in ("LOG_ANALYSIS", "ANOMALY_DETECTION"):
            inline_docs = self._extract_inline_logs(query)

        # 4. If memory is sufficient AND no inline logs, answer from memory only
        if use_memory and not inline_docs:
            memory_context = self.memory.get_context_for_query(query)
            self.log("Answering from memory — skipping vector DB search.")
            state["retrieved_docs"] = [{
                "rank": 1,
                "content": memory_context,
                "similarity": 1.0,
                "metadata": {"type": "memory"},
                "relevance": "HIGH",
                "source": "memory",
                "flags": [],
            }]
            state.setdefault("reasoning_steps", [])
            state["reasoning_steps"].append(
                "OBSERVATION: Query answered from session memory — no vector search needed."
            )
            state.setdefault("agent_logs", [])
            state["agent_logs"].append(
                f"Retriever: memory-only | intent={query_intent}"
            )
            return state

        # 5. Vector DB search
        top_k = self.SEARCH_STRATEGIES.get(query_intent, 4)
        raw_results = self.vector_db.search(query, top_k=top_k)

        # 6. Build retrieved_docs — inline logs first, then filtered vector results
        retrieved_docs = list(inline_docs)
        offset = len(retrieved_docs)

        # For INCIDENT_RESPONSE and THREAT_INTEL, skip raw log_event docs —
        # they add noise and cause the LLM to re-analyse logs instead of giving guidance
        skip_types = set()
        if query_intent in ("INCIDENT_RESPONSE", "THREAT_INTEL"):
            skip_types = {"log_event", "inline_log"}

        skipped = 0
        for i, (distance, doc, metadata) in enumerate(raw_results):
            doc_type = metadata.get("type", "unknown")
            if doc_type in skip_types:
                skipped += 1
                continue

            similarity = max(0.0, 1.0 - distance)
            relevance = self.document_relevance(similarity)
            retrieved_docs.append({
                "rank": offset + i + 1,
                "content": doc.strip(),
                "similarity": similarity,
                "metadata": metadata,
                "relevance": relevance,
                "source": doc_type,
                "flags": [],
            })
            self.log(f"Doc {len(retrieved_docs)}: sim={similarity:.3f} | {relevance} | {doc_type}")

        if skipped:
            self.log(f"Skipped {skipped} raw log docs (not relevant for {query_intent}).")

        # 7. Always inject memory context so the answer generator can reference prior incidents
        memory_context = self.memory.get_context_for_query(query)
        if memory_context != "No prior context.":
            retrieved_docs.append({
                "rank": len(retrieved_docs) + 1,
                "content": f"[PRIOR INCIDENT CONTEXT]\n{memory_context}",
                "similarity": 0.95,
                "metadata": {"type": "memory"},
                "relevance": "HIGH",
                "source": "memory",
                "flags": [],
            })
            self.log("Injected prior memory context as additional document.")

        state["retrieved_docs"] = retrieved_docs

        high_count = sum(1 for d in retrieved_docs if d["relevance"] == "HIGH")
        state.setdefault("reasoning_steps", [])
        state["reasoning_steps"].append(
            f"OBSERVATION: Retrieved {len(retrieved_docs)} docs "
            f"({high_count} HIGH relevance, {len(inline_docs)} inline logs, "
            f"{skipped} log docs skipped for {query_intent}, "
            f"memory={'injected' if memory_context != 'No prior context.' else 'none'})"
        )
        state.setdefault("agent_logs", [])
        state["agent_logs"].append(
            f"Retriever: {len(retrieved_docs)} docs | intent={query_intent} | "
            f"skipped={skipped} log docs | memory={use_memory}"
        )

        self.log(f"Retrieval complete: {len(retrieved_docs)} total sources")
        return state