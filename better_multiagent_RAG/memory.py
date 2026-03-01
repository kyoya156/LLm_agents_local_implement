import ollama, json, os, re
from datetime import datetime
from typing import List, Dict, Optional


class MemoryManager:
    """
    Manages short-term and long-term memory for the cybersecurity agents.
    Short-term: conversation history + summary (resets each session).
    Long-term:  learned incident facts + analyst preferences (persisted to JSON).
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory_dir = "./memory"
        os.makedirs(self.memory_dir, exist_ok=True)
        self.summarize_agent = "llama3"

        # Short-term memory — reset each session
        self.short_term_memory = {
            "conversation_history": [],
            "conversation_summary": ""
        }

        # Long-term memory — persisted per user
        self.user_file = os.path.join(self.memory_dir, f"{user_id}.json")
        self.long_term_memory = {
            "user_preferences": {},
            "learned_facts": []
        }
        self.load_long_term_memory()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_long_term_memory(self):
        """Load long-term memory from the user's JSON file."""
        if os.path.exists(self.user_file):
            with open(self.user_file, "r") as f:
                data = json.load(f)
                self.long_term_memory["user_preferences"] = data.get("user_preferences", {})
                self.long_term_memory["learned_facts"] = data.get("learned_facts", [])
            print(f"[Memory] Loaded {len(self.long_term_memory['learned_facts'])} facts for '{self.user_id}'.")
        else:
            print("[Memory] No existing memory file — starting fresh.")
            self.long_term_memory["user_preferences"] = {}
            self.long_term_memory["learned_facts"] = []

    def save_long_term_memory(self):
        """Persist long-term memory to disk."""
        data = {
            "user_id": self.user_id,
            "user_preferences": self.long_term_memory["user_preferences"],
            "learned_facts": self.long_term_memory["learned_facts"],
            "last_updated": datetime.now().isoformat()
        }
        try:
            with open(self.user_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Memory] Error saving: {e}")

    # ------------------------------------------------------------------
    # Short-term memory
    # ------------------------------------------------------------------

    def add_to_short_term(self, role: str, content: str):
        """Append a message to conversation history (capped at 10 entries)."""
        self.short_term_memory["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only the last 10 messages
        if len(self.short_term_memory["conversation_history"]) > 10:
            self.short_term_memory["conversation_history"] = \
                self.short_term_memory["conversation_history"][-10:]

    def summarize_conversation(self) -> str:
        """Summarise the recent conversation and store it in short-term memory."""
        history = self.short_term_memory["conversation_history"]
        if len(history) < 3:
            return "Conversation too short to summarise."

        recent = history[-6:]
        history_text = "\n".join(
            f"{msg['role']}: {msg['content'][:200]}" for msg in recent
        )

        from agents import prompts
        summary_prompt = prompts.summarize_prompt.format(history_text=history_text)

        response = ollama.chat(
            model=self.summarize_agent,
            messages=[
                {"role": "system", "content": "You summarise cybersecurity incident conversations concisely."},
                {"role": "user", "content": summary_prompt}
            ],
            stream=False
        )

        summary = response["message"]["content"]
        self.short_term_memory["conversation_summary"] = summary
        return summary

    def clear_short_term(self):
        """Reset short-term memory."""
        self.short_term_memory["conversation_history"] = []
        self.short_term_memory["conversation_summary"] = ""

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def add_learned_fact(self, fact: str, category: str):
        """
        Store a new incident fact.
        Avoids storing exact duplicates.
        """
        existing = [f["fact"] for f in self.long_term_memory["learned_facts"]]
        if fact in existing:
            return  # skip duplicate

        self.long_term_memory["learned_facts"].append({
            "fact": fact,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        self.save_long_term_memory()

    def add_attacker_ip(self, ip: str, context: str = ""):
        """Convenience method to store a suspicious IP."""
        fact = f"Suspicious IP: {ip}" + (f" — {context}" if context else "")
        self.add_learned_fact(fact, category="attacker_ip")

    def add_attack_technique(self, technique_id: str, technique_name: str = ""):
        """Convenience method to store a detected MITRE technique."""
        fact = f"Detected technique: {technique_id}" + (f" ({technique_name})" if technique_name else "")
        self.add_learned_fact(fact, category="attack_technique")

    def update_user_preference(self, key: str, value):
        """Update a persistent analyst preference."""
        self.long_term_memory["user_preferences"][key] = value
        self.save_long_term_memory()

    # ------------------------------------------------------------------
    # Context retrieval (query-aware)
    # ------------------------------------------------------------------

    def get_context_for_query(self, query: str = "") -> str:
        """
        Build a context string relevant to the given query by:
          1. Including the conversation summary (short-term)
          2. Including recent conversation messages (short-term)
          3. Filtering learned facts by keyword relevance to the query (long-term)
          4. Falling back to the 5 most recent facts if no keyword match

        Returns "No prior context." if memory is empty.
        """
        context_parts = []
        query_lower = query.lower()

        # 1. Conversation summary
        summary = self.short_term_memory.get("conversation_summary", "")
        if summary:
            context_parts.append(f"Session Summary:\n{summary}")

        # 2. Recent conversation history (last 3 exchanges = 6 messages)
        recent = self.short_term_memory["conversation_history"][-6:]
        if recent:
            history_lines = "\n".join(
                f"  {msg['role']}: {msg['content'][:120]}"
                for msg in recent
            )
            context_parts.append(f"Recent Messages:\n{history_lines}")

        # 3. Analyst preferences
        prefs = self.long_term_memory.get("user_preferences", {})
        if prefs:
            prefs_text = ", ".join(f"{k}: {v}" for k, v in prefs.items())
            context_parts.append(f"Analyst Preferences:\n{prefs_text}")

        # 4. Learned facts — filter by query keywords
        all_facts = self.long_term_memory.get("learned_facts", [])
        if all_facts:
            relevant_facts = self._filter_relevant_facts(all_facts, query_lower)
            if relevant_facts:
                facts_lines = "\n".join(
                    f"  [{f.get('category', '?')}] {f['fact']}"
                    for f in relevant_facts
                )
                context_parts.append(f"Relevant Incident Facts:\n{facts_lines}")

        if not context_parts:
            return "No prior context."

        return "\n\n".join(context_parts)

    def _filter_relevant_facts(self, facts: List[Dict], query_lower: str) -> List[Dict]:
        """
        Return facts relevant to the query using keyword matching.
        Falls back to the 5 most recent facts if no match is found.
        """
        if not query_lower:
            return facts[-5:]

        # Extract meaningful keywords (length > 3, skip common words)
        stopwords = {"what", "when", "where", "this", "that", "with", "from",
                     "have", "does", "should", "about", "there", "their", "they"}
        keywords = [
            w for w in re.findall(r"\b\w+\b", query_lower)
            if len(w) > 3 and w not in stopwords
        ]

        # Also extract IP addresses from the query directly
        ips_in_query = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", query_lower)
        # And MITRE technique IDs
        techniques_in_query = re.findall(r"t\d{4}(?:\.\d{3})?", query_lower)

        all_search_terms = keywords + ips_in_query + techniques_in_query

        if not all_search_terms:
            return facts[-5:]

        relevant = [
            f for f in facts
            if any(term in f["fact"].lower() for term in all_search_terms)
               or any(term in f.get("category", "").lower() for term in all_search_terms)
        ]

        # If keyword match found, return up to 8 most recent relevant facts
        if relevant:
            return relevant[-8:]

        # Fallback: most recent 5 facts
        return facts[-5:]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return a summary of current memory state."""
        facts = self.long_term_memory["learned_facts"]
        categories: Dict[str, int] = {}
        for f in facts:
            cat = f.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "user_id": self.user_id,
            "short_term_messages": len(self.short_term_memory["conversation_history"]),
            "has_summary": bool(self.short_term_memory["conversation_summary"]),
            "preferences_count": len(self.long_term_memory["user_preferences"]),
            "learned_facts_count": len(facts),
            "facts_by_category": categories,
        }