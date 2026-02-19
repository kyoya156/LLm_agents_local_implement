import ollama, prompts, json, os
from datetime import datetime
import prompts


class MemoryManager:
    """Manages short-term and long-term memory for the agents."""
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory_dir = "./memory"
        os.makedirs(self.memory_dir, exist_ok=True)

        # short-term memory (in-memory, reset each session)
        # tried using dict with specific keys 
        self.short_term_memory = {
            "conversation_history": [],
            "conversation_summary": ""
        }

        # long-term memory
        # stored in a JSON file per user
        self.user_file = os.path.join(self.memory_dir, f"{user_id}_data.json")
        self.load_long_term_memory()

    def load_long_term_memory(self):
        """Load long-term memory from file"""
        if os.path.exists(self.user_file):
            with open(self.user_file, 'r') as f:
                data = json.load(f)
                self.user_preferences = data.get("user_preferences", {})
                self.learned_facts = data.get("learned_facts", [])
        else:
            self.user_preferences = {}
            self.learned_facts = []

    def save_long_term_memory(self):
        """Save long-term memory to file"""
        data = {
            "user_id": self.user_id,
            "preferences": self.user_preferences,
            "learned_facts": self.learned_facts,
            "last_updated": datetime.now().isoformat()# timestamp for tracking when memory was last updated
        }
        with open(self.user_file, 'w') as f:
            json.dump(data, f, indent=4)

    def add_to_short_term(self, role: str, content: str):
        """Add an entry to short-term memory"""
        self.short_term_memory["conversation_history"].append(
            {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
        )

        if len(self.short_term_memory["conversation_history"]) > 10: # keep only last 10 interactions
            self.short_term_memory["conversation_history"].pop(0) # remove oldest entry

    def summarize_conversation(self):
        """ summaize conversation for short-term memory """
        if len(self.short_term_memory["conversation_history"]) < 3: # if conversation is too short, skip summarization
            return "New conversation too soon to summarize."

        recent_conversation = self.short_term_memory["conversation_history"][-6:] # get last 6 interactions
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_conversation
        ])

        summary_prompt = prompts.summarize_prompt.format(history_text=history_text)

        from agents import  MultiAgentRAG
        response = ollama.chat(
            model= MultiAgentRAG.llm_model,
            messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                      {"role": "user", "content": summary_prompt}],
            stream=False
        )

        self.short_term_memory["conversation_summary"] = response["message"]["content"]
        return self.short_term_memory["conversation_summary"]
    
    def add_learned_fact(self, fact: str, category: str):
        """Add a new fact about user to long-term memory"""
        self.learned_facts.append({
            "fact": fact,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        self.save_long_term_memory()

    def update_user_preference(self, key: str, value: any):
        """Update a user preference in long-term memory"""
        self.user_preferences[key] = value
        self.save_long_term_memory()

    def get_context_for_query(self):
        """Get relevant context from both short-term and long-term memory for answering a query"""
        context_parts = []

        #short-term context (conversation summary)
        if self.short_term_memory["conversation_summary"]:
            context_parts.append(f"Recent conversation: {self.short_term_memory['conversation_summary']}")

        #long-term context (user preferences and learned facts)
        if self.user_preferences:
            prefs = ", ".join([f"{k}: {v}" for k, v in self.user_preferences.items()])
            context_parts.append(f"User preferences:\n{prefs}")

        if self.learned_facts:
            facts = [f['fact'] for f in self.learned_facts[-5:]] # get last 5 learned facts
            context_parts.append(f"Learned facts about user:\n{facts}")

        return "\n\n".join(context_parts)