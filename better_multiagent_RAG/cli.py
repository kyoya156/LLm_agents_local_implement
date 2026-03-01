"""
Interactive CLI — Cybersecurity Multi-Agent RAG
"""
import sys
import os
from vector_db import VectorDBManager
from main import MultiAgentRAG


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_header():
    print("""
╔══════════════════════════════════════════════════════════╗
║        CYBERSECURITY RAG — Multi-Agent SOC Assistant     ║
╚══════════════════════════════════════════════════════════╝

Agents: Planner → Retriever → Analyzer (MITRE ATT&CK) → Report Writer → Memory

Commands:
  <question>      — Ask a cybersecurity question
  <paste logs>    — Paste raw log lines for instant analysis
  logfile <path>  — Analyze a log file on disk
  mitre <T-ID>    — Look up a MITRE technique (e.g. mitre T1110)
  mitre <keyword> — Search MITRE by keyword (e.g. mitre brute force)
  memory          — Show current incident memory
  reasoning       — Show ReAct reasoning from last query
  clear           — Clear short-term memory
  stats           — Show DB statistics
  help            — Show this help
  quit            — Exit
""")


def print_result(result: dict, show_details: bool = True):
    print("\n" + "═" * 60)
    print("  INCIDENT REPORT")
    print("═" * 60)
    print(result.get("final_answer", "(no answer generated)"))

    if not show_details:
        return

    print("\n" + "─" * 60)
    print("  ANALYSIS")
    print("─" * 60)
    print(f"  Intent:     {result.get('query_intent', 'N/A')}")
    print(f"  Severity:   {result.get('severity', 'N/A')}")
    print(f"  Confidence: {result.get('quality_score', 0):.2f}/1.00")
    print(f"  Facts:      {len(result.get('verified_facts', []))}")
    print(f"  Docs used:  {len(result.get('retrieved_docs', []))}")

    mitre = result.get("mitre_assessment", "")
    if mitre:
        print("\n" + "─" * 60)
        print("  MITRE ATT&CK ASSESSMENT")
        print("─" * 60)
        # Print first 600 chars of the MITRE assessment
        print(mitre[:600] + ("..." if len(mitre) > 600 else ""))

    print("\n" + "─" * 60)
    print("  AGENT WORKFLOW")
    print("─" * 60)
    for log in result.get("agent_logs", []):
        print(f"  • {log}")


def print_memory_state(rag: MultiAgentRAG):
    print("\n" + "═" * 60)
    print("  INCIDENT MEMORY")
    print("═" * 60)

    history = rag.memory.short_term_memory.get("conversation_history", [])
    print(f"\nConversation History: {len(history)} messages")
    for msg in history[-4:]:
        role = "Analyst" if msg["role"] == "user" else "System"
        content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
        print(f"  [{role}] {content}")

    summary = rag.memory.short_term_memory.get("conversation_summary", "")
    if summary:
        print(f"\nSession Summary:\n  {summary}")

    facts = rag.memory.long_term_memory.get("learned_facts", [])
    print(f"\nStored Incident Facts: {len(facts)}")
    for fact in facts[-8:]:
        print(f"  • [{fact.get('category', '?')}] {fact.get('fact', '')}")


def print_reasoning_trace(result: dict):
    print("\n" + "═" * 60)
    print("  REACT REASONING TRACE")
    print("═" * 60)
    for i, step in enumerate(result.get("reasoning_steps", []), 1):
        print(f"\n{i}. {step}")
    for action in result.get("actions_taken", []):
        print(f"\n→ {action['agent']}: {action['action']} (severity={action.get('severity','?')})")


def handle_mitre_command(rag: MultiAgentRAG, args: str):
    """Handle: mitre T1110 OR mitre brute force"""
    args = args.strip()
    if not args:
        print("Usage: mitre <T-ID> or mitre <keyword>")
        return

    kb = rag.mitre_kb
    # Direct ID lookup
    if args.upper().startswith("T") and len(args) <= 9:
        technique = kb.get_technique(args.upper())
        if technique:
            print(f"\n  {technique['id']} — {technique['name']}")
            print(f"  Tactics:      {', '.join(technique['tactics'])}")
            print(f"  Platforms:    {', '.join(technique['platforms'])}")
            print(f"  Description:  {technique['description'][:300]}")
            print(f"  Detection:    {technique['detection'][:300]}")
            print(f"  URL:          {technique['url']}")
        else:
            print(f"  Technique '{args}' not found in loaded KB.")
        return

    # Keyword search
    results = kb.search_by_keyword(args, limit=5)
    if results:
        print(f"\n  Search results for '{args}':")
        for t in results:
            print(f"  • {t['id']} — {t['name']} [{', '.join(t['tactics'])}]")
    else:
        print(f"  No techniques found matching '{args}'.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("\n  Initialising Cybersecurity RAG System...")

    db = VectorDBManager(collection_name="cybersec_knowledge")

    if db.get_collection_count() == 0:
        print("\n  ERROR: Database is empty!")
        print("  Run 'python setup.py' first to populate the knowledge base.\n")
        sys.exit(1)

    user_id = input("  Enter analyst name (or press Enter for 'analyst'): ").strip()
    if not user_id:
        user_id = "analyst"

    rag = MultiAgentRAG(db, user_id=user_id)
    print(f"\n  System ready — {db.get_collection_count()} documents | {len(rag.mitre_kb.techniques)} MITRE techniques")

    facts = rag.memory.long_term_memory.get("learned_facts", [])
    if facts:
        print(f"  Loaded session memory: {len(facts)} stored incident facts")

    print_header()

    last_result = None

    while True:
        try:
            raw = input("  SOC> ").strip()
            if not raw:
                continue

            # ── Commands ──────────────────────────────────────────────
            lower = raw.lower()

            if lower in ("quit", "exit", "q"):
                print("\n  Saving memory...")
                rag.memory.save_long_term_memory()
                print("  Goodbye.\n")
                break

            elif lower == "help":
                print_header()

            elif lower == "memory":
                print_memory_state(rag)

            elif lower == "reasoning":
                if last_result:
                    print_reasoning_trace(last_result)
                else:
                    print("  No previous query yet.")

            elif lower == "clear":
                rag.memory.short_term_memory["conversation_history"] = []
                rag.memory.short_term_memory["conversation_summary"] = ""
                print("  Short-term memory cleared.")

            elif lower == "stats":
                print(f"\n  DB documents:      {db.get_collection_count()}")
                print(f"  MITRE techniques:  {len(rag.mitre_kb.techniques)}")
                print(f"  MITRE file:        {rag.mitre_kb.stix_file} ({'loaded' if rag.mitre_kb.available else 'fallback'})")
                print(f"  Embedding model:   {db.embedding_model}")
                print(f"  Collection:        {db.collection.name}")

            elif lower.startswith("mitre "):
                handle_mitre_command(rag, raw[6:])

            elif lower.startswith("logfile "):
                filepath = raw[8:].strip()
                if not os.path.exists(filepath):
                    print(f"  File not found: {filepath}")
                else:
                    print(f"\n  Analysing log file: {filepath}")
                    result = rag.analyze_log_file(filepath)
                    last_result = result
                    print_result(result)

            # ── Security query ────────────────────────────────────────
            else:
                print("\n  Processing query...")
                result = rag.process_query(raw)
                last_result = result
                print_result(result)

                # First-use tips
                if len(rag.memory.short_term_memory.get("conversation_history", [])) == 2:
                    print("\n  Tip: Type 'mitre T1110' to look up ATT&CK techniques directly.")
                    print("  Tip: Type 'reasoning' to see the full ReAct thought process.")
                    print("  Tip: Paste raw log lines directly into the prompt for instant analysis.")

        except KeyboardInterrupt:
            print("\n\n  Saving memory...")
            rag.memory.save_long_term_memory()
            print("  Goodbye.\n")
            break

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()