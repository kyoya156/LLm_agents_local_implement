"""
Interactive CLI for Multi-Agent RAG System
"""
from vector_db import VectorDBManager
from main import MultiAgentRAG
import sys


def print_header():
    """Print CLI header"""

    print(" RAG SYSTEM - Interactive CLI")
    print("\nThis system uses 3 agents:")

    print("\nCommands:")
    print("  - Type your question")
    print("  - 'memory' - Show current memory state")
    print("  - 'reasoning' - Show ReAct reasoning from last query")
    print("  - 'clear' - Clear short-term memory")
    print("  - 'stats' - Database statistics")
    print("  - 'help' - Show this help")
    print("  - 'quit' - Exit")

def print_memory_state(rag: MultiAgentRAG):
    """Display current memory state"""
    print("\n")
    print(" MEMORY STATE")
    print("\n")

    
    # Short-term memory
    print("\n SHORT-TERM MEMORY (Conversation History):")
    if rag.memory.short_term_memory["conversation_history"]:
        print(f"   Messages in history: {len(rag.memory.short_term_memory['conversation_history'])}")
        print(f"   Last {min(3, len(rag.memory.short_term_memory['conversation_history']))} messages:")
        for msg in rag.memory.short_term_memory['conversation_history'][-3:]:
            role = "You" if msg['role'] == 'user' else "Bot"
            content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
            print(f"      {role}: {content}")
    else:
        print("   (empty)")
    
    print(f"\n Conversation Summary:")
    if rag.memory.short_term_memory['conversation_summary']:
        print(f"   {rag.memory.short_term_memory['conversation_summary']}")
    else:
        print("   (no summary yet)")
    
    # Long-term memory
    print("\n LONG-TERM MEMORY (Persistent):")
    
    print(f"\n   User Preferences:")
    if rag.memory.long_term_memory['user_preferences']:
        for key, value in rag.memory.long_term_memory['user_preferences'].items():
            print(f"      • {key}: {value}")
    else:
        print("      (none learned yet)")
    
    print(f"\n   Learned Facts about You:")
    if rag.memory.long_term_memory['learned_facts']:
        for fact in rag.memory.long_term_memory['learned_facts'][-5:]:  # Show last 5
            print(f"      • [{fact['category']}] {fact['fact']}")
    else:
        print("      (none learned yet)")

def print_reasoning_trace(result: dict):
    """Display ReAct reasoning process"""
    print("\n")
    print(" REACT REASONING TRACE")
    print("\n")
    
    if 'reasoning_steps' in result and result['reasoning_steps']:
        for i, step in enumerate(result['reasoning_steps'], 1):
            print(f"\n{i}. {step}")
    else:
        print("   No reasoning trace available")
    
    if 'actions_taken' in result and result['actions_taken']:
        print("\n ACTIONS TAKEN:")
        for action in result['actions_taken']:
            print(f"   • {action['agent']}: {action['action']}")

def print_result(result: dict, show_details: bool = True):
    """Pretty print the result"""
    print("\n")
    print(" FINAL ANSWER:")
    print("\n")
    print(result['final_answer'])
    
    if show_details:
        print("\n")
        print(" ANALYSIS:")
        print("\n")
        print(f"Query Intent: {result.get('query_intent', 'N/A')}")
        print(f"Quality Score: {result.get('quality_score', 0):.2f}/1.00")
        print(f"Facts Verified: {len(result.get('verified_facts', []))}")
        print(f"Memory Used: {len(result.get('conversation_history', []))} messages in context")
        
        print("\n")
        print(" TOP RETRIEVED DOCUMENTS:")
        print("\n")
        for doc in result['retrieved_docs'][:3]:
            print(f"\n{doc['rank']}. Similarity: {doc['similarity']:.4f} | Relevance: {doc.get('relevance', 'N/A')}")
            print(f"   {doc['content'][:120]}...")

        print("\n")
        print(" AGENT WORKFLOW:")
        print("\n")
        for log in result['agent_logs']:
            print(f"   {log}")
    


def main():
    """Main CLI loop"""
    # Initialize system
    print(" Initializing enhanced system...")
    db = VectorDBManager()
    
    if db.get_collection_count() == 0:
        print("\n Error: No documents in database!")
        print("   Please run 'python setup.py' first to initialize the database.\n")
        sys.exit(1)
    
    # Get user ID (for memory persistence)
    user_id = input("Enter your username (or press Enter for 'default'): ").strip()
    if not user_id:
        user_id = "default_user"
    
    rag = MultiAgentRAG(db, user_id=user_id)
    print(f" System ready! (User: {user_id})")
    print(f" Database: {db.get_collection_count()} documents")
    
    # Check if user has existing memory
    if rag.memory.long_term_memory['learned_facts'] or rag.memory.long_term_memory['user_preferences']:
        print(f" Loaded your memory: {len(rag.memory.long_term_memory['learned_facts'])} facts, {len(rag.memory.long_term_memory['user_preferences'])} preferences")

    print_header()
    
    last_result = None
    
    # Main loop
    while True:
        try:
            # Get user input
            query = input(" Your question: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n Saving memory...")
                rag.memory.save_long_term_memory()
                print(" Bye Dawg!\n")
                break
            
            elif query.lower() == 'help':
                print_header()
                continue
            
            elif query.lower() == 'memory':
                print_memory_state(rag)
                continue
            
            elif query.lower() == 'reasoning':
                if last_result:
                    print_reasoning_trace(last_result)
                else:
                    print("   No previous query to show reasoning for.\n")
                continue
            
            elif query.lower() == 'clear':
                rag.memory.short_term_memory["conversation_history"] = []
                rag.memory.short_term_memory["conversation_summary"] = ""
                print("   ✓ Short-term memory cleared\n")
                continue
            
            elif query.lower() == 'stats':
                print(f"\n Database Statistics:")
                print(f"   Total documents: {db.get_collection_count()}")
                print(f"   Collection: {db.collection.name}")
                print(f"   Embedding model: {db.embedding_model}")
                print()
                continue
            
            # Process query through enhanced system
            result = rag.process_query(query)
            last_result = result
            
            # Display result
            print_result(result, show_details=True)
            
            # Show hint about commands
            if len(rag.memory.short_term_memory["conversation_history"]) == 2:  # First Q&A
                print(" Tip: Type 'memory' to see what I'm learning about you!")
                print(" Tip: Type 'reasoning' to see my thought process!\n")
        
        except KeyboardInterrupt:
            print("\n\n Saving memory...")
            rag.memory.save_long_term_memory()
            print(" Bye Dawg!\n")
            break
        
        except Exception as e:
            print(f"\n Error: {str(e)}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
