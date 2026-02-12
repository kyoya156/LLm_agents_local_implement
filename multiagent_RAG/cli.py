"""
Interactive CLI for Multi-Agent RAG System
"""
from vector_db import VectorDBManager
from agents import MultiAgentRAG
import sys


def print_header():
    """Print CLI header"""
    print("\n" + "="*70)
    print("🤖 MULTI-AGENT RAG SYSTEM - Interactive CLI")
    print("="*70)
    print("\nThis system uses 3 agents working together:")
    print("  1️⃣  Retriever Agent - Searches vector database")
    print("  2️⃣  Analyzer Agent - Analyzes document relevance")
    print("  3️⃣  Answer Agent - Generates final answer")
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'stats' - Show database statistics")
    print("  - 'help' - Show this help message")
    print("  - 'quit' or 'exit' - Exit the program")
    print("="*70 + "\n")


def print_result(result: dict, show_details: bool = True):
    """Pretty print the result"""
    print("\n" + "─"*70)
    print("📝 FINAL ANSWER:")
    print("─"*70)
    print(result['final_answer'])
    
    if show_details:
        print("\n" + "─"*70)
        print("📚 RETRIEVED DOCUMENTS:")
        print("─"*70)
        for doc in result['retrieved_docs']:
            print(f"\n{doc['rank']}. Similarity: {doc['similarity']:.4f}")
            print(f"   {doc['content'][:150]}...")
        
        print("\n" + "─"*70)
        print("🔄 AGENT WORKFLOW:")
        print("─"*70)
        for log in result['agent_logs']:
            print(f"  ✓ {log}")
    
    print("="*70 + "\n")


def main():
    """Main CLI loop"""
    # Initialize system
    print("🔄 Initializing system...")
    db = VectorDBManager()
    
    if db.get_collection_count() == 0:
        print("\n❌ Error: No documents in database!")
        print("   Please run 'python setup.py' first to initialize the database.\n")
        sys.exit(1)
    
    rag = MultiAgentRAG(db)
    print(f"✓ System ready! Database has {db.get_collection_count()} documents\n")
    
    print_header()
    
    # Main loop
    while True:
        try:
            # Get user input
            query = input("💬 Your question: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            elif query.lower() == 'help':
                print_header()
                continue
            
            elif query.lower() == 'stats':
                print(f"\n📊 Database Statistics:")
                print(f"   Total documents: {db.get_collection_count()}")
                print(f"   Collection: {db.collection.name}")
                print(f"   Embedding model: {db.embedding_model}")
                print()
                continue
            
            # Process query through multi-agent system
            result = rag.process_query(query)
            
            # Display result
            print_result(result, show_details=True)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()