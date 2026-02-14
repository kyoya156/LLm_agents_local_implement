"""
Interactive CLI for Multi-Agent RAG System
"""
from vector_db import VectorDBManager
from agents import MultiAgentRAG
import sys


def print_header():
    """Print CLI header"""

    print(" RAG SYSTEM - Interactive CLI")

    print("\nThis system uses 3 agents:")
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'stats' - Show database statistics")
    print("  - 'help' - Show this help message")
    print("  - 'quit' or 'exit' - Exit the program")



def print_result(result: dict, show_details: bool = True):
    """Pretty print the result"""
    print("FINAL ANSWER:")
    print(result['final_answer'])

    if show_details:
        print("AGENT INSIGHTS:")
        print(f"Query Intent: {result.get('query_intent', 'N/A')}")
        print(f"Quality Score: {result.get('quality_score', 0):.2f}/1.00")
        print(f"Facts Verified: {len(result.get('verified_facts', []))}")

        print("RETRIEVED DOCUMENTS:")

        for doc in result['retrieved_docs'][:3]:
            print(f"\n{doc['rank']}. Similarity: {doc['similarity']:.4f} | Relevance: {doc.get('relevance', 'N/A')}")
            print(f"   {doc['content'][:150]}...")

        print("AGENT WORKFLOW:")

        for log in result['agent_logs']:
            print(f"{log}")
    


def main():
    """Main CLI loop"""
    # Initialize system
    print("Initializing system...")
    db = VectorDBManager()
    
    if db.get_collection_count() == 0:
        print("\nError: No documents in database!")
        print("   Please run 'python setup.py' first to initialize the database.\n")
        sys.exit(1)
    
    rag = MultiAgentRAG(db)
    print(f"System ready! Database has {db.get_collection_count()} documents\n")
    
    print_header()
    
    # Main loop
    while True:
        try:
            # Get user input
            query = input("Your question: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nBye Dawg!\n")
                break
            
            elif query.lower() == 'help':
                print_header()
                continue
            
            elif query.lower() == 'stats':
                print(f"\nDatabase Statistics:")
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
            print("\nBye Dawg!\n")
            break
        
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()