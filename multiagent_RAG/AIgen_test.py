"""
Test Script - Verify the multi-agent RAG system works correctly
"""
import sys
from vector_db import VectorDBManager
from agents import MultiAgentRAG


def test_system():
    """Run a series of tests to verify the system"""
    print("\n" + "="*70)
    print("🧪 RUNNING SYSTEM TESTS")
    print("="*70)
    
    # Test 1: Database
    print("\n1️⃣ Testing ChromaDB...")
    try:
        db = VectorDBManager()
        doc_count = db.get_collection_count()
        print(f"   ✓ ChromaDB initialized")
        print(f"   ✓ Found {doc_count} documents in database")
        
        if doc_count == 0:
            print("\n   ⚠️  Warning: No documents in database!")
            print("   Run 'python setup.py' first to load data")
            return False
    except Exception as e:
        print(f"   ❌ ChromaDB test failed: {e}")
        return False
    
    # Test 2: Vector Search
    print("\n2️⃣ Testing vector search...")
    try:
        test_query = "cats and humans"
        results = db.search(test_query, top_k=2)
        print(f"   ✓ Search completed")
        print(f"   ✓ Retrieved {len(results)} documents")
        
        if results:
            dist, doc, meta = results[0]
            print(f"   ✓ Top result: {doc[:60]}...")
    except Exception as e:
        print(f"   ❌ Vector search test failed: {e}")
        return False
    
    # Test 3: Multi-Agent System
    print("\n3️⃣ Testing multi-agent system...")
    try:
        rag = MultiAgentRAG(db)
        print(f"   ✓ Multi-agent system initialized")
        print(f"   ✓ LangGraph workflow compiled")
    except Exception as e:
        print(f"   ❌ Multi-agent test failed: {e}")
        return False
    
    # Test 4: End-to-End Query
    print("\n4️⃣ Testing end-to-end query processing...")
    try:
        test_query = "What are some interesting facts about cats?"
        print(f"   Query: {test_query}")
        print(f"   Running through 3-agent pipeline...")
        
        result = rag.process_query(test_query)
        
        # Check all expected fields are present
        assert "query" in result, "Missing 'query' in result"
        assert "retrieved_docs" in result, "Missing 'retrieved_docs' in result"
        assert "analysis" in result, "Missing 'analysis' in result"
        assert "final_answer" in result, "Missing 'final_answer' in result"
        assert "agent_logs" in result, "Missing 'agent_logs' in result"
        
        print(f"\n   ✓ Query processed successfully")
        print(f"   ✓ Retrieved {len(result['retrieved_docs'])} documents")
        print(f"   ✓ Generated analysis: {len(result['analysis'])} chars")
        print(f"   ✓ Generated answer: {len(result['final_answer'])} chars")
        print(f"   ✓ Agent logs: {len(result['agent_logs'])} entries")
        
        print(f"\n   📝 Answer preview:")
        print(f"   {result['final_answer'][:150]}...")
        
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All tests passed
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe system is working correctly. You can now:")
    print("  1. Use the CLI: python cli.py")
    print("  2. Start the API: python api.py")
    print("  3. Run agents directly: python agents.py")
    print()
    
    return True


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)