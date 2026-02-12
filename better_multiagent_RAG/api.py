"""
FastAPI Backend Server
Provides API endpoints for the system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from vector_db import VectorDBManager
from agents import MultiAgentRAG


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent RAG API",
    description="A multi-agent RAG system with ChromaDB and LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and agents
db = VectorDBManager()
rag_system = MultiAgentRAG(db)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_docs: List[Dict]
    analysis: str
    agent_logs: List[str]


class HealthResponse(BaseModel):
    status: str
    database_count: int
    message: str


# API Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent RAG API",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "search": "/search (POST)",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    doc_count = db.get_collection_count()
    return {
        "status": "healthy",
        "database_count": doc_count,
        "message": f"System operational with {doc_count} documents in database"
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the multi-agent RAG pipeline"""
    try:
        # Check if database has documents
        if db.get_collection_count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents in database. Please initialize the database first."
            )
        
        # Process query through multi-agent system
        result = rag_system.process_query(request.query)
        
        return {
            "query": result["query"],
            "final_answer": result["final_answer"],
            "retrieved_docs": result["retrieved_docs"],
            "analysis": result["analysis"],
            "agent_logs": result["agent_logs"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: QueryRequest):
    """Direct search in vector database (without agents)"""
    try:
        results = db.search(request.query, top_k=request.top_k)
        
        formatted_results = []
        for distance, doc, metadata in results:
            formatted_results.append({
                "document": doc.strip(),
                "similarity": 1 - distance,
                "metadata": metadata
            })
        
        return {
            "query": request.query,
            "results": formatted_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    return {
        "total_documents": db.get_collection_count(),
        "collection_name": db.collection.name,
        "embedding_model": db.embedding_model
    }


if __name__ == "__main__":

    print("Starting FastAPI Server")
    print(f"Database documents: {db.get_collection_count()}")
    print("\nAPI will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)