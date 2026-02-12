import chromadb
from chromadb.config import Settings
import ollama
from typing import List, Dict, Tuple

class VectorDBManager:
    """Manages a ChromaDB vector database with Ollama embeddings."""
    def __init__(self, collection_name: str = "cat_facts", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Try to get the collection, if it doesn't exist, create it
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Collection {collection_name} found ({self.collection.count()} docs).")
        except:
            # Collection does not exist, create it (just a demo)
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Collection {collection_name} created.")
        
        self.embedding_model = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'# i use the alr downloaded embedding model used in the simple rag example

    def embed_texts(self, text: str) -> List[float]:
        # Generate embeddings using Ollama for a single text input
        respond = ollama.embed(model=self.embedding_model, input=text)
        return respond['embeddings'][0]

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        # Embed and add documents to the collection
        # if there are no docs, return
        if not documents:
            return
        
        print(f"Embedding and adding {len(documents)} documents to the DB...")

        embeddings = []
        # Embed each document and track progress
        for i, doc in enumerate(documents):
            emb = self.embed_texts(doc.strip())
            embeddings.append(emb)
            # Print progress every 25 documents
            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents.")

        # prepare metadata
        if metadata is None:
            # create simple metadata with document index
            metadata = [{"source": f"doc_{i}", "text_preview": doc[:50]} for i, doc in enumerate(documents)]

        # add to collection
        # Generate unique IDs for each document
        ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata
        )
        print(f"Added {len(documents)} documents to the DB.")

    def search(self, 
                query: str, 
                top_k: int = 3 # number of top results to return
                ) -> List[Tuple[str, float]]:
        # Embed the query text
        query_embedding = self.embed_texts(query)

        # Search the in the DB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # format the results as (distance, document, metadata)
        formatted_results = []
        # Check if there are any documents returned
        if results['documents'] and len(results['documents']) > 0:
            # Iterate through the results and format them
            for i in range(len(results['documents'][0])):
                # Get the document, metadata, and distance
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                formatted_results.append((distance, doc, metadata))

        return formatted_results

    def get_collection_count(self) -> int:
        '''Get the number of documents in the collection'''
        return self.collection.count()

# quick test
if __name__ == "__main__":
    db = VectorDBManager()
    print(f"Collection has {db.get_collection_count()} documents.")