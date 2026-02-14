from vector_db import VectorDBManager

def setup_database(data_file: str = "cat_facts.txt") -> bool:
    """Setup the vector database manager"""
    # Initialize the vector database
    db = VectorDBManager()
    # Load information into the database
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            information = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        return False
    db.add_documents(information)
    final_count = db.get_collection_count()
    print(f"Database setup complete. Total documents in DB: {final_count}")
    return True

# Quick test
if __name__ == "__main__":
    success = setup_database()
    if success:
        print("Database initialized successfully.")
    else:
        print("Database initialization failed.")