import ollama

dataset = []
with open("cat_facts.txt", "r", encoding="utf-8") as file:
    dataset = file.readlines()
    print(f"Loaded {len(dataset)} cat facts entries.")

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

# Function to embed and store chunks in the vector database
def embed_chunk(chunk):
    response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
    embedding = response.embeddings[0]
    VECTOR_DB.append({'text': chunk, 'embedding': embedding})

# ASSUMPTION: Each line in the dataset is a separate fact (A CHUNK)
for i, chunk in enumerate(dataset):
    embed_chunk(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

# Function to compute cosine similarity
def cosine_similarity(a, b):
    dot_product = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(y**2 for y in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

# Function to retrieve relevant chunks based on a query
def retrieve(query, top_n=3):
    response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    query_embedding = response.embeddings[0]
    # temporary list to store (similarity, chunk) pairs
    similarities = []
    for entry in VECTOR_DB:
        sim = cosine_similarity(query_embedding, entry['embedding'])
        similarities.append((sim, entry['text']))
    # sort by similarity in descending order
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_n]


print("Ask questions about cats (type 'quit' to exit)")


while True:
    input_query = input("\nAsk a question about cats: ").strip()
    
    # Check for exit commands first
    if input_query.lower() in ['exit', 'quit']:
        print("Exiting the program.")
        break

    # Check for empty input
    if not input_query:
        print("Please enter a valid question.")
        continue

    # Retrieve relevant information
    retrieved_information = retrieve(input_query)

    # Display retrieved information
    print("\nRetrieved Information:")
    for sim, chunk in retrieved_information:
        print(f"Similarity: {sim:.4f}, Chunk: {chunk.strip()}")

    # Create instruction prompt
    instruction_prompt = f"""
You are a helpful assistant. Answer the user's question based on the following cat facts.
Do not make up any information. If the answer is not in the facts, respond with "I don't know."

Cat Facts:
{chr(10).join([f' - {chunk.strip()}' for sim, chunk in retrieved_information])}

User Question: {input_query}
"""

    # Get response from language model
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query}
        ],
        stream=True,
    )

    # Print the response
    print("\nResponse:")
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")