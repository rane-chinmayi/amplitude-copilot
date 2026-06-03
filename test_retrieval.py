import json
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

client = genai.Client(api_key=api_key)

# Load chunks, index, and embeddings
print("Loading chunks, index, and embeddings...\n")

try:
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index("amplitude_index.faiss")
    embeddings = np.load("embeddings.npy")

except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    exit(1)

# Test query
query = "How do I build a funnel in Amplitude?"
print(f"Query: {query}\n")

# Embed the query
print("Embedding query...")
try:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_embedding = np.array(response.embeddings[0].values).astype("float32").reshape(1, -1)
except Exception as e:
    print(f"Error embedding query: {e}")
    exit(1)

# Search FAISS index
print("Searching for top 3 most relevant chunks...\n")
distances, indices = index.search(query_embedding, 3)

# Print results
print("=" * 80)
print("RETRIEVAL RESULTS")
print("=" * 80)

for i, idx in enumerate(indices[0]):
    chunk = chunks[idx]
    text = chunk["text"]
    source = chunk["source"]

    # Get first 200 characters
    preview = text[:200] + "..." if len(text) > 200 else text

    print(f"\n[Result {i + 1}]")
    print(f"  Source: {source}")
    print(f"  Similarity Distance: {distances[0][i]:.4f}")
    print(f"  Preview: {preview}")

print("\n" + "=" * 80)
