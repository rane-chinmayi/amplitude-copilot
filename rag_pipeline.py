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
print("Loading resources...")
try:
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    index = faiss.read_index("amplitude_index.faiss")
    embeddings = np.load("embeddings.npy")
except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    exit(1)

print("Resources loaded successfully.\n")


def get_answer(query):
    """
    Retrieves relevant documentation chunks and generates an answer using Gemini.

    Args:
        query: User's question

    Returns:
        Dictionary with 'answer' and 'source' keys
    """

    # Embed the query
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_embedding = np.array(response.embeddings[0].values).astype("float32").reshape(1, -1)

    # Search FAISS index for top 3 chunks
    distances, indices = index.search(query_embedding, 3)

    # Build context from retrieved chunks
    context = "Amplitude Documentation:\n\n"
    top_source = None

    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
        if i == 0:
            top_source = chunk['source']

    # Build the prompt for Gemini
    system_prompt = """You are an expert Amplitude analytics assistant. Answer questions based ONLY on the provided Amplitude documentation.

Instructions:
- Answer only using information from the provided documentation
- Keep answers concise and practical
- If the answer isn't in the provided documentation, respond with: "I couldn't find this in the Amplitude documentation. Try searching docs.amplitude.com directly."
- Provide actionable and clear answers"""

    user_prompt = f"""{system_prompt}

{context}

User Question: {query}

Please answer based on the documentation above."""

    # Generate answer with Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt
    )

    answer = response.text

    return {
        "answer": answer,
        "source": top_source
    }


# Test the pipeline
if __name__ == "__main__":
    test_queries = [
        "What is a cohort in Amplitude?",
        "How do I track retention?",
        "What is the weather in London?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}\n")
        print("Generating answer...\n")

        result = get_answer(query)

        print("ANSWER:")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)
        print(f"Top Source Document: {result['source']}\n")
