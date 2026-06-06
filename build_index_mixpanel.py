import json
import os
import time
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

chunks_file = "chunks_mixpanel.json"
index_output = "amplitude_index_mixpanel.faiss"
embeddings_output = "embeddings_mixpanel.npy"
model_name = "gemini-embedding-001"

print("Loading chunks...")
try:
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
except FileNotFoundError:
    print(f"Error: {chunks_file} not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: {chunks_file} is not valid JSON.")
    exit(1)

if not chunks:
    print("Error: No chunks found in the JSON file.")
    exit(1)

print(f"Loaded {len(chunks)} chunks.")
print(f"Creating embeddings using {model_name}...\n")

embeddings = []

for i, chunk in enumerate(chunks):
    try:
        text = chunk["text"]

        # Try to embed with retry on rate limit (429 error)
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=text
                )
                embeddings.append(response.embeddings[0].values)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    print(f"  Rate limited on chunk {i + 1}. Waiting 60 seconds before retry...")
                    time.sleep(60)
                else:
                    raise

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{len(chunks)} chunks embedded")

        # Pause between calls to avoid rate limiting
        time.sleep(2)

    except Exception as e:
        print(f"Error embedding chunk {i}: {e}")
        exit(1)

print(f"  Progress: {len(chunks)}/{len(chunks)} chunks embedded\n")

embeddings = np.array(embeddings).astype("float32")

print(f"Creating FAISS index with {len(embeddings)} embeddings...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Saving FAISS index to {index_output}...")
faiss.write_index(index, index_output)

print(f"Saving embeddings to {embeddings_output}...")
np.save(embeddings_output, embeddings)

print(f"\n✓ Successfully created embeddings and FAISS index")
print(f"  - Model: {model_name}")
print(f"  - Chunks embedded: {len(chunks)}")
print(f"  - Embedding dimension: {dimension}")
print(f"  - Index file: {index_output}")
print(f"  - Embeddings file: {embeddings_output}")
