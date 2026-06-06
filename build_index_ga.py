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

chunks_file = "chunks_ga.json"
index_output = "amplitude_index_ga.faiss"
embeddings_output = "embeddings_ga.npy"
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

        # Try to embed with up to 3 attempts on rate limit (429 error)
        retry_waits = [90, 120]  # wait 90s on 1st failure, 120s on 2nd failure
        success = False
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=text
                )
                embeddings.append(response.embeddings[0].values)
                success = True
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = retry_waits[attempt]
                    print(f"  Rate limited on chunk {i + 1} (attempt {attempt + 1}/3). Waiting {wait} seconds before retry...")
                    time.sleep(wait)
                else:
                    raise

        if not success:
            print(f"  Skipping chunk {i + 1} after 3 failed attempts.")

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i + 1}/{len(chunks)} chunks embedded")

        # Pause between calls to avoid rate limiting
        time.sleep(4)

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
