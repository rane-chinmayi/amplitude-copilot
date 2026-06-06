import os
import json
from pathlib import Path

docs_folder = "docs_ga"
chunks_list = []

chunk_size = 500
overlap = 50
step_size = chunk_size - overlap

if not os.path.exists(docs_folder):
    print(f"Error: {docs_folder} folder not found.")
    exit(1)

txt_files = list(Path(docs_folder).glob("*.txt"))

if not txt_files:
    print(f"No .txt files found in {docs_folder} folder.")
    exit(1)

for txt_file in txt_files:
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        words = content.split()

        if len(words) == 0:
            print(f"⊘ {txt_file.name}: Empty file")
            continue

        for i in range(0, len(words), step_size):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk = {
                "text": chunk_text,
                "source": txt_file.name,
                "tool": "google_analytics"
            }
            chunks_list.append(chunk)

        num_chunks = (len(words) - chunk_size) // step_size + 1 if len(words) >= chunk_size else 1
        print(f"✓ {txt_file.name}: {num_chunks} chunk(s)")

    except Exception as e:
        print(f"✗ Error processing {txt_file.name}: {e}")

output_file = "chunks_ga.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks_list, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(chunks_list)} total chunks to {output_file}")
