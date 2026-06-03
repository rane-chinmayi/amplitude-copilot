import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

client = genai.Client(api_key=api_key)

print("Fetching available models...\n")

try:
    models = client.models.list()

    print("Available Models:")
    print("-" * 50)

    for model in models:
        print(f"  {model.name}")

    print("-" * 50)
    print(f"\nTotal models available: {len(list(client.models.list()))}")

except Exception as e:
    print(f"Error fetching models: {e}")
    exit(1)
