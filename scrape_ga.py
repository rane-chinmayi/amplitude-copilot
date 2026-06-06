import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

urls = [
    "https://support.google.com/analytics/answer/9304153",
    "https://support.google.com/analytics/answer/9143382",
    "https://support.google.com/analytics/answer/9212670",
    "https://support.google.com/analytics/answer/11986666",
    "https://support.google.com/analytics/answer/9267568",
    "https://support.google.com/analytics/answer/9756891",
    "https://support.google.com/analytics/answer/9356048",
]

docs_folder = "docs_ga"
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

for url in urls:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        # Generate filename from URL
        path_parts = urlparse(url).path.strip("/").split("/")
        filename = "_".join(filter(None, path_parts)) + ".txt"
        filepath = os.path.join(docs_folder, filename)

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✓ Saved: {filepath}")

    except requests.RequestException as e:
        print(f"✗ Error fetching {url}: {e}")
    except Exception as e:
        print(f"✗ Error processing {url}: {e}")

print("\nScraping complete!")
