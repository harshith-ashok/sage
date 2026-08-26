"""One-time setup download for fastText's lid.176 language-identification
model (~126MB, publicly downloadable, no auth) — the same "download once,
then fully air-gapped" pattern as `ollama pull` or `scripts/ingest_knowledge.py`
warming the reranker. Run once from `backend/`:

    uv run python -m scripts.download_lid_model
"""

import os
import urllib.request

URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
DEST = os.path.join(os.path.dirname(__file__), "..", "data", "models", "lid.176.bin")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.isfile(DEST):
        print(f"Already downloaded: {DEST}")
    else:
        print(f"Downloading {URL} -> {DEST} ...")
        urllib.request.urlretrieve(URL, DEST)
        print("Done.")
