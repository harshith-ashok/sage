"""CLI wrapper, ported from insurance_claim_agent/ingest.py.

Usage (from backend/): uv run python scripts/ingest_knowledge.py
"""

import argparse
import logging

from app.knowledge.config import DOCS_DIR
from app.knowledge.pipeline import ingest

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Ingest SOP PDFs into Qdrant.")
    parser.add_argument(
        "--docs-dir",
        default=DOCS_DIR,
        help=f"Directory containing PDF files to ingest (default: {DOCS_DIR})",
    )
    args = parser.parse_args()
    ingest(args.docs_dir)
