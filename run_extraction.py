import logging
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.agent import run_extraction


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

INPUT_FILE = PROJECT_ROOT / "data" / "output" / "ingestion_output.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "extraction_output.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    batch = run_extraction(INPUT_FILE, OUTPUT_FILE)

    documents = len(batch.documents)
    records = sum(len(document.records) for document in batch.documents)
    failed = sum(document.status == "failed" for document in batch.documents)
    partial = sum(document.status == "partial" for document in batch.documents)
    unsupported = sum(document.status == "unsupported" for document in batch.documents)

    print("\n=== Extraction Summary ===")
    print(f"Ingestion batch: {batch.ingestion_batch_id}")
    print(f"Documents processed: {documents}")
    print(f"Financial records extracted: {records}")
    print(f"Partial: {partial}")
    print(f"Unsupported: {unsupported}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_FILE}")

    for document in batch.documents:
        if document.status == "failed":
            print(f"- Failed: {document.source_file}")
            for error in document.errors:
                print(f"  {error}")


if __name__ == "__main__":
    main()
