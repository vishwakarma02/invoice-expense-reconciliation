
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion.agent import run_ingestion


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "ingestion_output.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    batch = run_ingestion(INPUT_DIR, OUTPUT_FILE)

    total = len(batch.documents)
    successful = sum(d.status == "success" for d in batch.documents)
    partial = sum(d.status == "partial" for d in batch.documents)
    failed = sum(d.status == "failed" for d in batch.documents)

    print("\n=== Ingestion Summary ===")
    print(f"Batch ID: {batch.batch_id}")
    print(f"Files discovered: {total}")
    print(f"Successful: {successful}")
    print(f"Partial: {partial}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_FILE}")

    if failed:
        print("\nFailed files:")
        for document in batch.documents:
            if document.status == "failed":
                print(f"- {document.source_file}")
                for error in document.errors:
                    print(f"  {error}")

    if partial:
        print("\nPartially processed files:")
        for document in batch.documents:
            if document.status == "partial":
                print(f"- {document.source_file}")
                for error in document.errors:
                    print(f"  {error}")


if __name__ == "__main__":
    main()
