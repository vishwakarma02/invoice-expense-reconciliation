import logging
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.agent import run_extraction
from src.ingestion.agent import run_ingestion
from src.matching.agent import run_matching


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
INGESTION_OUTPUT = OUTPUT_DIR / "ingestion_output.json"
EXTRACTION_OUTPUT = OUTPUT_DIR / "extraction_output.json"
MATCHING_OUTPUT = OUTPUT_DIR / "matching_output.json"


def run_pipeline() -> None:
    ingestion_batch = run_ingestion(INPUT_DIR, INGESTION_OUTPUT)
    if not INGESTION_OUTPUT.is_file():
        raise FileNotFoundError(
            f"Ingestion completed without creating its output: {INGESTION_OUTPUT}"
        )

    extraction_batch = run_extraction(INGESTION_OUTPUT, EXTRACTION_OUTPUT)
    matching_batch = run_matching(EXTRACTION_OUTPUT, MATCHING_OUTPUT)

    print("\n=== Three-Agent Pipeline Summary ===")
    print(f"Ingestion batch: {ingestion_batch.batch_id}")
    print(f"Documents ingested: {len(ingestion_batch.documents)}")
    print(f"Extraction batch: {extraction_batch.extraction_id}")
    print(
        "Financial records extracted: "
        f"{sum(len(document.records) for document in extraction_batch.documents)}"
    )
    print(
        "Extraction failures: "
        f"{sum(document.status == 'failed' for document in extraction_batch.documents)}"
    )
    print(
        "Invoices matched: "
        f"{sum(item.status == 'matched' for item in matching_batch.invoices)}"
    )
    print(
        "Invoice exceptions: "
        f"{sum(item.status == 'exception' for item in matching_batch.invoices)}"
    )
    print(
        "Invoice reviews: "
        f"{sum(item.status == 'review' for item in matching_batch.invoices)}"
    )
    print(f"Bank payment results: {len(matching_batch.bank_payments)}")
    print(f"Ingestion output: {INGESTION_OUTPUT}")
    print(f"Extraction output: {EXTRACTION_OUTPUT}")
    print(f"Matching output: {MATCHING_OUTPUT}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv(PROJECT_ROOT / ".env")
    run_pipeline()


if __name__ == "__main__":
    main()
