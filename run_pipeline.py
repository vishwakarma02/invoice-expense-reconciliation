import logging
from pathlib import Path

from dotenv import load_dotenv

from src.anomaly.agent import run_anomaly
from src.extraction.agent import run_extraction
from src.explanation.agent import run_explanation
from src.hitl.agent import DEFAULT_DATABASE, HITLRouter
from src.ingestion.agent import run_ingestion
from src.matching.agent import run_matching


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
INGESTION_OUTPUT = OUTPUT_DIR / "ingestion_output.json"
EXTRACTION_OUTPUT = OUTPUT_DIR / "extraction_output.json"
MATCHING_OUTPUT = OUTPUT_DIR / "matching_output.json"
ANOMALY_OUTPUT = OUTPUT_DIR / "anomaly_output.json"
EXPLANATION_OUTPUT = OUTPUT_DIR / "explanation_output.json"
HITL_DATABASE = DEFAULT_DATABASE


def run_pipeline(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    hitl_database: Path | None = None,
) -> None:
    input_dir = input_dir or INPUT_DIR
    if output_dir is None:
        ingestion_output = INGESTION_OUTPUT
        extraction_output = EXTRACTION_OUTPUT
        matching_output = MATCHING_OUTPUT
        anomaly_output = ANOMALY_OUTPUT
        explanation_output = EXPLANATION_OUTPUT
    else:
        ingestion_output = output_dir / "ingestion_output.json"
        extraction_output = output_dir / "extraction_output.json"
        matching_output = output_dir / "matching_output.json"
        anomaly_output = output_dir / "anomaly_output.json"
        explanation_output = output_dir / "explanation_output.json"
    hitl_database = hitl_database or HITL_DATABASE

    ingestion_batch = run_ingestion(input_dir, ingestion_output)
    if not ingestion_output.is_file():
        raise FileNotFoundError(
            f"Ingestion completed without creating its output: {ingestion_output}"
        )

    extraction_batch = run_extraction(ingestion_output, extraction_output)
    matching_batch = run_matching(extraction_output, matching_output)
    anomaly_report = run_anomaly(extraction_output, anomaly_output)
    explanation_report = run_explanation(
        matching_output, explanation_output, anomaly_report
    )
    routed_cases = HITLRouter(hitl_database).route(
        matching_batch,
        explanation_report,
        anomaly_report,
        extraction_batch,
    )

    print("\n=== Reconciliation Pipeline Summary ===")
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
    print(f"Anomaly findings: {len(anomaly_report.findings)}")
    print(f"Exception explanations: {explanation_report.explanation_count}")
    print(
        "Human review queue: "
        f"{sum(case.status == 'pending_review' for case in routed_cases)}"
    )
    print(f"Ingestion output: {ingestion_output}")
    print(f"Extraction output: {extraction_output}")
    print(f"Matching output: {matching_output}")
    print(f"Anomaly output: {anomaly_output}")
    print(f"Explanation output: {explanation_output}")
    print(f"Review database: {hitl_database}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv(PROJECT_ROOT / ".env")
    run_pipeline()


if __name__ == "__main__":
    main()
