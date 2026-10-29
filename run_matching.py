import logging
from pathlib import Path

from dotenv import load_dotenv

from src.matching.agent import run_matching


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

INPUT_FILE = PROJECT_ROOT / "data" / "output" / "extraction_output.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "matching_output.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    batch = run_matching(INPUT_FILE, OUTPUT_FILE)

    invoice_statuses = {
        status: sum(invoice.status == status for invoice in batch.invoices)
        for status in ("matched", "exception", "review")
    }
    payment_statuses = {
        status: sum(payment.status == status for payment in batch.bank_payments)
        for status in ("matched", "exception", "review", "ignored")
    }

    print("\n=== Matching Summary ===")
    print(f"Extraction batch: {batch.extraction_id}")
    print(f"Invoices: {len(batch.invoices)}")
    print(f"Invoice results: {invoice_statuses}")
    print(f"Bank payment results: {payment_statuses}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
