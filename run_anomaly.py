import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.anomaly import AnomalyAgent, AnomalyPolicy


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "data" / "output" / "extraction_output.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "anomaly_results.json"
MAX_INVOICE_AMOUNT = Decimal("100000")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Extraction results not found: {INPUT_FILE}")

    extraction_results: Any = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    if not isinstance(extraction_results, (list, dict)):
        raise ValueError("Extraction results must be an array or batch object.")

    report = AnomalyAgent(
        AnomalyPolicy(
            max_invoice_amount_by_currency={"INR": MAX_INVOICE_AMOUNT}
        )
    ).analyze(extraction_results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        OUTPUT_FILE.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError:
        logging.exception("Could not write anomaly report to %s", OUTPUT_FILE)
        raise

    findings_by_type: dict[str, int] = {}
    for finding in report.findings:
        findings_by_type[finding.anomaly_type] = (
            findings_by_type.get(finding.anomaly_type, 0) + 1
        )

    print("\n=== Anomaly Summary ===")
    print(f"Invoices analyzed: {report.analyzed_invoice_count}")
    print(f"Findings: {len(report.findings)}")
    print(f"Findings by type: {findings_by_type}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
