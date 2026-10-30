import logging
from pathlib import Path

from src.anomaly.agent import AnomalyReport
from src.explanation.agent import run_explanation


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "data" / "output" / "matching_output.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "explanation_output.json"
ANOMALY_FILE = PROJECT_ROOT / "data" / "output" / "anomaly_output.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    anomalies = (
        AnomalyReport.model_validate_json(ANOMALY_FILE.read_text(encoding="utf-8"))
        if ANOMALY_FILE.is_file()
        else None
    )
    report = run_explanation(INPUT_FILE, OUTPUT_FILE, anomalies)
    print("\n=== Explanation Summary ===")
    print(f"Exceptions explained: {report.explanation_count}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
