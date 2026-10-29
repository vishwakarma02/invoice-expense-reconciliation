import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .ingestion_adapter import process_ingestion_file


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Run the Extraction Agent using "
            "the Ingestion Agent output."
        )
    )

    parser.add_argument(
        "--input",
        default="data/output/ingestion_output.json",
        help="Path to the Ingestion Agent output JSON.",
    )

    parser.add_argument(
        "--output",
        default="data/output/extraction_results.json",
        help="Path for the Extraction Agent output JSON.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ingestion output was not found: "
            f"{input_path.resolve()}"
        )

    results = process_ingestion_file(
        str(input_path)
    )

    output_json = json.dumps(
        results,
        indent=2,
        ensure_ascii=False,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        output_json,
        encoding="utf-8",
    )

    print(output_json)
    print(
        f"\nExtraction results saved to: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()