from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extractor import extract_invoice, extract_receipt
from .models import GenericTableRecord


def get_document_text(
    document: dict[str, Any],
) -> str:
    """
    The real ingestion output supplies document text through
    the top-level extracted_text property.
    """
    return str(
        document.get("extracted_text", "")
    ).strip()


def normalize_ocr_confidence(
    value: Any,
) -> float | None:
    """
    Ingestion output provides OCR confidence as a
    percentage, for example 90.15.

    The Extraction Agent uses values from 0.0 to 1.0.
    """
    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence > 1.0:
        confidence = confidence / 100.0

    return max(0.0, min(1.0, confidence))


def identify_document_type(
    document: dict[str, Any],
) -> str:
    """
    Route using ingestion metadata first.

    category:
      invoice          -> invoice extraction
      receipt + image  -> expense receipt extraction
      bank             -> structured table
      expense          -> structured table
      purchase_order   -> structured table
      other            -> structured table

    category=receipt and document_type=csv represents
    goods receipts, not an image receipt.
    """
    category = str(
        document.get("category", "")
    ).strip().lower()

    document_type = str(
        document.get("document_type", "")
    ).strip().lower()

    if category == "invoice":
        return "invoice"

    if (
        category == "receipt"
        and document_type == "image"
    ):
        return "expense_receipt"

    if (
        category == "receipt"
        and document_type == "csv"
    ):
        return "goods_receipt_table"

    if category == "purchase_order":
        return "purchase_order_table"

    if category == "bank":
        return "bank_table"

    if category == "expense":
        return "expense_table"

    if document_type == "csv":
        return "generic_table"

    return "unsupported"


def table_rows_as_objects(
    table: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert:

        headers = ["id", "amount"]
        rows = [[1, 2500]]

    into:

        [{"id": 1, "amount": 2500}]
    """
    headers = table.get("headers", [])
    source_rows = table.get("rows", [])

    converted_rows: list[dict[str, Any]] = []

    for source_row in source_rows:
        row_object: dict[str, Any] = {}

        for index, header in enumerate(headers):
            row_object[header] = (
                source_row[index]
                if index < len(source_row)
                else None
            )

        converted_rows.append(row_object)

    return converted_rows


def create_table_record(
    document: dict[str, Any],
    routed_type: str,
) -> GenericTableRecord:
    """
    Preserve CSV values without asking the LLM to guess
    invoice meanings.
    """
    tables = document.get("tables", [])

    if not tables:
        return GenericTableRecord(
            table_id=routed_type,
            columns=[],
            rows=[],
            warnings=[
                "No structured table was supplied."
            ],
            requires_human_review=True,
        )

    table = tables[0]
    headers = table.get("headers", [])

    return GenericTableRecord(
        table_id=(
            table.get("table_id")
            or routed_type
        ),
        columns=headers,
        rows=table_rows_as_objects(table),
        warnings=[],
        requires_human_review=False,
    )


def build_invoice_context(
    document: dict[str, Any],
) -> str:
    text = get_document_text(document)

    ocr_confidence = normalize_ocr_confidence(
        document.get("ocr_confidence")
    )

    metadata = (
        f"Document ID: "
        f"{document.get('document_id')}\n"
        f"Source file: "
        f"{document.get('source_file')}\n"
        f"Category: "
        f"{document.get('category')}\n"
        f"OCR confidence: "
        f"{ocr_confidence}\n"
    )

    return (
        f"{metadata}\n"
        "DOCUMENT TEXT BEGIN\n"
        f"{text}\n"
        "DOCUMENT TEXT END"
    )


def build_receipt_context(
    document: dict[str, Any],
) -> str:
    text = get_document_text(document)

    ocr_confidence = normalize_ocr_confidence(
        document.get("ocr_confidence")
    )

    return (
        f"Document ID: "
        f"{document.get('document_id')}\n"
        f"Source file: "
        f"{document.get('source_file')}\n"
        f"OCR confidence: "
        f"{ocr_confidence}\n\n"
        "RECEIPT TEXT BEGIN\n"
        f"{text}\n"
        "RECEIPT TEXT END"
    )


def process_ingestion_file(
    input_path: str,
) -> list[dict[str, Any]]:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file was not found: "
            f"{input_file.resolve()}"
        )

    ingestion_data = json.loads(
        input_file.read_text(
            encoding="utf-8"
        )
    )

    batch_id = ingestion_data.get("batch_id")
    results: list[dict[str, Any]] = []

    for document in ingestion_data.get(
        "documents",
        [],
    ):
        document_id = document.get(
            "document_id"
        )

        source_file = document.get(
            "source_file"
        )

        content_hash = document.get(
            "content_hash"
        )

        if document.get("status") != "success":
            results.append(
                {
                    "batch_id": batch_id,
                    "document_id": document_id,
                    "source_file": source_file,
                    "content_hash": content_hash,
                    "status": "skipped",
                    "reason": (
                        document.get("errors")
                        or ["Ingestion failed"]
                    ),
                }
            )
            continue

        routed_type = identify_document_type(
            document
        )

        try:
            if routed_type == "invoice":
                record = extract_invoice(
                    raw_text=build_invoice_context(
                        document
                    ),
                    ocr_tables=[],
                )

            elif routed_type == "expense_receipt":
                record = extract_receipt(
                    raw_text=build_receipt_context(
                        document
                    )
                )

            elif routed_type in {
                "bank_table",
                "expense_table",
                "goods_receipt_table",
                "purchase_order_table",
                "generic_table",
            }:
                record = create_table_record(
                    document=document,
                    routed_type=routed_type,
                )

            else:
                results.append(
                    {
                        "batch_id": batch_id,
                        "document_id": document_id,
                        "source_file": source_file,
                        "content_hash": content_hash,
                        "status": "unsupported",
                        "reason": (
                            "No supported route for "
                            f"category="
                            f"{document.get('category')} "
                            f"and document_type="
                            f"{document.get('document_type')}."
                        ),
                    }
                )
                continue

            results.append(
                {
                    "batch_id": batch_id,
                    "document_id": document_id,
                    "source_file": source_file,
                    "content_hash": content_hash,
                    "input_category": document.get(
                        "category"
                    ),
                    "input_document_type": (
                        document.get(
                            "document_type"
                        )
                    ),
                    "ocr_confidence": (
                        normalize_ocr_confidence(
                            document.get(
                                "ocr_confidence"
                            )
                        )
                    ),
                    "status": "success",
                    "record": record.model_dump(
                        mode="json"
                    ),
                }
            )

        except Exception as error:
            results.append(
                {
                    "batch_id": batch_id,
                    "document_id": document_id,
                    "source_file": source_file,
                    "content_hash": content_hash,
                    "status": "extraction_failed",
                    "reason": str(error),
                }
            )

    return results
