
import hashlib
import json
import logging
import os
import uuid

from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf as fitz
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from src.ingestion.schemas import (
    ExtractedTable,
    PageEvidence,
    IngestedDocument,
    IngestionBatch,
)

logger = logging.getLogger(__name__)

SUPPORTED_PDF = {".pdf"}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_CSV = {".csv"}


# ---------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------

def configure_tesseract() -> None:
    """Configure Tesseract using the optional .env setting."""
    tesseract_cmd = os.getenv("TESSERACT_CMD")

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# ---------------------------------------------------------
# File utilities
# ---------------------------------------------------------

def sha256_file(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def infer_category(file_path: Path) -> str:
    """Infer document category from its folder or filename."""
    parts = [part.lower() for part in file_path.parts]

    category_mapping = {
        "invoice": "invoice",
        "invoices": "invoice",
        "purchase_order": "purchase_order",
        "purchase_orders": "purchase_order",
        "po": "purchase_order",
        "pos": "purchase_order",
        "receipt": "receipt",
        "receipts": "receipt",
        "grn": "receipt",
        "service_entries": "receipt",
        "bank": "bank",
        "bank_statement": "bank",
        "bank_statements": "bank",
        "expense": "expense",
        "expenses": "expense",
        "expense_claim": "expense",
        "expense_claims": "expense",
    }

    for part in reversed(parts):
        if part in category_mapping:
            return category_mapping[part]

    filename = file_path.stem.lower()

    if "invoice" in filename or filename.startswith("inv-"):
        return "invoice"

    if "receipt" in filename or "grn" in filename:
        return "receipt"

    if "purchase_order" in filename or filename.startswith("po-"):
        return "purchase_order"

    if "bank" in filename or "statement" in filename:
        return "bank"

    if "expense" in filename or filename.startswith("exp-"):
        return "expense"

    return "other"


def discover_files(input_dir: Path) -> list[Path]:
    """Discover supported files recursively."""
    supported_extensions = (
        SUPPORTED_PDF | SUPPORTED_IMAGES | SUPPORTED_CSV
    )

    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
        return []

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


# ---------------------------------------------------------
# OCR utilities
# ---------------------------------------------------------

def prepare_image(image: Image.Image) -> Image.Image:
    """Prepare an image for OCR."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    return image


def get_ocr_confidence(image: Image.Image) -> float | None:
    """
    Calculate mean confidence for recognized OCR words.

    Tesseract confidence is a recognition-quality estimate,
    not a guarantee that extracted financial values are correct.
    """
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--psm 3",
    )

    valid_scores: list[float] = []

    for text, raw_score in zip(data["text"], data["conf"]):
        if not str(text).strip():
            continue

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue

        # Tesseract uses negative scores for non-word entries.
        if 0 <= score <= 100:
            valid_scores.append(score)

    if not valid_scores:
        return None

    return round(sum(valid_scores) / len(valid_scores), 2)


def run_ocr(
    image: Image.Image,
) -> tuple[str, float | None]:
    """Extract OCR text and its mean word confidence."""
    image = prepare_image(image)

    # image_to_data returns recognized words and confidence scores.
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--psm 3",
    )

    valid_scores: list[float] = []

    for word, raw_score in zip(data["text"], data["conf"]):
        if not str(word).strip():
            continue

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue

        if 0 <= score <= 100:
            valid_scores.append(score)

    # Use Tesseract's text extraction to preserve readable line layout.
    extracted_text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 3",
    ).strip()

    confidence = (
        round(sum(valid_scores) / len(valid_scores), 2)
        if valid_scores
        else None
    )

    return extracted_text, confidence


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

def extract_pdf(
    file_path: Path,
) -> tuple[str, list[ExtractedTable], list[PageEvidence]]:
    """Extract text, tables, and page-level OCR confidence from a PDF."""
    all_text: list[str] = []
    tables: list[ExtractedTable] = []
    page_evidence: list[PageEvidence] = []

    with fitz.open(str(file_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            extraction_method = "text_layer"
            ocr_confidence = None

            # Use OCR when the PDF text layer is absent or nearly empty.
            if len(page_text) < 20:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                image = Image.open(
                    BytesIO(pixmap.tobytes("png"))
                )

                page_text, ocr_confidence = run_ocr(image)
                extraction_method = "ocr"

            all_text.append(
                f"--- Page {page_number} ---\n{page_text}"
            )

            page_evidence.append(
                PageEvidence(
                    page_number=page_number,
                    extraction_method=extraction_method,
                    character_count=len(page_text),
                    ocr_confidence=ocr_confidence,
                )
            )

            # Extract tables when the installed PyMuPDF version supports it.
            find_tables = getattr(page, "find_tables", None)

            if callable(find_tables):
                try:
                    table_finder = find_tables()

                    for table in table_finder.tables:
                        extracted = table.extract()

                        if not extracted:
                            continue

                        headers = [
                            str(value) if value is not None else ""
                            for value in (extracted[0] or [])
                        ]

                        rows = [
                            [
                                value if value is not None else ""
                                for value in (row or [])
                            ]
                            for row in extracted[1:]
                        ]

                        tables.append(
                            ExtractedTable(
                                source=str(file_path),
                                page_number=page_number,
                                headers=headers,
                                rows=rows,
                            )
                        )

                except Exception:
                    logger.exception(
                        "Table extraction failed for %s, page %s",
                        file_path,
                        page_number,
                    )

    return "\n\n".join(all_text), tables, page_evidence


# ---------------------------------------------------------
# Image extraction
# ---------------------------------------------------------

def extract_image(
    file_path: Path,
) -> tuple[str, list[ExtractedTable], list[PageEvidence]]:
    """Extract OCR text and confidence from an image."""
    with Image.open(file_path) as source_image:
        image = source_image.convert("RGB")
        extracted_text, confidence = run_ocr(image)

    evidence = PageEvidence(
        page_number=1,
        extraction_method="ocr",
        character_count=len(extracted_text),
        ocr_confidence=confidence,
    )

    return extracted_text, [], [evidence]


# ---------------------------------------------------------
# CSV extraction
# ---------------------------------------------------------

def extract_csv(
    file_path: Path,
) -> tuple[str, list[ExtractedTable], list[PageEvidence]]:
    """Extract CSV headers, rows, and text."""
    dataframe = pd.read_csv(
        file_path,
        dtype=str,
        keep_default_na=False,
    )

    headers = [str(column) for column in dataframe.columns]
    rows: list[list[Any]] = dataframe.values.tolist()

    table = ExtractedTable(
        source=str(file_path),
        page_number=None,
        headers=headers,
        rows=rows,
    )

    extracted_text = dataframe.to_csv(index=False)

    evidence = PageEvidence(
        page_number=1,
        extraction_method="csv",
        character_count=len(extracted_text),
        ocr_confidence=None,
    )

    return extracted_text, [table], [evidence]


# ---------------------------------------------------------
# Document processing
# ---------------------------------------------------------

def process_file(
    file_path: Path,
    input_root: Path,
) -> IngestedDocument:
    """Process one supported file and return its structured result."""
    extension = file_path.suffix.lower()
    errors: list[str] = []

    try:
        source_file = file_path.relative_to(input_root).as_posix()
    except ValueError:
        source_file = file_path.as_posix()

    document_type = (
        "pdf" if extension in SUPPORTED_PDF
        else "image" if extension in SUPPORTED_IMAGES
        else "csv" if extension in SUPPORTED_CSV
        else "unknown"
    )

    try:
        content_hash = sha256_file(file_path)
    except Exception as exc:
        content_hash = ""
        errors.append(f"File hashing failed: {exc}")

    extracted_text = ""
    tables: list[ExtractedTable] = []
    pages: list[PageEvidence] = []

    try:
        if document_type == "pdf":
            extracted_text, tables, pages = extract_pdf(file_path)

        elif document_type == "image":
            extracted_text, tables, pages = extract_image(file_path)

        elif document_type == "csv":
            extracted_text, tables, pages = extract_csv(file_path)

        else:
            errors.append(f"Unsupported file type: {extension}")

    except Exception as exc:
        errors.append(f"Extraction failed: {type(exc).__name__}: {exc}")
        logger.exception("Failed to process %s", file_path)

    if errors and (extracted_text or tables or pages):
        status = "partial"
    elif errors:
        status = "failed"
    else:
        status = "success"

    # Average confidence across pages where OCR produced a valid score.
    # Text-layer PDF pages and CSV files are excluded.
    page_scores = [
        page.ocr_confidence
        for page in pages
        if page.extraction_method == "ocr"
        and page.ocr_confidence is not None
    ]

    document_confidence = (
        round(sum(page_scores) / len(page_scores), 2)
        if page_scores
        else None
    )

    return IngestedDocument(
        document_id=str(uuid.uuid4()),
        source_file=source_file,
        document_type=document_type,
        category=infer_category(file_path),
        content_hash=content_hash,
        status=status,
        extracted_text=extracted_text,
        tables=tables,
        pages=pages,
        ocr_confidence=document_confidence,
        errors=errors,
    )


# ---------------------------------------------------------
# Batch ingestion and JSON output
# ---------------------------------------------------------

def run_ingestion(
    input_dir: str | Path,
    output_file: str | Path,
) -> IngestionBatch:
    """Process all supported files and atomically write the JSON output."""
    configure_tesseract()

    input_root = Path(input_dir).resolve()
    output_path = Path(output_file).resolve()

    files = discover_files(input_root)

    batch = IngestionBatch(
        batch_id=str(uuid.uuid4()),
        documents=[
            process_file(file_path, input_root)
            for file_path in files
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(
                batch.model_dump(mode="json"),
                output,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    successful = sum(
        document.status == "success"
        for document in batch.documents
    )
    partial = sum(
        document.status == "partial"
        for document in batch.documents
    )
    failed = sum(
        document.status == "failed"
        for document in batch.documents
    )

    print(f"Input directory: {input_root}")
    print(f"Output file: {output_path}")
    print(f"Files discovered: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Partial: {partial}")
    print(f"Failed: {failed}")

    for document in batch.documents:
        if document.errors:
            print(
                f"[ERROR] {document.source_file}: "
                f"{'; '.join(document.errors)}"
            )

    return batch
