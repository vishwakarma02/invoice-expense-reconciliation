
from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentType = Literal["pdf", "image", "csv", "unknown"]
DocumentCategory = Literal[
    "invoice",
    "purchase_order",
    "receipt",
    "bank",
    "expense",
    "other",
]
ProcessingStatus = Literal["success", "partial", "failed"]


class ExtractedTable(BaseModel):
    source: str
    page_number: int | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)


class PageEvidence(BaseModel):
    page_number: int
    extraction_method: Literal["text_layer", "ocr", "csv", "none"]
    character_count: int = 0
    ocr_confidence: float | None = None


class IngestedDocument(BaseModel):
    document_id: str
    source_file: str
    document_type: DocumentType
    category: DocumentCategory
    content_hash: str
    status: ProcessingStatus

    extracted_text: str = ""
    tables: list[ExtractedTable] = Field(default_factory=list)
    pages: list[PageEvidence] = Field(default_factory=list)
    ocr_confidence: float | None = None
    errors: list[str] = Field(default_factory=list)


class IngestionBatch(BaseModel):
    batch_id: str
    documents: list[IngestedDocument] = Field(default_factory=list)
