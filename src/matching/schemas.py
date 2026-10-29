from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.extraction.schemas import SourceEvidence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatchingConfig(StrictModel):
    currency: str = "INR"
    unit_price_tolerance_percent: Decimal = Field(
        default=Decimal("1.0"), ge=Decimal("0")
    )
    amount_tolerance: Decimal = Field(default=Decimal("1.00"), ge=Decimal("0"))
    minimum_field_confidence: float = Field(default=0.6, ge=0, le=1)


class MatchIssue(StrictModel):
    code: str
    severity: Literal["exception", "review"]
    message: str
    evidence: list[SourceEvidence] = Field(default_factory=list)


class LineMatchResult(StrictModel):
    invoice_line_number: str | None = None
    purchase_order_line_number: str | None = None
    item_description: str | None = None
    item_code: str | None = None
    invoiced_quantity: Decimal | None = None
    ordered_quantity: Decimal | None = None
    received_quantity: Decimal | None = None
    invoiced_unit_price: Decimal | None = None
    ordered_unit_price: Decimal | None = None
    unit_price_variance_percent: Decimal | None = None
    issues: list[MatchIssue] = Field(default_factory=list)
    evidence: list[SourceEvidence] = Field(default_factory=list)


class InvoiceMatchResult(StrictModel):
    invoice_document_id: str
    invoice_source_file: str
    invoice_number: str | None = None
    normalized_invoice_number: str | None = None
    purchase_order_number: str | None = None
    vendor_id: str | None = None
    purchase_order_document_id: str | None = None
    receipt_document_ids: list[str] = Field(default_factory=list)
    status: Literal["matched", "exception", "review"]
    duplicate_of_document_id: str | None = None
    lines: list[LineMatchResult] = Field(default_factory=list)
    issues: list[MatchIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_warnings: list[str] = Field(default_factory=list)


class BankPaymentMatchResult(StrictModel):
    transaction_document_id: str
    transaction_source_file: str
    transaction_date: str | None = None
    narration: str | None = None
    debit_amount: Decimal | None = None
    matched_invoice_document_id: str | None = None
    matched_invoice_number: str | None = None
    expected_invoice_amount: Decimal | None = None
    amount_variance: Decimal | None = None
    status: Literal["matched", "exception", "review", "ignored"]
    issues: list[MatchIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[SourceEvidence] = Field(default_factory=list)


class MatchingBatch(StrictModel):
    matching_id: str
    extraction_id: str
    config: MatchingConfig
    invoices: list[InvoiceMatchResult] = Field(default_factory=list)
    bank_payments: list[BankPaymentMatchResult] = Field(default_factory=list)
