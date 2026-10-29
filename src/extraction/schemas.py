from datetime import date
from decimal import Decimal
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")
Confidence = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceEvidence(StrictModel):
    document_id: str
    source_file: str
    quote: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    table_row: int | None = Field(default=None, ge=1)
    column: str | None = None


class ExtractedField(StrictModel, Generic[T]):
    value: T | None = None
    raw_value: str | None = None
    confidence: Confidence | None = None
    evidence: list[SourceEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _empty_str_field() -> ExtractedField[str]:
    return ExtractedField[str]()


def _empty_decimal_field() -> ExtractedField[Decimal]:
    return ExtractedField[Decimal]()


def _empty_date_field() -> ExtractedField[date]:
    return ExtractedField[date]()


def _empty_int_field() -> ExtractedField[int]:
    return ExtractedField[int]()


class LineItem(StrictModel):
    line_number: ExtractedField[str] = Field(default_factory=_empty_str_field)
    description: ExtractedField[str]
    item_code: ExtractedField[str] = Field(default_factory=_empty_str_field)
    hsn_sac: ExtractedField[str] = Field(default_factory=_empty_str_field)
    quantity: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    unit: ExtractedField[str] = Field(default_factory=_empty_str_field)
    unit_price: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    tax_rate: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    line_total: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)


class TaxLine(StrictModel):
    tax_type: ExtractedField[str]
    rate: ExtractedField[Decimal] = Field(default_factory=ExtractedField)
    amount: ExtractedField[Decimal]


class InvoiceRecord(StrictModel):
    record_type: Literal["invoice"] = "invoice"
    vendor_name: ExtractedField[str] = Field(default_factory=_empty_str_field)
    vendor_id: ExtractedField[str] = Field(default_factory=_empty_str_field)
    vendor_gstin: ExtractedField[str] = Field(default_factory=_empty_str_field)
    buyer_gstin: ExtractedField[str] = Field(default_factory=_empty_str_field)
    invoice_number: ExtractedField[str] = Field(default_factory=_empty_str_field)
    invoice_date: ExtractedField[date] = Field(default_factory=_empty_date_field)
    po_number: ExtractedField[str] = Field(default_factory=_empty_str_field)
    currency: ExtractedField[str] = Field(default_factory=_empty_str_field)
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    taxable_value: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    taxes: list[TaxLine] = Field(default_factory=list)
    grand_total: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    warnings: list[str] = Field(default_factory=list)


class PurchaseOrderRecord(StrictModel):
    record_type: Literal["purchase_order"] = "purchase_order"
    po_number: ExtractedField[str]
    vendor_id: ExtractedField[str] = Field(default_factory=_empty_str_field)
    po_date: ExtractedField[date] = Field(default_factory=_empty_date_field)
    currency: ExtractedField[str] = Field(default_factory=_empty_str_field)
    line_items: list[LineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReceiptRecord(StrictModel):
    record_type: Literal["receipt"] = "receipt"
    receipt_number: ExtractedField[str] = Field(default_factory=_empty_str_field)
    receipt_type: ExtractedField[str] = Field(default_factory=_empty_str_field)
    po_number: ExtractedField[str] = Field(default_factory=_empty_str_field)
    receipt_date: ExtractedField[date] = Field(default_factory=_empty_date_field)
    vendor_name: ExtractedField[str] = Field(default_factory=_empty_str_field)
    line_items: list[LineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BankTransactionRecord(StrictModel):
    record_type: Literal["bank_transaction"] = "bank_transaction"
    transaction_date: ExtractedField[date] = Field(default_factory=_empty_date_field)
    narration: ExtractedField[str] = Field(default_factory=_empty_str_field)
    debit: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    credit: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    balance: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    warnings: list[str] = Field(default_factory=list)


class ExpenseClaimRecord(StrictModel):
    record_type: Literal["expense_claim"] = "expense_claim"
    claim_id: ExtractedField[str] = Field(default_factory=_empty_str_field)
    employee_id: ExtractedField[str] = Field(default_factory=_empty_str_field)
    claim_date: ExtractedField[date] = Field(default_factory=_empty_date_field)
    category: ExtractedField[str] = Field(default_factory=_empty_str_field)
    merchant: ExtractedField[str] = Field(default_factory=_empty_str_field)
    amount_claimed: ExtractedField[Decimal] = Field(default_factory=_empty_decimal_field)
    currency: ExtractedField[str] = Field(default_factory=_empty_str_field)
    receipt_file: ExtractedField[str] = Field(default_factory=_empty_str_field)
    warnings: list[str] = Field(default_factory=list)


class VendorRecord(StrictModel):
    record_type: Literal["vendor"] = "vendor"
    vendor_id: ExtractedField[str] = Field(default_factory=_empty_str_field)
    vendor_name: ExtractedField[str] = Field(default_factory=_empty_str_field)
    gstin: ExtractedField[str] = Field(default_factory=_empty_str_field)
    state_code: ExtractedField[str] = Field(default_factory=_empty_str_field)
    city: ExtractedField[str] = Field(default_factory=_empty_str_field)
    payment_terms_days: ExtractedField[int] = Field(default_factory=_empty_int_field)
    warnings: list[str] = Field(default_factory=list)


FinancialRecord = Annotated[
    InvoiceRecord
    | PurchaseOrderRecord
    | ReceiptRecord
    | BankTransactionRecord
    | ExpenseClaimRecord
    | VendorRecord,
    Field(discriminator="record_type"),
]


class DocumentExtraction(StrictModel):
    document_id: str
    source_file: str
    source_category: str
    status: Literal["success", "partial", "failed", "unsupported"]
    records: list[FinancialRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ExtractionBatch(StrictModel):
    extraction_id: str
    ingestion_batch_id: str
    documents: list[DocumentExtraction] = Field(default_factory=list)


class LLMField(StrictModel):
    value: str | None
    raw_value: str | None
    confidence: Confidence | None
    evidence_quote: str | None


class LLMLineItem(StrictModel):
    line_number: LLMField
    description: LLMField
    item_code: LLMField
    hsn_sac: LLMField
    quantity: LLMField
    unit: LLMField
    unit_price: LLMField
    tax_rate: LLMField
    line_total: LLMField


class LLMTaxLine(StrictModel):
    tax_type: LLMField
    rate: LLMField
    amount: LLMField


class LLMInvoiceExtraction(StrictModel):
    vendor_name: LLMField
    vendor_id: LLMField
    vendor_gstin: LLMField
    buyer_gstin: LLMField
    invoice_number: LLMField
    invoice_date: LLMField
    po_number: LLMField
    currency: LLMField
    line_items: list[LLMLineItem]
    subtotal: LLMField
    taxable_value: LLMField
    taxes: list[LLMTaxLine]
    grand_total: LLMField


class LLMReceiptExtraction(StrictModel):
    receipt_number: LLMField
    receipt_type: LLMField
    po_number: LLMField
    receipt_date: LLMField
    vendor_name: LLMField
    line_items: list[LLMLineItem]
