from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    raw_text: str | None = None
    missing_reason: str | None = None

    @model_validator(mode="after")
    def validate_missing_value(self):
        if self.value is None:
            self.confidence = 0.0
            self.raw_text = None
            self.missing_reason = (
                self.missing_reason
                or "Not present in source"
            )
        else:
            self.missing_reason = None

        return self


class InvoiceLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_code: ExtractedField[str]
    description: ExtractedField[str]
    quantity: ExtractedField[Decimal]
    unit_price: ExtractedField[Decimal]
    total: ExtractedField[Decimal]


class InvoiceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["invoice"] = "invoice"

    invoice_number: ExtractedField[str]
    invoice_date: ExtractedField[date]
    due_date: ExtractedField[date]

    vendor_name: ExtractedField[str]
    customer_name: ExtractedField[str]
    currency: ExtractedField[str]

    subtotal: ExtractedField[Decimal]
    tax: ExtractedField[Decimal]
    grand_total: ExtractedField[Decimal]

    line_items: list[InvoiceLineItem] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    requires_human_review: bool = False


class ReceiptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["receipt"] = "receipt"

    receipt_number: ExtractedField[str]
    receipt_date: ExtractedField[date]
    vendor_name: ExtractedField[str]
    currency: ExtractedField[str]
    total_paid: ExtractedField[Decimal]

    warnings: list[str] = Field(
        default_factory=list
    )

    requires_human_review: bool = False


class GenericTableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal[
        "generic_table"
    ] = "generic_table"

    table_id: str | None = None
    columns: list[str] = Field(
        default_factory=list
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    requires_human_review: bool = False