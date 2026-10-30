import json
import logging
import re
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from src.extraction.schemas import (
    BankTransactionRecord,
    ExtractionBatch,
    ExtractedField,
    InvoiceRecord,
    LineItem,
    PurchaseOrderRecord,
    ReceiptRecord,
    SourceEvidence,
    TaxLine,
    VendorRecord,
)
from src.matching.schemas import (
    BankPaymentMatchResult,
    InvoiceMatchResult,
    LineMatchResult,
    MatchIssue,
    MatchingBatch,
    MatchingConfig,
)


logger = logging.getLogger(__name__)
ZERO = Decimal("0")
HUNDRED = Decimal("100")
IssueSeverity = Literal["exception", "review"]


def normalize_identifier(value: str | None) -> str:
    """Normalize punctuation and leading zeroes in identifier components."""
    if not value:
        return ""
    parts = re.findall(r"[A-Za-z]+|\d+", value.upper())
    return "".join(str(int(part)) if part.isdigit() else part for part in parts)


def _text_key(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def normalize_currency(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[\W_]+", "", value.casefold())
    if normalized in {
        "inr",
        "rs",
        "rupee",
        "rupees",
        "indianrupee",
        "indianrupees",
    } or value.strip() == "₹":
        return "INR"
    if normalized in {"usd", "usdollar", "usdollars", "dollar", "dollars"} or value.strip() == "$":
        return "USD"
    if normalized in {"eur", "euro", "euros"} or value.strip() == "€":
        return "EUR"
    if normalized in {"gbp", "pound", "pounds"} or value.strip() == "£":
        return "GBP"
    return normalized.upper()


def _field_evidence(*fields: ExtractedField[Any] | None) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    seen: set[tuple[Any, ...]] = set()
    for field in fields:
        if field is None:
            continue
        for item in field.evidence:
            key = (
                item.document_id,
                item.source_file,
                item.page_number,
                item.table_row,
                item.column,
                item.quote,
            )
            if key not in seen:
                evidence.append(item)
                seen.add(key)
    return evidence


def _unique_evidence(items: list[SourceEvidence]) -> list[SourceEvidence]:
    unique: list[SourceEvidence] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        key = (
            item.document_id,
            item.source_file,
            item.page_number,
            item.table_row,
            item.column,
            item.quote,
        )
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _add_issue(
    issues: list[MatchIssue],
    code: str,
    severity: IssueSeverity,
    message: str,
    evidence: list[SourceEvidence] | None = None,
    *,
    expected_amount: Decimal | None = None,
    found_amount: Decimal | None = None,
    currency: str | None = None,
) -> None:
    if any(issue.code == code and issue.message == message for issue in issues):
        return
    issues.append(
        MatchIssue(
            code=code,
            severity=severity,
            message=message,
            expected_amount=expected_amount,
            found_amount=found_amount,
            currency=currency,
            evidence=evidence or [],
        )
    )


def _vendor_state(vendor: VendorRecord | None, gstin: str | None) -> str | None:
    if vendor and vendor.state_code.value:
        return vendor.state_code.value.strip().zfill(2)
    if gstin and len(gstin) >= 2 and gstin[:2].isdigit():
        return gstin[:2]
    return None


class MatchingAgent:
    """Perform reproducible PO/invoice/receipt and invoice/payment checks."""

    def __init__(self, config: MatchingConfig | None = None) -> None:
        self.config = config or MatchingConfig()

    def match_batch(self, extraction: ExtractionBatch) -> MatchingBatch:
        invoices: list[tuple[Any, InvoiceRecord]] = []
        purchase_orders: list[tuple[Any, PurchaseOrderRecord]] = []
        receipts: list[tuple[Any, ReceiptRecord]] = []
        bank_transactions: list[tuple[Any, BankTransactionRecord]] = []
        vendors: list[VendorRecord] = []
        document_status: dict[str, str] = {}

        for document in extraction.documents:
            document_status[document.document_id] = document.status
            for record in document.records:
                if isinstance(record, InvoiceRecord):
                    invoices.append((document, record))
                elif isinstance(record, PurchaseOrderRecord):
                    purchase_orders.append((document, record))
                elif isinstance(record, ReceiptRecord):
                    receipts.append((document, record))
                elif isinstance(record, BankTransactionRecord):
                    bank_transactions.append((document, record))
                elif isinstance(record, VendorRecord):
                    vendors.append(record)

        vendor_by_id = {
            normalize_identifier(vendor.vendor_id.value): vendor
            for vendor in vendors
            if vendor.vendor_id.value
        }
        vendor_by_gstin = {
            normalize_identifier(vendor.gstin.value): vendor
            for vendor in vendors
            if vendor.gstin.value
        }
        vendor_by_name = {
            _text_key(vendor.vendor_name.value): vendor
            for vendor in vendors
            if vendor.vendor_name.value
        }

        duplicate_map = self._find_duplicates(invoices, vendor_by_gstin, vendor_by_name)
        invoice_results: list[InvoiceMatchResult] = []
        for document, invoice in invoices:
            result = self._match_invoice(
                document,
                invoice,
                purchase_orders,
                receipts,
                vendor_by_id,
                vendor_by_gstin,
                vendor_by_name,
                document_status.get(document.document_id, "failed"),
            )
            duplicate_of = duplicate_map.get(document.document_id)
            if duplicate_of:
                result.duplicate_of_document_id = duplicate_of
                _add_issue(
                    result.issues,
                    "DUPLICATE_INVOICE",
                    "exception",
                    "This invoice has the same vendor and normalized invoice number "
                    f"as document {duplicate_of}.",
                    _field_evidence(invoice.invoice_number, invoice.vendor_gstin),
                )
            result.status = self._invoice_status(result.issues)
            invoice_results.append(result)

        payment_results = self._match_bank_payments(
            bank_transactions,
            invoices,
            invoice_results,
            vendor_by_id,
            vendor_by_gstin,
            vendor_by_name,
            document_status,
        )

        return MatchingBatch(
            matching_id=str(uuid.uuid4()),
            extraction_id=extraction.extraction_id,
            config=self.config,
            invoices=invoice_results,
            bank_payments=payment_results,
        )

    def _find_duplicates(
        self,
        invoices: list[tuple[Any, InvoiceRecord]],
        vendor_by_gstin: dict[str, VendorRecord],
        vendor_by_name: dict[str, VendorRecord],
    ) -> dict[str, str]:
        seen: dict[tuple[str, str], str] = {}
        duplicates: dict[str, str] = {}
        for document, invoice in invoices:
            invoice_number = normalize_identifier(invoice.invoice_number.value)
            if not invoice_number:
                continue
            vendor = self._resolve_vendor(invoice, {}, vendor_by_gstin, vendor_by_name)
            vendor_key = (
                normalize_identifier(vendor.vendor_id.value)
                if vendor and vendor.vendor_id.value
                else normalize_identifier(invoice.vendor_gstin.value)
                or _text_key(invoice.vendor_name.value)
            )
            if not vendor_key:
                continue
            key = (vendor_key, invoice_number)
            first_document_id = seen.get(key)
            if first_document_id:
                duplicates[document.document_id] = first_document_id
            else:
                seen[key] = document.document_id
        return duplicates

    def _resolve_vendor(
        self,
        invoice: InvoiceRecord,
        vendor_by_id: dict[str, VendorRecord],
        vendor_by_gstin: dict[str, VendorRecord],
        vendor_by_name: dict[str, VendorRecord],
    ) -> VendorRecord | None:
        if invoice.vendor_id.value:
            vendor = vendor_by_id.get(normalize_identifier(invoice.vendor_id.value))
            if vendor:
                return vendor
        if invoice.vendor_gstin.value:
            vendor = vendor_by_gstin.get(
                normalize_identifier(invoice.vendor_gstin.value)
            )
            if vendor:
                return vendor
        if invoice.vendor_name.value:
            return vendor_by_name.get(_text_key(invoice.vendor_name.value))
        return None

    def _match_invoice(
        self,
        document: Any,
        invoice: InvoiceRecord,
        purchase_orders: list[tuple[Any, PurchaseOrderRecord]],
        receipts: list[tuple[Any, ReceiptRecord]],
        vendor_by_id: dict[str, VendorRecord],
        vendor_by_gstin: dict[str, VendorRecord],
        vendor_by_name: dict[str, VendorRecord],
        extraction_status: str,
    ) -> InvoiceMatchResult:
        issues: list[MatchIssue] = []
        warnings = list(invoice.warnings)
        vendor = self._resolve_vendor(
            invoice, vendor_by_id, vendor_by_gstin, vendor_by_name
        )
        vendor_id = (
            vendor.vendor_id.value
            if vendor and vendor.vendor_id.value
            else invoice.vendor_id.value
        )
        invoice_number = invoice.invoice_number.value
        normalized_invoice_number = normalize_identifier(invoice_number)

        if extraction_status != "success":
            _add_issue(
                issues,
                "EXTRACTION_NOT_SUCCESSFUL",
                "review",
                f"Source extraction status is {extraction_status}.",
            )
        self._check_confidence(
            issues,
            "invoice number",
            invoice.invoice_number,
            required=True,
        )
        self._check_confidence(
            issues,
            "invoice total",
            invoice.grand_total,
            required=True,
        )
        if not vendor:
            _add_issue(
                issues,
                "VENDOR_NOT_IDENTIFIED",
                "review",
                "Invoice vendor could not be resolved against the vendor master.",
                _field_evidence(
                    invoice.vendor_id, invoice.vendor_gstin, invoice.vendor_name
                ),
            )
        elif (
            invoice.vendor_id.value
            and vendor.vendor_id.value
            and normalize_identifier(invoice.vendor_id.value)
            != normalize_identifier(vendor.vendor_id.value)
        ):
            _add_issue(
                issues,
                "VENDOR_MISMATCH",
                "exception",
                f"Invoice vendor ID {invoice.vendor_id.value} conflicts with vendor "
                f"master ID {vendor.vendor_id.value}.",
                _field_evidence(invoice.vendor_id, invoice.vendor_gstin),
            )

        po_value = invoice.po_number.value
        if not po_value:
            _add_issue(
                issues,
                "MISSING_PO",
                "exception",
                "Invoice does not contain a purchase-order reference.",
                _field_evidence(invoice.po_number),
            )
            po_record = None
            po_document = None
        else:
            po_matches = [
                (po_doc, po)
                for po_doc, po in purchase_orders
                if normalize_identifier(po.po_number.value)
                == normalize_identifier(po_value)
            ]
            if len(po_matches) == 1:
                po_document, po_record = po_matches[0]
            elif not po_matches:
                po_document, po_record = None, None
                _add_issue(
                    issues,
                    "PO_NOT_FOUND",
                    "exception",
                    f"Referenced purchase order {po_value} was not extracted.",
                    _field_evidence(invoice.po_number),
                )
            else:
                po_document, po_record = None, None
                _add_issue(
                    issues,
                    "AMBIGUOUS_PO",
                    "review",
                    f"More than one purchase order normalizes to {po_value}.",
                    _field_evidence(invoice.po_number),
                )

        if po_record and vendor_id and po_record.vendor_id.value:
            if normalize_identifier(vendor_id) != normalize_identifier(
                po_record.vendor_id.value
            ):
                _add_issue(
                    issues,
                    "VENDOR_MISMATCH",
                    "exception",
                    f"Invoice vendor {vendor_id} differs from PO vendor "
                    f"{po_record.vendor_id.value}.",
                    _field_evidence(
                        invoice.vendor_id,
                        invoice.vendor_gstin,
                        po_record.vendor_id,
                    ),
                )
        if po_record and invoice.currency.value:
            if normalize_currency(invoice.currency.value) != normalize_currency(
                self.config.currency
            ):
                _add_issue(
                    issues,
                    "CURRENCY_MISMATCH",
                    "exception",
                    f"Invoice currency {invoice.currency.value} differs from "
                    f"configured currency {self.config.currency}.",
                    _field_evidence(invoice.currency),
                )
        elif not invoice.currency.value:
            warnings.append("Invoice currency is not stated; configured currency assumed.")

        receipt_matches = (
            [
                (receipt_doc, receipt)
                for receipt_doc, receipt in receipts
                if po_record
                and normalize_identifier(receipt.po_number.value)
                == normalize_identifier(po_record.po_number.value)
            ]
            if po_record
            else []
        )
        receipt_ids = list(dict.fromkeys(doc.document_id for doc, _ in receipt_matches))
        if po_record and not receipt_matches:
            _add_issue(
                issues,
                "MISSING_RECEIPT",
                "exception",
                f"No receipt or service-entry record was found for PO "
                f"{po_record.po_number.value}.",
                _field_evidence(invoice.po_number, po_record.po_number),
            )

        line_results: list[LineMatchResult] = []
        matched_line_pairs: list[tuple[LineItem, LineItem]] = []
        for invoice_line in invoice.line_items:
            line_result, po_line = self._match_line(
                invoice_line,
                po_record,
                receipt_matches,
                invoice.currency.value or self.config.currency,
            )
            line_results.append(line_result)
            issues.extend(line_result.issues)
            if po_line:
                matched_line_pairs.append((invoice_line, po_line))

        if not invoice.line_items:
            _add_issue(
                issues,
                "INVOICE_LINES_MISSING",
                "review",
                "Invoice contains no extracted line items to compare with the PO.",
                _field_evidence(invoice.taxable_value),
            )

        self._check_taxes(
            invoice,
            matched_line_pairs,
            vendor,
            issues,
            warnings,
        )
        self._check_invoice_total(invoice, issues)

        return InvoiceMatchResult(
            invoice_document_id=document.document_id,
            invoice_source_file=document.source_file,
            invoice_number=invoice_number,
            normalized_invoice_number=normalized_invoice_number or None,
            purchase_order_number=po_value,
            vendor_id=vendor_id,
            purchase_order_document_id=(
                po_document.document_id if po_document else None
            ),
            receipt_document_ids=receipt_ids,
            status=self._invoice_status(issues),
            lines=line_results,
            issues=self._deduplicate_issues(issues),
            warnings=list(dict.fromkeys(warnings)),
            source_warnings=list(document.warnings),
        )

    def _check_confidence(
        self,
        issues: list[MatchIssue],
        label: str,
        field: ExtractedField[Any],
        *,
        required: bool,
    ) -> None:
        if field.value is None:
            if required:
                _add_issue(
                    issues,
                    "REQUIRED_FIELD_MISSING",
                    "review",
                    f"Required {label} is missing or could not be parsed.",
                    _field_evidence(field),
                )
            return
        if (
            field.confidence is not None
            and field.confidence < self.config.minimum_field_confidence
        ):
            _add_issue(
                issues,
                "LOW_EXTRACTION_CONFIDENCE",
                "review",
                f"{label.capitalize()} confidence {field.confidence:.2f} is below "
                f"the configured minimum {self.config.minimum_field_confidence:.2f}.",
                _field_evidence(field),
            )

    def _find_po_line(
        self,
        invoice_line: LineItem,
        purchase_order: PurchaseOrderRecord,
    ) -> LineItem | None:
        invoice_code = normalize_identifier(invoice_line.item_code.value)
        if invoice_code:
            candidates = [
                item
                for item in purchase_order.line_items
                if normalize_identifier(item.item_code.value) == invoice_code
            ]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return None

        invoice_description = _text_key(invoice_line.description.value)
        if invoice_description:
            candidates = [
                item
                for item in purchase_order.line_items
                if _text_key(item.description.value) == invoice_description
            ]
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                return None

        if not invoice_description:
            line_number = invoice_line.line_number.value
            if line_number:
                candidates = [
                    item
                    for item in purchase_order.line_items
                    if item.line_number.value == line_number
                    and not item.description.value
                ]
                if len(candidates) == 1:
                    return candidates[0]
        return None

    def _line_receipts(
        self,
        po_line: LineItem,
        receipts: list[tuple[Any, ReceiptRecord]],
    ) -> tuple[Decimal, list[SourceEvidence]]:
        identity = normalize_identifier(po_line.item_code.value) or _text_key(
            po_line.description.value
        )
        quantity = ZERO
        evidence: list[SourceEvidence] = []
        for _, receipt in receipts:
            for line in receipt.line_items:
                line_identity = normalize_identifier(line.item_code.value) or _text_key(
                    line.description.value
                )
                if identity and line_identity == identity and line.quantity.value is not None:
                    quantity += line.quantity.value
                    evidence.extend(
                        _field_evidence(line.quantity, line.item_code, line.description)
                    )
        return quantity, evidence

    def _match_line(
        self,
        invoice_line: LineItem,
        purchase_order: PurchaseOrderRecord | None,
        receipts: list[tuple[Any, ReceiptRecord]],
        currency: str,
    ) -> tuple[LineMatchResult, LineItem | None]:
        issues: list[MatchIssue] = []
        evidence = _field_evidence(
            invoice_line.line_number,
            invoice_line.item_code,
            invoice_line.description,
            invoice_line.quantity,
            invoice_line.unit_price,
            invoice_line.line_total,
        )
        po_line = (
            self._find_po_line(invoice_line, purchase_order)
            if purchase_order
            else None
        )
        received_quantity: Decimal | None = None
        if purchase_order and not po_line:
            _add_issue(
                issues,
                "PO_LINE_NOT_FOUND",
                "exception",
                f"Invoice line {invoice_line.description.value or invoice_line.line_number.value!r} "
                "could not be matched to a unique PO line.",
                evidence,
            )
        elif po_line:
            po_evidence = _field_evidence(
                po_line.item_code,
                po_line.description,
                po_line.quantity,
                po_line.unit_price,
                po_line.tax_rate,
            )
            evidence.extend(po_evidence)
            received_quantity, receipt_evidence = self._line_receipts(
                po_line, receipts
            )
            evidence.extend(receipt_evidence)
            if not receipt_evidence:
                _add_issue(
                    issues,
                    "RECEIPT_LINE_NOT_FOUND",
                    "exception",
                    f"No receipt quantity was found for PO item "
                    f"{po_line.item_code.value or po_line.description.value}.",
                    po_evidence,
                )
            if invoice_line.quantity.value is None:
                _add_issue(
                    issues,
                    "INVOICE_QUANTITY_MISSING",
                    "review",
                    "Invoice line quantity is missing or could not be parsed.",
                    evidence,
                )
            else:
                if po_line.quantity.value is None:
                    _add_issue(
                        issues,
                        "PO_QUANTITY_MISSING",
                        "review",
                        "Matched PO line has no usable ordered quantity.",
                        po_evidence,
                    )
                elif invoice_line.quantity.value > po_line.quantity.value:
                    _add_issue(
                        issues,
                        "QUANTITY_EXCEEDS_ORDERED",
                        "exception",
                        f"Invoiced quantity {invoice_line.quantity.value} exceeds "
                        f"ordered quantity {po_line.quantity.value}.",
                        evidence,
                    )
                if received_quantity is not None and invoice_line.quantity.value > received_quantity:
                    _add_issue(
                        issues,
                        "QUANTITY_EXCEEDS_RECEIVED",
                        "exception",
                        f"Invoiced quantity {invoice_line.quantity.value} exceeds "
                        f"received quantity {received_quantity}.",
                        evidence,
                    )
            if invoice_line.unit_price.value is None or po_line.unit_price.value is None:
                _add_issue(
                    issues,
                    "UNIT_PRICE_UNAVAILABLE",
                    "review",
                    "Invoice or PO unit price is missing or could not be parsed.",
                    evidence,
                )
            elif po_line.unit_price.value == ZERO:
                _add_issue(
                    issues,
                    "PO_UNIT_PRICE_ZERO",
                    "review",
                    "PO unit price is zero; percentage variance is undefined.",
                    evidence,
                )
            else:
                variance = (
                    abs(invoice_line.unit_price.value - po_line.unit_price.value)
                    / abs(po_line.unit_price.value)
                    * HUNDRED
                )
                if variance > self.config.unit_price_tolerance_percent:
                    _add_issue(
                        issues,
                        "PRICE_VARIANCE",
                        "exception",
                        f"Invoice unit price {invoice_line.unit_price.value} differs "
                        f"from PO price {po_line.unit_price.value} by "
                        f"{variance.quantize(Decimal('0.01'))}%, exceeding the "
                        f"{self.config.unit_price_tolerance_percent}% tolerance.",
                        evidence,
                        expected_amount=po_line.unit_price.value,
                        found_amount=invoice_line.unit_price.value,
                        currency=currency,
                    )
            if (
                invoice_line.quantity.value is not None
                and invoice_line.unit_price.value is not None
                and invoice_line.line_total.value is not None
            ):
                calculated_line_total = (
                    invoice_line.quantity.value * invoice_line.unit_price.value
                )
                if (
                    abs(calculated_line_total - invoice_line.line_total.value)
                    > self.config.amount_tolerance
                ):
                    _add_issue(
                        issues,
                        "LINE_TOTAL_MISMATCH",
                        "exception",
                        f"Quantity × unit price is {calculated_line_total}, but the "
                        f"invoice line total is {invoice_line.line_total.value}.",
                        evidence,
                        expected_amount=calculated_line_total,
                        found_amount=invoice_line.line_total.value,
                        currency=currency,
                    )

        variance_pct = None
        if (
            po_line
            and po_line.unit_price.value not in (None, ZERO)
            and invoice_line.unit_price.value is not None
        ):
            variance_pct = (
                abs(invoice_line.unit_price.value - po_line.unit_price.value)
                / abs(po_line.unit_price.value)
                * HUNDRED
            )
        result = LineMatchResult(
            invoice_line_number=invoice_line.line_number.value,
            purchase_order_line_number=po_line.line_number.value if po_line else None,
            item_description=invoice_line.description.value,
            item_code=invoice_line.item_code.value or (
                po_line.item_code.value if po_line else None
            ),
            invoiced_quantity=invoice_line.quantity.value,
            ordered_quantity=po_line.quantity.value if po_line else None,
            received_quantity=received_quantity,
            invoiced_unit_price=invoice_line.unit_price.value,
            ordered_unit_price=po_line.unit_price.value if po_line else None,
            unit_price_variance_percent=variance_pct,
            issues=issues,
            evidence=_unique_evidence(evidence),
        )
        return result, po_line

    def _check_taxes(
        self,
        invoice: InvoiceRecord,
        matched_lines: list[tuple[LineItem, LineItem]],
        vendor: VendorRecord | None,
        issues: list[MatchIssue],
        warnings: list[str],
    ) -> None:
        if not invoice.taxes:
            _add_issue(
                issues,
                "TAX_DATA_MISSING",
                "review",
                "No invoice tax lines were extracted.",
                _field_evidence(invoice.taxable_value, invoice.grand_total),
            )
            return
        if any(tax.amount.value is None for tax in invoice.taxes):
            _add_issue(
                issues,
                "TAX_DATA_INCOMPLETE",
                "review",
                "One or more invoice tax amounts are missing or invalid.",
                self._tax_evidence(invoice.taxes),
            )
            return

        if not matched_lines:
            warnings.append(
                "Tax rate comparison against PO lines was unavailable because no "
                "invoice lines were matched."
            )
            return
        if any(po_line.tax_rate.value is None for _, po_line in matched_lines):
            warnings.append(
                "PO tax rates are incomplete; expected invoice tax was not calculated."
            )
            return
        if any(
            invoice_line.line_total.value is None
            and (
                invoice_line.quantity.value is None
                or invoice_line.unit_price.value is None
            )
            for invoice_line, _ in matched_lines
        ):
            warnings.append(
                "Invoice taxable bases are incomplete; expected tax was not calculated."
            )
            return

        expected_tax = ZERO
        for invoice_line, po_line in matched_lines:
            taxable_base = invoice_line.line_total.value
            if taxable_base is None:
                taxable_base = (
                    invoice_line.quantity.value * invoice_line.unit_price.value
                )
            expected_tax += taxable_base * po_line.tax_rate.value / HUNDRED
        actual_tax = sum((tax.amount.value for tax in invoice.taxes), ZERO)
        if abs(actual_tax - expected_tax) > self.config.amount_tolerance:
            _add_issue(
                issues,
                "TAX_AMOUNT_MISMATCH",
                "exception",
                f"Invoice tax totals {actual_tax}; PO rates applied to invoice "
                f"taxable bases produce {expected_tax}.",
                _field_evidence(invoice.taxable_value)
                + self._tax_evidence(invoice.taxes)
                + [
                    evidence
                    for _, po_line in matched_lines
                    for evidence in _field_evidence(po_line.tax_rate)
                ],
                expected_amount=expected_tax,
                found_amount=actual_tax,
                currency=invoice.currency.value or self.config.currency,
            )

        seller_state = _vendor_state(vendor, invoice.vendor_gstin.value)
        buyer_gstin = invoice.buyer_gstin.value
        buyer_state = (
            buyer_gstin[:2]
            if buyer_gstin and len(buyer_gstin) >= 2 and buyer_gstin[:2].isdigit()
            else None
        )
        if not seller_state or not buyer_state:
            warnings.append(
                "GST jurisdiction was not fully validated because seller or buyer "
                "state code is unavailable."
            )
            return
        if any(tax.rate.value is None for tax in invoice.taxes):
            _add_issue(
                issues,
                "TAX_RATE_UNAVAILABLE",
                "review",
                "One or more invoice tax rates are unavailable for jurisdiction checks.",
                self._tax_evidence(invoice.taxes),
            )

        actual_types = {
            self._tax_type(tax.tax_type.value)
            for tax in invoice.taxes
            if tax.tax_type.value
        }
        expected_types = {"igst"} if seller_state != buyer_state else {"cgst", "sgst"}
        if actual_types != expected_types:
            _add_issue(
                issues,
                "TAX_TYPE_MISMATCH",
                "exception",
                f"Supplier state {seller_state} and buyer state {buyer_state} require "
                f"{', '.join(sorted(expected_types)).upper()}, but invoice has "
                f"{', '.join(sorted(actual_types)).upper() or 'no recognized tax types'}.",
                _field_evidence(invoice.vendor_gstin, invoice.buyer_gstin)
                + self._tax_evidence(invoice.taxes),
            )

        expected_component_amounts: dict[tuple[str, Decimal], Decimal] = {}
        for invoice_line, po_line in matched_lines:
            rate = po_line.tax_rate.value
            taxable_base = invoice_line.line_total.value
            if taxable_base is None:
                taxable_base = (
                    invoice_line.quantity.value * invoice_line.unit_price.value
                )
            if seller_state == buyer_state:
                half_rate = rate / Decimal("2")
                for tax_type in ("cgst", "sgst"):
                    key = (tax_type, half_rate)
                    expected_component_amounts[key] = (
                        expected_component_amounts.get(key, ZERO)
                        + taxable_base * half_rate / HUNDRED
                    )
            else:
                key = ("igst", rate)
                expected_component_amounts[key] = (
                    expected_component_amounts.get(key, ZERO)
                    + taxable_base * rate / HUNDRED
                )
        expected_components = set(expected_component_amounts)
        actual_components = {
            (self._tax_type(tax.tax_type.value), tax.rate.value)
            for tax in invoice.taxes
            if tax.tax_type.value and tax.rate.value is not None
        }
        if actual_components and actual_components != expected_components:
            _add_issue(
                issues,
                "TAX_RATE_MISMATCH",
                "exception",
                f"Invoice tax components {self._format_components(actual_components)} "
                f"do not match PO components "
                f"{self._format_components(expected_components)}.",
                self._tax_evidence(invoice.taxes)
                + [
                    evidence
                    for _, po_line in matched_lines
                    for evidence in _field_evidence(po_line.tax_rate)
                ],
            )
        actual_component_amounts: dict[tuple[str, Decimal], Decimal] = {}
        for tax in invoice.taxes:
            if (
                tax.tax_type.value
                and tax.rate.value is not None
                and tax.amount.value is not None
            ):
                key = (self._tax_type(tax.tax_type.value), tax.rate.value)
                actual_component_amounts[key] = (
                    actual_component_amounts.get(key, ZERO) + tax.amount.value
                )
        for key, expected_amount in expected_component_amounts.items():
            actual_amount = actual_component_amounts.get(key, ZERO)
            if abs(actual_amount - expected_amount) > self.config.amount_tolerance:
                _add_issue(
                    issues,
                    "TAX_COMPONENT_AMOUNT_MISMATCH",
                    "exception",
                    f"{key[0].upper()} at {key[1]}% totals {actual_amount}; "
                    f"PO rates and invoice taxable bases produce {expected_amount}.",
                    self._tax_evidence(invoice.taxes)
                    + [
                        evidence
                        for _, po_line in matched_lines
                        for evidence in _field_evidence(po_line.tax_rate)
                    ],
                    expected_amount=expected_amount,
                    found_amount=actual_amount,
                    currency=invoice.currency.value or self.config.currency,
                )

    @staticmethod
    def _tax_type(value: str | None) -> str:
        if not value:
            return ""
        normalized = value.strip().casefold()
        match = re.match(r"\s*(igst|cgst|sgst|cess)\b", normalized)
        return match.group(1) if match else normalized

    @staticmethod
    def _format_components(components: set[tuple[str, Decimal]]) -> str:
        return ", ".join(
            f"{tax_type.upper()} {rate}%"
            for tax_type, rate in sorted(components, key=lambda item: (item[0], item[1]))
        )

    @staticmethod
    def _tax_evidence(taxes: list[TaxLine]) -> list[SourceEvidence]:
        return _field_evidence(
            *(field for tax in taxes for field in (tax.tax_type, tax.rate, tax.amount))
        )

    def _check_invoice_total(
        self,
        invoice: InvoiceRecord,
        issues: list[MatchIssue],
    ) -> None:
        taxable_value = invoice.taxable_value.value
        if taxable_value is None:
            line_totals = [line.line_total.value for line in invoice.line_items]
            if line_totals and all(value is not None for value in line_totals):
                taxable_value = sum(line_totals, ZERO)
        tax_amounts = [tax.amount.value for tax in invoice.taxes]
        grand_total = invoice.grand_total.value
        if taxable_value is None or not tax_amounts or any(
            amount is None for amount in tax_amounts
        ) or grand_total is None:
            _add_issue(
                issues,
                "INVOICE_TOTAL_UNVERIFIABLE",
                "review",
                "Taxable value, complete tax amounts, and grand total are all "
                "required for invoice arithmetic validation.",
                _field_evidence(
                    invoice.taxable_value,
                    invoice.grand_total,
                )
                + self._tax_evidence(invoice.taxes),
            )
            return
        calculated_total = taxable_value + sum(tax_amounts, ZERO)
        if abs(calculated_total - grand_total) > self.config.amount_tolerance:
            _add_issue(
                issues,
                "INVOICE_TOTAL_MISMATCH",
                "exception",
                f"Taxable value {taxable_value} plus tax {sum(tax_amounts, ZERO)} "
                f"equals {calculated_total}, but invoice grand total is {grand_total}.",
                _field_evidence(
                    invoice.taxable_value,
                    invoice.grand_total,
                )
                + self._tax_evidence(invoice.taxes),
                expected_amount=calculated_total,
                found_amount=grand_total,
                currency=invoice.currency.value or self.config.currency,
            )

    def _match_bank_payments(
        self,
        transactions: list[tuple[Any, BankTransactionRecord]],
        invoices: list[tuple[Any, InvoiceRecord]],
        invoice_results: list[InvoiceMatchResult],
        vendor_by_id: dict[str, VendorRecord],
        vendor_by_gstin: dict[str, VendorRecord],
        vendor_by_name: dict[str, VendorRecord],
        document_status: dict[str, str],
    ) -> list[BankPaymentMatchResult]:
        results_by_invoice = {
            result.invoice_document_id: result for result in invoice_results
        }
        duplicates = {
            result.invoice_document_id
            for result in invoice_results
            if result.duplicate_of_document_id
        }
        primary_invoices = [
            (doc, invoice)
            for doc, invoice in invoices
            if doc.document_id not in duplicates
        ]
        allocated: set[str] = set()
        results: list[BankPaymentMatchResult] = []

        for document, transaction in transactions:
            date_value = transaction.transaction_date.value
            narration = transaction.narration.value
            debit = transaction.debit.value
            evidence = _field_evidence(
                transaction.transaction_date,
                transaction.narration,
                transaction.debit,
            )
            if debit is None or debit <= ZERO:
                results.append(
                    BankPaymentMatchResult(
                        transaction_document_id=document.document_id,
                        transaction_source_file=document.source_file,
                        transaction_date=date_value.isoformat() if date_value else None,
                        narration=narration,
                        debit_amount=debit,
                        status="ignored",
                        warnings=[
                            "Transaction has no positive debit amount and is outside "
                            "accounts-payable payment matching."
                        ],
                        evidence=evidence,
                    )
                )
                continue
            if narration and "bank charges" in narration.casefold():
                results.append(
                    BankPaymentMatchResult(
                        transaction_document_id=document.document_id,
                        transaction_source_file=document.source_file,
                        transaction_date=date_value.isoformat() if date_value else None,
                        narration=narration,
                        debit_amount=debit,
                        status="ignored",
                        warnings=["Bank charges are outside invoice payment matching."],
                        evidence=evidence,
                    )
                )
                continue

            reference_candidates = [
                (invoice_doc, invoice)
                for invoice_doc, invoice in primary_invoices
                if self._invoice_reference_matches(
                    invoice.invoice_number.value, narration
                )
            ]
            candidates = reference_candidates
            if not candidates and narration:
                candidates = [
                    (invoice_doc, invoice)
                    for invoice_doc, invoice in primary_invoices
                    if self._vendor_mentioned(
                        self._resolve_vendor(
                            invoice,
                            vendor_by_id,
                            vendor_by_gstin,
                            vendor_by_name,
                        ),
                        narration,
                    )
                ]

            if not candidates:
                issue = MatchIssue(
                    code="UNMATCHED_PAYMENT",
                    severity="exception",
                    message="Debit transaction could not be linked to an invoice "
                    "using its reference or vendor narration.",
                    evidence=evidence,
                )
                results.append(
                    BankPaymentMatchResult(
                        transaction_document_id=document.document_id,
                        transaction_source_file=document.source_file,
                        transaction_date=date_value.isoformat() if date_value else None,
                        narration=narration,
                        debit_amount=debit,
                        status="exception",
                        issues=[issue],
                        evidence=evidence,
                    )
                )
                continue

            if len(candidates) > 1:
                exact_amount = [
                    (invoice_doc, invoice)
                    for invoice_doc, invoice in candidates
                    if invoice.grand_total.value is not None
                    and abs(invoice.grand_total.value - debit)
                    <= self.config.amount_tolerance
                ]
                if len(exact_amount) == 1:
                    candidates = exact_amount
                else:
                    issue = MatchIssue(
                        code="AMBIGUOUS_PAYMENT_MATCH",
                        severity="review",
                        message="Payment narration matches multiple invoices and "
                        "does not identify a unique invoice.",
                        evidence=evidence,
                    )
                    results.append(
                        BankPaymentMatchResult(
                            transaction_document_id=document.document_id,
                            transaction_source_file=document.source_file,
                            transaction_date=(
                                date_value.isoformat() if date_value else None
                            ),
                            narration=narration,
                            debit_amount=debit,
                            status="review",
                            issues=[issue],
                            evidence=evidence,
                        )
                    )
                    continue

            invoice_doc, invoice = candidates[0]
            match_issues: list[MatchIssue] = []
            expected_amount = invoice.grand_total.value
            variance = (
                debit - expected_amount if expected_amount is not None else None
            )
            if expected_amount is None:
                _add_issue(
                    match_issues,
                    "INVOICE_TOTAL_UNAVAILABLE",
                    "review",
                    "Matched invoice has no usable grand total.",
                    _field_evidence(invoice.grand_total),
                )
            elif abs(variance) > self.config.amount_tolerance:
                _add_issue(
                    match_issues,
                    "PAYMENT_VARIANCE",
                    "exception",
                    f"Bank debit {debit} differs from invoice total {expected_amount} "
                    f"by {variance}.",
                    evidence + _field_evidence(invoice.grand_total),
                    expected_amount=expected_amount,
                    found_amount=debit,
                    currency=invoice.currency.value or self.config.currency,
                )
            if invoice_doc.document_id in allocated:
                _add_issue(
                    match_issues,
                    "DUPLICATE_PAYMENT",
                    "exception",
                    "More than one bank debit was linked to the same invoice.",
                    evidence + _field_evidence(invoice.invoice_number),
                )
            allocated.add(invoice_doc.document_id)

            if document_status.get(document.document_id) != "success":
                _add_issue(
                    match_issues,
                    "BANK_EXTRACTION_NOT_SUCCESSFUL",
                    "review",
                    "Bank transaction extraction was not successful.",
                    evidence,
                )
            linked_result = results_by_invoice.get(invoice_doc.document_id)
            results.append(
                BankPaymentMatchResult(
                    transaction_document_id=document.document_id,
                    transaction_source_file=document.source_file,
                    transaction_date=date_value.isoformat() if date_value else None,
                    narration=narration,
                    debit_amount=debit,
                    matched_invoice_document_id=invoice_doc.document_id,
                    matched_invoice_number=invoice.invoice_number.value,
                    expected_invoice_amount=expected_amount,
                    amount_variance=variance,
                    status=self._bank_status(match_issues),
                    issues=match_issues,
                    warnings=(
                        [
                            "Linked invoice has unresolved matching issues."
                        ]
                        if linked_result and linked_result.status != "matched"
                        else []
                    ),
                    evidence=evidence + _field_evidence(invoice.invoice_number),
                )
            )
        return results

    @staticmethod
    def _invoice_reference_matches(invoice_number: str | None, narration: str | None) -> bool:
        if not invoice_number or not narration:
            return False
        invoice_parts = re.findall(r"[A-Za-z]+|\d+", invoice_number.upper())
        narration_parts = re.findall(r"[A-Za-z]+|\d+", narration.upper())
        if not invoice_parts or not narration_parts:
            return False
        alpha_parts = [part for part in invoice_parts if part.isalpha()]
        numeric_parts = [part for part in invoice_parts if part.isdigit()]
        if not alpha_parts or not numeric_parts:
            return normalize_identifier(invoice_number) in normalize_identifier(narration)
        leading_alpha = alpha_parts[0]
        trailing_number = str(int(numeric_parts[-1]))
        narration_alpha = {part for part in narration_parts if part.isalpha()}
        narration_numbers = {
            str(int(part)) for part in narration_parts if part.isdigit()
        }
        return leading_alpha in narration_alpha and trailing_number in narration_numbers

    @staticmethod
    def _vendor_mentioned(vendor: VendorRecord | None, narration: str) -> bool:
        if not vendor or not vendor.vendor_name.value:
            return False
        ignored = {"private", "pvt", "limited", "ltd", "llp", "company", "co"}
        vendor_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", vendor.vendor_name.value.casefold())
            if term not in ignored and len(term) > 1
        }
        narration_terms = set(re.findall(r"[a-z0-9]+", narration.casefold()))
        shared = vendor_terms & narration_terms
        return len(shared) >= 2 and len(shared) / len(vendor_terms) >= 0.5

    @staticmethod
    def _invoice_status(issues: list[MatchIssue]) -> str:
        if any(issue.severity == "exception" for issue in issues):
            return "exception"
        if any(issue.severity == "review" for issue in issues):
            return "review"
        return "matched"

    @staticmethod
    def _bank_status(issues: list[MatchIssue]) -> str:
        if any(issue.severity == "exception" for issue in issues):
            return "exception"
        if any(issue.severity == "review" for issue in issues):
            return "review"
        return "matched"

    @staticmethod
    def _deduplicate_issues(issues: list[MatchIssue]) -> list[MatchIssue]:
        unique: dict[tuple[str, str], MatchIssue] = {}
        for issue in issues:
            unique.setdefault((issue.code, issue.message), issue)
        return list(unique.values())


def run_matching(
    input_file: Path,
    output_file: Path,
    *,
    config: MatchingConfig | None = None,
) -> MatchingBatch:
    try:
        extraction = ExtractionBatch.model_validate_json(
            input_file.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        logger.error("Could not read validated extraction input %s: %s", input_file, error)
        raise

    matching = MatchingAgent(config=config).match_batch(extraction)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    try:
        temporary_file.write_text(
            matching.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary_file.replace(output_file)
    except OSError:
        temporary_file.unlink(missing_ok=True)
        raise
    return matching
