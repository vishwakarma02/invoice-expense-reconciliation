from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


logger = logging.getLogger(__name__)


class AnomalyPolicy(BaseModel):
    """Thresholds for policy-limit and vendor-history checks."""

    model_config = ConfigDict(extra="forbid")

    max_invoice_amount_by_currency: dict[str, Decimal] = Field(
        default_factory=lambda: {"INR": Decimal("100000")}
    )
    outlier_modified_z_threshold: Decimal = Field(
        default=Decimal("3.5"), gt=Decimal("0")
    )
    min_vendor_history: int = Field(default=3, ge=3)

    @model_validator(mode="after")
    def validate_policy(self) -> AnomalyPolicy:
        normalized_limits: dict[str, Decimal] = {}
        for currency, amount in self.max_invoice_amount_by_currency.items():
            currency_code = _normalize_currency(currency)
            if not currency_code or not amount.is_finite() or amount < 0:
                raise ValueError(
                    "Policy limits must use a currency code and a finite "
                    "non-negative amount."
                )
            normalized_limits[currency_code] = amount
        self.max_invoice_amount_by_currency = normalized_limits
        return self


class AnomalyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str
    anomaly_type: Literal[
        "DUPLICATE_INVOICE",
        "VENDOR_AMOUNT_OUTLIER",
        "ABOVE_POLICY_LIMIT",
    ]
    severity: Literal["HIGH", "MEDIUM"]
    description: str
    observed_amount: Decimal
    currency: str
    duplicate_of: str | None = None


class AnomalyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analyzed_invoice_count: int
    findings: list[AnomalyFinding] = Field(default_factory=list)


class _Invoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    invoice_id: str
    vendor: str
    invoice_number: str
    amount: Decimal
    currency: str
    invoice_date: str | None = None
    line_signature: tuple[tuple[str, Decimal], ...] | None = None

    @cached_property
    def vendor_key(self) -> str:
        return _normalize(self.vendor)

    @cached_property
    def invoice_number_key(self) -> str:
        return _normalize(self.invoice_number)


def _field_value(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid number.") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be a finite non-negative number.")
    return result


def _normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _normalize_currency(value: Any) -> str:
    currency = str(value or "INR").strip().upper()
    if currency in {"₹", "RS", "RS."}:
        return "INR"
    if currency == "$":
        return "USD"
    if currency == "€":
        return "EUR"
    return currency


def _line_signature(
    record: Mapping[str, Any], invoice_id: str
) -> tuple[tuple[str, Decimal], ...] | None:
    line_items = record.get("line_items")
    if not isinstance(line_items, list) or not line_items:
        return None

    signature: list[tuple[str, Decimal]] = []
    for index, line_item in enumerate(line_items, start=1):
        if not isinstance(line_item, Mapping):
            raise ValueError(
                f"Invoice {invoice_id} line item {index} must be an object."
            )
        item_code = next(
            (
                value
                for field_name in ("item_code", "product_id", "sku")
                if isinstance(
                    (value := _field_value(line_item, field_name)), str
                )
                and value.strip()
            ),
            None,
        )
        item_identity = item_code or _field_value(line_item, "description")
        quantity = _field_value(line_item, "quantity")
        if (
            not isinstance(item_identity, str)
            or not item_identity.strip()
            or quantity is None
        ):
            return None
        normalized_identity = _normalize(item_identity)
        if not normalized_identity:
            return None
        signature.append(
            (
                normalized_identity,
                _decimal(quantity, f"Invoice {invoice_id} line item {index} quantity"),
            )
        )
    return tuple(sorted(signature))


def _read_invoices(
    records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[_Invoice]:
    if isinstance(records, Mapping):
        documents = records.get("documents")
        if not isinstance(documents, list):
            raise ValueError("Extraction results must be an array of objects.")
        normalized_records: list[Mapping[str, Any]] = []
        for document_index, document in enumerate(documents, start=1):
            if not isinstance(document, Mapping):
                raise ValueError(
                    f"Extraction document {document_index} must be an object."
                )
            if document.get("status") != "success":
                continue
            document_records = document.get("records")
            if not isinstance(document_records, list):
                raise ValueError(
                    f"Extraction document {document_index} records must be an array."
                )
            for record_index, record in enumerate(document_records, start=1):
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"Extraction document {document_index} record "
                        f"{record_index} must be an object."
                    )
                normalized_records.append(
                    {
                        "status": document["status"],
                        "document_id": document.get("document_id"),
                        "source_file": document.get("source_file"),
                        "record": record,
                    }
                )
        records_to_read = normalized_records
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        records_to_read = records
    else:
        raise ValueError("Extraction results must be an array of objects.")

    invoices: list[_Invoice] = []
    for index, result in enumerate(records_to_read, start=1):
        if not isinstance(result, Mapping):
            raise ValueError(f"Extraction result {index} must be an object.")
        if result.get("status") not in {None, "success"}:
            continue
        record = result.get("record", result)
        if not isinstance(record, Mapping) or (
            record.get("document_type") != "invoice"
            and record.get("record_type") != "invoice"
        ):
            continue

        invoice_number = _field_value(record, "invoice_number")
        vendor = _field_value(record, "vendor_name")
        amount = _field_value(record, "grand_total")
        if invoice_number is None or vendor is None or amount is None:
            raise ValueError(
                f"Invoice record {index} must include invoice_number, "
                "vendor_name, and grand_total."
            )
        if (
            not isinstance(invoice_number, str)
            or not isinstance(vendor, str)
            or not invoice_number.strip()
            or not vendor.strip()
            or not _normalize(invoice_number)
            or not _normalize(vendor)
        ):
            raise ValueError(
                f"Invoice record {index} must have non-empty text values for "
                "invoice_number and vendor_name."
            )

        invoice_id = (
            record.get("invoice_id")
            or result.get("invoice_id")
            or result.get("source_file")
            or result.get("document_id")
            or str(index)
        )
        invoice_date = _field_value(record, "invoice_date")
        normalized_date = None
        if invoice_date:
            try:
                parsed_date = date.fromisoformat(str(invoice_date))
            except ValueError as error:
                raise ValueError(
                    f"Invoice {invoice_id} invoice_date must use YYYY-MM-DD."
                ) from error
            normalized_date = parsed_date.isoformat()
            if normalized_date != str(invoice_date):
                raise ValueError(
                    f"Invoice {invoice_id} invoice_date must use YYYY-MM-DD."
                )
        invoices.append(
            _Invoice(
                invoice_id=str(invoice_id),
                vendor=str(vendor).strip(),
                invoice_number=str(invoice_number).strip(),
                amount=_decimal(amount, f"Invoice {invoice_id} grand_total"),
                currency=_normalize_currency(_field_value(record, "currency")),
                invoice_date=normalized_date,
                line_signature=_line_signature(record, str(invoice_id)),
            )
        )
    invoice_ids = [invoice.invoice_id for invoice in invoices]
    if len(invoice_ids) != len(set(invoice_ids)):
        raise ValueError("Invoice identifiers must be unique within a batch.")
    return invoices


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal("2")


def _read_history(
    history: Mapping[
        str,
        Sequence[Decimal | int | float | str]
        | Mapping[str, Sequence[Decimal | int | float | str]],
    ],
) -> dict[str, dict[str, list[Decimal]]]:
    normalized: dict[str, dict[str, list[Decimal]]] = {}
    for vendor, amounts in history.items():
        key = _normalize(str(vendor))
        if not key:
            raise ValueError("Vendor history contains an empty vendor name.")
        history_by_currency: dict[str, list[Decimal]] = {}
        if isinstance(amounts, Mapping):
            currency_amounts = amounts.items()
        elif isinstance(amounts, Sequence) and not isinstance(amounts, (str, bytes)):
            currency_amounts = (("*", amounts),)
        else:
            raise ValueError(
                f"Vendor history for {vendor} must be an amount array or "
                "a currency-to-amount-array object."
            )

        for currency, values in currency_amounts:
            currency_key = (
                "*" if currency == "*" else _normalize_currency(currency)
            )
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
            ):
                raise ValueError(
                    f"Vendor history for {vendor} and {currency} must be an "
                    "amount array."
                )
            history_by_currency[currency_key] = [
                _decimal(amount, f"Vendor history amount for {vendor}")
                for amount in values
            ]
        normalized[key] = history_by_currency
    return normalized


class AnomalyAgent:
    """Detect duplicate invoices, vendor amount outliers, and policy breaches."""

    def __init__(self, policy: AnomalyPolicy | None = None) -> None:
        self.policy = policy or AnomalyPolicy()

    def analyze(
        self,
        extraction_results: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        vendor_history: Mapping[
            str,
            Sequence[Decimal | int | float | str]
            | Mapping[str, Sequence[Decimal | int | float | str]],
        ]
        | None = None,
    ) -> AnomalyReport:
        """Analyze successful extracted invoices without persisting input data.

        ``vendor_history`` maps vendor names to currency-specific amount
        sequences, or to a legacy amount sequence assumed to match the
        candidate invoice currency. Validation errors are logged and re-raised.
        """
        extraction_count = (
            len(extraction_results["documents"])
            if isinstance(extraction_results, Mapping)
            and isinstance(extraction_results.get("documents"), list)
            else len(extraction_results)
        )
        logger.info(
            "Starting invoice anomaly analysis: extraction_results=%d "
            "vendor_history_vendors=%d",
            extraction_count,
            len(vendor_history or {}),
        )
        try:
            invoices = _read_invoices(extraction_results)
            history = _read_history(vendor_history or {})
        except ValueError as error:
            logger.warning(
                "Invoice anomaly analysis input validation failed (%s).",
                type(error).__name__,
            )
            raise

        findings: list[AnomalyFinding] = []
        seen_by_reference: dict[tuple[str, str, str], _Invoice] = {}
        seen_by_details: dict[
            tuple[str, str, str, Decimal, tuple[tuple[str, Decimal], ...]],
            _Invoice,
        ] = {}

        for invoice in invoices:
            reference_key = (
                invoice.vendor_key,
                invoice.currency,
                invoice.invoice_number_key,
            )
            previous = seen_by_reference.get(reference_key)
            match_reason = "normalized vendor and invoice reference"

            if previous is None and invoice.invoice_date and invoice.line_signature:
                details_key = (
                    invoice.vendor_key,
                    invoice.currency,
                    invoice.invoice_date,
                    invoice.amount,
                    invoice.line_signature,
                )
                previous = seen_by_details.get(details_key)
                match_reason = (
                    "vendor, date, currency, total, and matching item codes "
                    "and quantities"
                )

            if previous is not None:
                logger.warning(
                    "Duplicate invoice detected: invoice_id=%s duplicate_of=%s",
                    invoice.invoice_id,
                    previous.invoice_id,
                )
                findings.append(
                    AnomalyFinding(
                        invoice_id=invoice.invoice_id,
                        anomaly_type="DUPLICATE_INVOICE",
                        severity="HIGH",
                        description=(
                            f"Invoice matches {previous.invoice_id} by "
                            f"{match_reason}."
                        ),
                        observed_amount=invoice.amount,
                        currency=invoice.currency,
                        duplicate_of=previous.invoice_id,
                    )
                )

            if invoice.invoice_number_key:
                seen_by_reference.setdefault(reference_key, invoice)
            if invoice.invoice_date and invoice.line_signature:
                details_key = (
                    invoice.vendor_key,
                    invoice.currency,
                    invoice.invoice_date,
                    invoice.amount,
                    invoice.line_signature,
                )
                seen_by_details.setdefault(details_key, invoice)

            threshold = self.policy.max_invoice_amount_by_currency.get(
                invoice.currency
            )
            if threshold is not None and invoice.amount > threshold:
                logger.warning(
                    "Invoice exceeds the configured policy limit: "
                    "invoice_id=%s currency=%s",
                    invoice.invoice_id,
                    invoice.currency,
                )
                findings.append(
                    AnomalyFinding(
                        invoice_id=invoice.invoice_id,
                        anomaly_type="ABOVE_POLICY_LIMIT",
                        severity="HIGH",
                        description=(
                            f"Invoice amount {invoice.amount} {invoice.currency} "
                            f"exceeds the policy limit of {threshold} "
                            f"{invoice.currency}."
                        ),
                        observed_amount=invoice.amount,
                        currency=invoice.currency,
                    )
                )

            vendor_history = history.get(invoice.vendor_key, {})
            vendor_history_amounts = vendor_history.get(
                invoice.currency, vendor_history.get("*", [])
            )
            if len(vendor_history_amounts) >= self.policy.min_vendor_history:
                historical_median = _median(vendor_history_amounts)
                deviations = [
                    abs(amount - historical_median)
                    for amount in vendor_history_amounts
                ]
                mad = _median(deviations)
                deviation = abs(invoice.amount - historical_median)
                if mad == 0:
                    # A zero MAD means history has no measured spread; any
                    # deviation is surfaced rather than silently ignored.
                    is_outlier = deviation > 0
                    modified_z_score = (
                        Decimal("Infinity") if is_outlier else Decimal("0")
                    )
                else:
                    modified_z_score = Decimal("0.6745") * deviation / mad
                    is_outlier = (
                        modified_z_score
                        > self.policy.outlier_modified_z_threshold
                    )
                if is_outlier:
                    logger.warning(
                        "Vendor-history amount outlier detected: "
                        "invoice_id=%s currency=%s history_count=%d",
                        invoice.invoice_id,
                        invoice.currency,
                        len(vendor_history_amounts),
                    )
                    findings.append(
                        AnomalyFinding(
                            invoice_id=invoice.invoice_id,
                            anomaly_type="VENDOR_AMOUNT_OUTLIER",
                            severity="MEDIUM",
                            description=(
                                f"Invoice amount {invoice.amount} "
                                f"{invoice.currency} is unusual for vendor "
                                f"{invoice.vendor}: historical median is "
                                f"{historical_median} {invoice.currency} "
                                f"(modified z-score "
                                f"{modified_z_score:.2f})."
                            ),
                            observed_amount=invoice.amount,
                            currency=invoice.currency,
                        )
                    )

        report = AnomalyReport(
            analyzed_invoice_count=len(invoices),
            findings=findings,
        )
        findings_by_type: dict[str, int] = {}
        for finding in findings:
            findings_by_type[finding.anomaly_type] = (
                findings_by_type.get(finding.anomaly_type, 0) + 1
            )
        logger.info(
            "Completed invoice anomaly analysis: invoices=%d findings=%d "
            "findings_by_type=%s",
            report.analyzed_invoice_count,
            len(report.findings),
            findings_by_type,
        )
        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect duplicate and anomalous invoices."
    )
    parser.add_argument(
        "--input",
        default="data/output/extraction_output.json",
        help="Extraction Agent batch JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/output/anomaly_output.json",
        help="Path for the anomaly report JSON.",
    )
    parser.add_argument(
        "--vendor-history",
        help=(
            "Optional JSON mapping vendor names to historical invoice "
            "amount arrays."
        ),
    )
    parser.add_argument(
        "--max-invoice-amount",
        type=Decimal,
        default=Decimal("100000"),
        help="INR invoice limit; invoices strictly above it are flagged.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Extraction results file does not exist: %s", input_path)
        raise FileNotFoundError(f"Extraction results not found: {input_path}")

    extraction_results = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(extraction_results, (list, dict)):
        logger.error("Extraction results must be an array or batch object.")
        raise ValueError("Extraction results must be an array or batch object.")

    history: Mapping[str, Sequence[Decimal | int | float | str]] = {}
    if args.vendor_history:
        history_path = Path(args.vendor_history)
        if not history_path.exists():
            logger.error("Vendor history file does not exist: %s", history_path)
            raise FileNotFoundError(f"Vendor history not found: {history_path}")
        history_data = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history_data, dict):
            logger.error("Vendor history must be a JSON object.")
            raise ValueError("Vendor history must be a JSON object.")
        history = history_data

    report = AnomalyAgent(
        AnomalyPolicy(
            max_invoice_amount_by_currency={"INR": args.max_invoice_amount}
        )
    ).analyze(extraction_results, history)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.exception("Could not write anomaly report to %s", output_path)
        raise
    logger.info("Wrote anomaly report to %s", output_path)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
