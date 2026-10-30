from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal
import os

from pydantic import BaseModel, ConfigDict, Field

from src.anomaly.agent import AnomalyReport
from src.explanation.agent import ExplanationReport
from src.extraction.schemas import ExtractionBatch, InvoiceRecord
from src.matching.schemas import MatchingBatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "output" / "hitl_decisions.sqlite3"


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject", "request_info"]
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


class ReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    run_id: str
    case_type: Literal["invoice", "bank_payment"]
    source_document_id: str
    source_file: str
    invoice_number: str | None = None
    status: Literal[
        "auto_cleared",
        "pending_review",
        "approved",
        "rejected",
        "request_info",
    ]
    severity: Literal["high", "medium", "low"] = "low"
    confidence: float | None = None
    amount_at_risk: Decimal | None = None
    routing_reasons: list[str] = Field(default_factory=list)
    summary: str
    issue_codes: list[str] = Field(default_factory=list)
    explanations: list[dict] = Field(default_factory=list)
    anomalies: list[dict] = Field(default_factory=list)
    reviewer: str | None = None
    decision_note: str | None = None
    decided_at: datetime | None = None


def _case_id(run_id: str, kind: str, source_id: str) -> str:
    identity = f"{run_id}:{kind}:{source_id}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _decision_status(decision: str) -> str:
    return {
        "approve": "approved",
        "reject": "rejected",
        "request_info": "request_info",
    }.get(decision, decision)


def _issue_risk(explanations: list[dict]) -> Decimal | None:
    differences = [
        abs(Decimal(str(item["expected_amount"])) - Decimal(str(item["found_amount"])))
        for item in explanations
        if item.get("source_type") != "anomaly"
        if item.get("expected_amount") is not None
        and item.get("found_amount") is not None
    ]
    return max(differences) if differences else None


class HITLRouter:
    """Route exceptions to a durable review queue; never authorize payment."""

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE,
        *,
        minimum_auto_clear_confidence: float | None = None,
        maximum_auto_clear_amount: Decimal | None = None,
    ) -> None:
        if minimum_auto_clear_confidence is None:
            minimum_auto_clear_confidence = float(
                os.getenv("HITL_MIN_AUTO_CLEAR_CONFIDENCE", "0.9")
            )
        if maximum_auto_clear_amount is None:
            maximum_auto_clear_amount = Decimal(
                os.getenv("HITL_MAX_AUTO_CLEAR_AMOUNT", "50000")
            )
        if not 0 <= minimum_auto_clear_confidence <= 1:
            raise ValueError("Auto-clear confidence threshold must be between 0 and 1.")
        if not maximum_auto_clear_amount.is_finite() or maximum_auto_clear_amount < 0:
            raise ValueError("Auto-clear amount threshold must be finite and non-negative.")
        self.database_path = database_path
        self.minimum_auto_clear_confidence = minimum_auto_clear_confidence
        self.maximum_auto_clear_amount = maximum_auto_clear_amount

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS review_cases (
                case_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                decision TEXT,
                reviewer TEXT,
                decision_note TEXT,
                decided_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        return connection

    @staticmethod
    def _create_case(
        *,
        run_id: str,
        kind: Literal["invoice", "bank_payment"],
        source_id: str,
        source_file: str,
        invoice_number: str | None,
        pending: bool,
        severity: Literal["high", "medium", "low"],
        confidence: float | None,
        risk: Decimal | None,
        routing_reasons: list[str],
        summary: str,
        issue_codes: list[str],
        explanations: list[dict],
        anomalies: list[dict],
    ) -> ReviewCase:
        return ReviewCase(
            case_id=_case_id(run_id, kind, source_id),
            run_id=run_id,
            case_type=kind,
            source_document_id=source_id,
            source_file=source_file,
            invoice_number=invoice_number,
            status="pending_review" if pending else "auto_cleared",
            severity=severity,
            confidence=confidence,
            amount_at_risk=risk,
            routing_reasons=routing_reasons,
            summary=summary,
            issue_codes=issue_codes,
            explanations=explanations,
            anomalies=anomalies,
        )

    def route(
        self,
        matching: MatchingBatch,
        explanations: ExplanationReport,
        anomalies: AnomalyReport,
        extraction: ExtractionBatch | None = None,
    ) -> list[ReviewCase]:
        explanation_by_invoice: dict[str, list[dict]] = {}
        explanation_by_payment: dict[str, list[dict]] = {}
        for item in explanations.explanations:
            target = (
                explanation_by_invoice
                if item.source_type in {"invoice", "anomaly"}
                else explanation_by_payment
            )
            target.setdefault(item.source_document_id, []).append(
                item.model_dump(mode="json")
            )

        anomalies_by_file: dict[str, list[dict]] = {}
        for finding in anomalies.findings:
            anomalies_by_file.setdefault(finding.invoice_id, []).append(
                finding.model_dump(mode="json")
            )

        invoice_data: dict[str, tuple[float | None, Decimal | None]] = {}
        if extraction is not None:
            for document in extraction.documents:
                invoice = next(
                    (
                        record
                        for record in document.records
                        if isinstance(record, InvoiceRecord)
                    ),
                    None,
                )
                if invoice is None:
                    continue
                critical_fields = (invoice.invoice_number, invoice.grand_total)
                confidence = (
                    min(field.confidence for field in critical_fields)
                    if all(
                        field.value is not None and field.confidence is not None
                        for field in critical_fields
                    )
                    else None
                )
                invoice_data[document.document_id] = (
                    confidence,
                    invoice.grand_total.value,
                )

        cases: list[ReviewCase] = []
        for invoice in matching.invoices:
            invoice_explanations = explanation_by_invoice.get(
                invoice.invoice_document_id, []
            )
            invoice_anomalies = anomalies_by_file.get(invoice.invoice_source_file, [])
            confidence, invoice_amount = invoice_data.get(
                invoice.invoice_document_id, (None, None)
            )
            risk = _issue_risk(invoice_explanations)
            if risk is None:
                risk = max(
                    (
                        Decimal(str(item["observed_amount"]))
                        for item in invoice_anomalies
                    ),
                    default=None,
                )
            if risk is None:
                risk = invoice_amount
            reasons = []
            if invoice.status == "exception":
                reasons.append("matching_exception")
            elif invoice.status == "review":
                reasons.append("matching_review")
            if invoice_anomalies:
                reasons.append("anomaly_detected")
            if extraction is not None:
                if confidence is None:
                    reasons.append("confidence_unavailable")
                elif confidence < self.minimum_auto_clear_confidence:
                    reasons.append("low_confidence")
                if (
                    risk is not None
                    and risk >= self.maximum_auto_clear_amount
                ):
                    reasons.append("amount_at_or_above_auto_clear_limit")
            severity: Literal["high", "medium", "low"] = (
                "high"
                if any(issue.severity == "exception" for issue in invoice.issues)
                or any(item["severity"] == "HIGH" for item in invoice_anomalies)
                else "medium"
                if invoice.status == "review" or invoice_anomalies or reasons
                else "low"
            )
            pending = (
                invoice.status in {"exception", "review"}
                or bool(invoice_anomalies)
                or bool(reasons)
            )
            issue_codes = [issue.code for issue in invoice.issues]
            summary_parts = [issue.message for issue in invoice.issues]
            summary_parts.extend(
                anomaly["description"] for anomaly in invoice_anomalies
            )
            cases.append(
                self._create_case(
                    run_id=matching.matching_id,
                    kind="invoice",
                    source_id=invoice.invoice_document_id,
                    source_file=invoice.invoice_source_file,
                    invoice_number=invoice.invoice_number,
                    pending=pending,
                    severity=severity,
                    confidence=confidence,
                    risk=risk,
                    routing_reasons=reasons,
                    summary=" ".join(summary_parts)
                    if summary_parts
                    else "Invoice passed deterministic matching and anomaly checks.",
                    issue_codes=issue_codes,
                    explanations=invoice_explanations,
                    anomalies=invoice_anomalies,
                )
            )

        for payment in matching.bank_payments:
            if payment.status == "ignored":
                continue
            payment_explanations = explanation_by_payment.get(
                payment.transaction_document_id, []
            )
            linked_invoice = next(
                (
                    item
                    for item in matching.invoices
                    if item.invoice_document_id == payment.matched_invoice_document_id
                ),
                None,
            )
            confidence, _ = (
                invoice_data.get(linked_invoice.invoice_document_id, (None, None))
                if linked_invoice
                else (None, None)
            )
            payment_risk = _issue_risk(payment_explanations)
            if payment_risk is None:
                payment_risk = payment.expected_invoice_amount
            if payment_risk is None:
                payment_risk = payment.debit_amount
            reasons = []
            if payment.status == "exception":
                reasons.append("matching_exception")
            elif payment.status == "review":
                reasons.append("matching_review")
            if extraction is not None:
                if confidence is None:
                    reasons.append("confidence_unavailable")
                elif confidence < self.minimum_auto_clear_confidence:
                    reasons.append("low_confidence")
                if (
                    payment_risk is not None
                    and payment_risk >= self.maximum_auto_clear_amount
                ):
                    reasons.append("amount_at_or_above_auto_clear_limit")
            severity: Literal["high", "medium", "low"] = (
                "high"
                if any(issue.severity == "exception" for issue in payment.issues)
                else "medium"
                if payment.status == "review" or reasons
                else "low"
            )
            pending = (
                payment.status in {"exception", "review"}
                or bool(reasons)
            )
            cases.append(
                self._create_case(
                    run_id=matching.matching_id,
                    kind="bank_payment",
                    source_id=payment.transaction_document_id,
                    source_file=payment.transaction_source_file,
                    invoice_number=payment.matched_invoice_number,
                    pending=pending,
                    severity=severity,
                    confidence=confidence,
                    risk=payment_risk,
                    routing_reasons=reasons,
                    summary=" ".join(issue.message for issue in payment.issues)
                    or (
                        f"Bank payment matched invoice "
                        f"{payment.matched_invoice_number}."
                        if payment.matched_invoice_number
                        else "Bank transaction is not associated with an invoice."
                    ),
                    issue_codes=[issue.code for issue in payment.issues],
                    explanations=payment_explanations,
                    anomalies=[],
                )
            )

        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                for case in cases:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO review_cases
                        (case_id, run_id, payload_json, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            case.case_id,
                            case.run_id,
                            case.model_dump_json(),
                            now,
                        ),
                    )
        return self.list_cases(matching.matching_id)

    def list_cases(self, run_id: str | None = None) -> list[ReviewCase]:
        with closing(self._connect()) as connection:
            if run_id is None:
                row = connection.execute(
                    "SELECT run_id FROM review_cases ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    return []
                run_id = row["run_id"]
            rows = connection.execute(
                """
                SELECT payload_json, decision, reviewer, decision_note, decided_at
                FROM review_cases WHERE run_id = ?
                ORDER BY CASE WHEN decision IS NULL THEN 0 ELSE 1 END, created_at
                """,
                (run_id,),
            ).fetchall()
        cases: list[ReviewCase] = []
        for row in rows:
            data = json.loads(row["payload_json"])
            if row["decision"]:
                data["status"] = _decision_status(row["decision"])
                data["reviewer"] = row["reviewer"]
                data["decision_note"] = row["decision_note"]
                data["decided_at"] = row["decided_at"]
            cases.append(ReviewCase.model_validate(data))
        return cases

    def get_case(self, case_id: str) -> ReviewCase | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT payload_json, decision, reviewer, decision_note, decided_at
                FROM review_cases WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["payload_json"])
        if row["decision"]:
            data["status"] = _decision_status(row["decision"])
            data["reviewer"] = row["reviewer"]
            data["decision_note"] = row["decision_note"]
            data["decided_at"] = row["decided_at"]
        return ReviewCase.model_validate(data)

    def decide(self, case_id: str, decision: HumanDecision) -> ReviewCase:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE review_cases
                    SET decision = ?, reviewer = ?, decision_note = ?, decided_at = ?
                    WHERE case_id = ? AND decision IS NULL
                    """,
                    (
                        decision.decision,
                        decision.reviewer.strip(),
                        decision.note.strip(),
                        now,
                        case_id,
                    ),
                )
                if cursor.rowcount != 1:
                    exists = connection.execute(
                        "SELECT 1 FROM review_cases WHERE case_id = ?", (case_id,)
                    ).fetchone()
                    if exists is None:
                        raise KeyError(f"Review case not found: {case_id}")
                    raise ValueError(
                        "This case already has a recorded human decision."
                    )
        case = self.get_case(case_id)
        if case is None:
            raise KeyError(f"Review case not found: {case_id}")
        return case
