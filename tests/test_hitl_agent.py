from decimal import Decimal

import pytest

from src.anomaly.agent import AnomalyFinding, AnomalyReport
from src.extraction.schemas import (
    DocumentExtraction,
    ExtractionBatch,
    ExtractedField,
    InvoiceRecord,
)
from src.explanation.agent import ExplanationReport
from src.hitl.agent import HITLRouter, HumanDecision
from src.matching.schemas import (
    InvoiceMatchResult,
    MatchIssue,
    MatchingBatch,
    MatchingConfig,
)


def test_router_persists_decision_and_routes_exceptions_to_human(tmp_path):
    matching = MatchingBatch(
        matching_id="run-1",
        extraction_id="extract-1",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id="invoice-1",
                invoice_source_file="invoices/invoice.pdf",
                invoice_number="INV-1",
                status="exception",
                issues=[
                    MatchIssue(
                        code="PRICE_VARIANCE",
                        severity="exception",
                        message="Invoice price differs from the PO.",
                        expected_amount=Decimal("100"),
                        found_amount=Decimal("110"),
                        currency="INR",
                    )
                ],
            ),
            InvoiceMatchResult(
                invoice_document_id="invoice-2",
                invoice_source_file="invoices/clean.pdf",
                invoice_number="INV-2",
                status="matched",
            ),
        ],
    )
    explanations = ExplanationReport(
        matching_id="run-1",
        explanation_count=1,
        explanations=[
            {
                "source_type": "invoice",
                "source_document_id": "invoice-1",
                "source_file": "invoices/invoice.pdf",
                "invoice_number": "INV-1",
                "issue_code": "PRICE_VARIANCE",
                "explanation": "Price differs.",
                "recommended_action": "Review the PO.",
                "expected_amount": "100",
                "found_amount": "110",
                "currency": "INR",
            }
        ],
    )
    anomalies = AnomalyReport(analyzed_invoice_count=2)
    router = HITLRouter(tmp_path / "review.sqlite3")

    cases = router.route(matching, explanations, anomalies)

    assert [case.status for case in cases] == ["pending_review", "auto_cleared"]
    assert cases[0].amount_at_risk == Decimal("10")
    decided = router.decide(
        cases[0].case_id,
        HumanDecision(decision="request_info", reviewer="Finance Clerk", note="Check PO"),
    )
    assert decided.status == "request_info"
    assert router.get_case(cases[0].case_id).reviewer == "Finance Clerk"
    with pytest.raises(ValueError, match="already has a recorded"):
        router.decide(
            cases[0].case_id,
            HumanDecision(decision="approve", reviewer="Another Reviewer"),
        )


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [
        ("approve", "approved"),
        ("reject", "rejected"),
        ("request_info", "request_info"),
    ],
)
def test_router_maps_decision_to_valid_case_status(
    tmp_path,
    decision,
    expected_status,
):
    matching = MatchingBatch(
        matching_id=f"run-{decision}",
        extraction_id="extract-1",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id="invoice-1",
                invoice_source_file="invoices/invoice.pdf",
                invoice_number="INV-1",
                status="exception",
                issues=[
                    MatchIssue(
                        code="PRICE_VARIANCE",
                        severity="exception",
                        message="Invoice price differs from the PO.",
                    )
                ],
            )
        ],
    )
    router = HITLRouter(tmp_path / f"{decision}.sqlite3")
    case = router.route(
        matching,
        ExplanationReport(matching_id=matching.matching_id, explanation_count=0),
        AnomalyReport(analyzed_invoice_count=1),
    )[0]

    decided = router.decide(
        case.case_id,
        HumanDecision(decision=decision, reviewer="Finance Clerk"),
    )

    assert decided.status == expected_status
    assert router.get_case(case.case_id).status == expected_status
    assert router.list_cases(matching.matching_id)[0].status == expected_status


def test_anomaly_findings_route_matched_invoice_to_review(tmp_path):
    matching = MatchingBatch(
        matching_id="run-2",
        extraction_id="extract-2",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id="invoice-1",
                invoice_source_file="invoices/invoice.pdf",
                invoice_number="INV-1",
                status="matched",
            )
        ],
    )
    anomaly = AnomalyReport(
        analyzed_invoice_count=1,
        findings=[
            AnomalyFinding(
                invoice_id="invoices/invoice.pdf",
                anomaly_type="ABOVE_POLICY_LIMIT",
                severity="HIGH",
                description="Invoice exceeds limit.",
                observed_amount=Decimal("120000"),
                currency="INR",
            )
        ],
    )

    case = HITLRouter(tmp_path / "review.sqlite3").route(
        matching,
        ExplanationReport(
            matching_id="run-2",
            explanation_count=1,
            explanations=[
                {
                    "source_type": "anomaly",
                    "source_document_id": "invoice-1",
                    "source_file": "invoices/invoice.pdf",
                    "invoice_number": "INV-1",
                    "issue_code": "ABOVE_POLICY_LIMIT",
                    "anomaly_type": "ABOVE_POLICY_LIMIT",
                    "severity": "HIGH",
                    "explanation": "Invoice exceeds the policy limit.",
                    "recommended_action": "Request required approval.",
                    "expected_amount": "100000",
                    "found_amount": "120000",
                    "currency": "INR",
                }
            ],
        ),
        anomaly,
    )[0]

    assert case.status == "pending_review"
    assert case.amount_at_risk == Decimal("120000")
    assert case.anomalies[0]["anomaly_type"] == "ABOVE_POLICY_LIMIT"
    assert case.explanations[0]["source_type"] == "anomaly"
    assert case.explanations[0]["explanation"] == "Invoice exceeds the policy limit."


def test_router_uses_confidence_and_amount_before_auto_clearing(tmp_path):
    matching = MatchingBatch(
        matching_id="run-3",
        extraction_id="extract-3",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id="low-confidence",
                invoice_source_file="invoices/low-confidence.pdf",
                invoice_number="INV-LOW",
                status="matched",
            ),
            InvoiceMatchResult(
                invoice_document_id="high-value",
                invoice_source_file="invoices/high-value.pdf",
                invoice_number="INV-HIGH",
                status="matched",
            ),
            InvoiceMatchResult(
                invoice_document_id="missing-confidence",
                invoice_source_file="invoices/missing-confidence.pdf",
                invoice_number="INV-UNKNOWN",
                status="matched",
            ),
        ],
    )
    extraction = ExtractionBatch(
        extraction_id="extract-3",
        ingestion_batch_id="ingest-3",
        documents=[
            DocumentExtraction(
                document_id="low-confidence",
                source_file="invoices/low-confidence.pdf",
                source_category="invoice",
                status="success",
                records=[
                    InvoiceRecord(
                        invoice_number=ExtractedField(value="INV-LOW", confidence=0.95),
                        vendor_name=ExtractedField(value="Vendor", confidence=0.95),
                        grand_total=ExtractedField(
                            value=Decimal("1000"),
                            confidence=0.65,
                        ),
                    )
                ],
            ),
            DocumentExtraction(
                document_id="high-value",
                source_file="invoices/high-value.pdf",
                source_category="invoice",
                status="success",
                records=[
                    InvoiceRecord(
                        invoice_number=ExtractedField(
                            value="INV-HIGH",
                            confidence=0.95,
                        ),
                        vendor_name=ExtractedField(value="Vendor", confidence=0.95),
                        grand_total=ExtractedField(
                            value=Decimal("60000"),
                            confidence=0.95,
                        ),
                    )
                ],
            ),
            DocumentExtraction(
                document_id="missing-confidence",
                source_file="invoices/missing-confidence.pdf",
                source_category="invoice",
                status="success",
                records=[
                    InvoiceRecord(
                        invoice_number=ExtractedField(value="INV-UNKNOWN"),
                        grand_total=ExtractedField(value=Decimal("1000"), confidence=0.95),
                    )
                ],
            ),
        ],
    )

    cases = HITLRouter(tmp_path / "review.sqlite3").route(
        matching,
        ExplanationReport(matching_id="run-3", explanation_count=0),
        AnomalyReport(analyzed_invoice_count=3),
        extraction,
    )

    assert [case.status for case in cases] == [
        "pending_review",
        "pending_review",
        "pending_review",
    ]
    assert [case.severity for case in cases] == ["medium", "medium", "medium"]
    assert cases[0].confidence == 0.65
    assert cases[0].routing_reasons == ["low_confidence"]
    assert cases[1].amount_at_risk == Decimal("60000")
    assert cases[1].routing_reasons == ["amount_at_or_above_auto_clear_limit"]
    assert cases[2].confidence is None
    assert cases[2].routing_reasons == ["confidence_unavailable"]
