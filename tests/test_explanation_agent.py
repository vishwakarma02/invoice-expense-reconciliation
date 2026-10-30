from decimal import Decimal
import json
from types import SimpleNamespace

from src.anomaly.agent import AnomalyFinding, AnomalyReport
from src.extraction.schemas import SourceEvidence
from src.explanation.agent import (
    ExplanationAgent,
    retrieve_policy_sources,
    run_explanation,
)
from src.matching.schemas import (
    BankPaymentMatchResult,
    InvoiceMatchResult,
    LineMatchResult,
    MatchIssue,
    MatchingBatch,
    MatchingConfig,
)


def issue(
    code="PRICE_VARIANCE",
    *,
    severity="exception",
    expected=Decimal("100"),
    found=Decimal("110"),
):
    return MatchIssue(
        code=code,
        severity=severity,
        message="Invoice unit price 110 differs from PO price 100.",
        expected_amount=expected,
        found_amount=found,
        currency="INR",
        evidence=[
            SourceEvidence(
                document_id="invoice-1",
                source_file="invoices/invoice.pdf",
                page_number=1,
                quote="Unit price: INR 110",
            )
        ],
    )


def matching_batch():
    price_issue = issue()
    return MatchingBatch(
        matching_id="matching-1",
        extraction_id="extraction-1",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id="invoice-1",
                invoice_source_file="invoices/invoice.pdf",
                invoice_number="INV-001",
                status="exception",
                lines=[
                    LineMatchResult(
                        invoice_line_number="1",
                        item_description="Widget",
                        issues=[price_issue],
                    )
                ],
                issues=[price_issue],
            ),
            InvoiceMatchResult(
                invoice_document_id="invoice-review",
                invoice_source_file="invoices/review.pdf",
                status="review",
                issues=[issue(severity="review")],
            ),
        ],
        bank_payments=[
            BankPaymentMatchResult(
                transaction_document_id="bank-1",
                transaction_source_file="bank/statement.csv",
                matched_invoice_number="INV-001",
                status="exception",
                issues=[
                    MatchIssue(
                        code="PAYMENT_VARIANCE",
                        severity="exception",
                        message="Bank debit differs from invoice total.",
                        expected_amount=Decimal("1180"),
                        found_amount=Decimal("1000"),
                        currency="INR",
                        evidence=[
                            SourceEvidence(
                                document_id="bank-1",
                                source_file="bank/statement.csv",
                                table_row=4,
                                column="debit",
                                quote="1000",
                            )
                        ],
                    )
                ],
            )
        ],
    )


def test_explains_each_exception_with_amounts_actions_and_evidence():
    report = ExplanationAgent().explain(matching_batch())

    assert report.matching_id == "matching-1"
    assert report.explanation_count == 2
    invoice_explanation, payment_explanation = report.explanations
    assert invoice_explanation.source_type == "invoice"
    assert invoice_explanation.invoice_number == "INV-001"
    assert "Invoice line 1 (Widget)" in invoice_explanation.explanation
    assert "Expected amount: INR 100; found amount: INR 110." in (
        invoice_explanation.explanation
    )
    assert "approved price change" in invoice_explanation.recommended_action
    assert invoice_explanation.evidence[0].page_number == 1

    assert payment_explanation.source_type == "bank_payment"
    assert "Expected amount: INR 1180; found amount: INR 1000." in (
        payment_explanation.explanation
    )
    assert "partial payments" in payment_explanation.recommended_action


def test_run_explanation_reads_matching_json_and_writes_report(tmp_path):
    input_file = tmp_path / "matching.json"
    output_file = tmp_path / "nested" / "explanations.json"
    input_file.write_text(
        matching_batch().model_dump_json(),
        encoding="utf-8",
    )

    report = run_explanation(input_file, output_file)

    assert report.explanation_count == 2
    saved_report = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_report["matching_id"] == "matching-1"
    assert saved_report["explanations"][0]["found_amount"] == "110"


def test_explains_anomaly_findings_with_reference_and_expected_amount():
    finding = AnomalyFinding(
        invoice_id="invoices/invoice.pdf",
        anomaly_type="ABOVE_POLICY_LIMIT",
        severity="HIGH",
        description="Invoice amount 120000 INR exceeds policy limit of 100000 INR.",
        observed_amount=Decimal("120000"),
        currency="INR",
        expected_amount=Decimal("100000"),
    )
    anomalies = AnomalyReport(analyzed_invoice_count=1, findings=[finding])

    report = ExplanationAgent().explain(matching_batch(), anomalies)

    anomaly_explanation = report.explanations[-1]
    assert report.explanation_count == 3
    assert anomaly_explanation.source_type == "anomaly"
    assert anomaly_explanation.source_document_id == "invoice-1"
    assert anomaly_explanation.invoice_number == "INV-001"
    assert anomaly_explanation.issue_code == "ABOVE_POLICY_LIMIT"
    assert anomaly_explanation.anomaly_type == "ABOVE_POLICY_LIMIT"
    assert anomaly_explanation.severity == "HIGH"
    assert anomaly_explanation.expected_amount == Decimal("100000")
    assert anomaly_explanation.found_amount == Decimal("120000")
    assert "policy limit of 100000 INR" in anomaly_explanation.explanation
    assert "required approval" in anomaly_explanation.recommended_action


def test_llm_receives_anomaly_metadata_and_duplicate_reference(monkeypatch):
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "explanation": "The invoice appears to duplicate an earlier invoice.",
                        "recommended_action": "Compare the referenced invoices.",
                    }
                )
            )

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
    matching = matching_batch()
    finding = AnomalyFinding(
        invoice_id="invoices/invoice.pdf",
        anomaly_type="DUPLICATE_INVOICE",
        severity="HIGH",
        description="Invoice matches invoices/original.pdf by invoice reference.",
        observed_amount=Decimal("110"),
        currency="INR",
        duplicate_of="invoices/original.pdf",
    )
    report = ExplanationAgent(llm_enabled=True, client=FakeClient()).explain(
        matching,
        AnomalyReport(analyzed_invoice_count=1, findings=[finding]),
    )
    anomaly_explanation = report.explanations[-1]
    prompt = json.loads(calls[-1]["input"])

    assert anomaly_explanation.generation_mode == "llm_rag"
    assert anomaly_explanation.duplicate_of == "invoices/original.pdf"
    assert prompt["anomaly_type"] == "DUPLICATE_INVOICE"
    assert prompt["anomaly_severity"] == "HIGH"
    assert prompt["duplicate_of"] == "invoices/original.pdf"


def test_retrieves_relevant_policy_and_uses_llm_without_changing_amounts(
    tmp_path, monkeypatch
):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "price-policy.md").write_text(
        "# Price Variance Policy\nCompare the invoice unit price with the "
        "approved purchase-order price before processing.",
        encoding="utf-8",
    )
    (policy_dir / "payment-policy.md").write_text(
        "# Payment Review\nCompare bank debits to invoice totals.",
        encoding="utf-8",
    )
    citations = retrieve_policy_sources("PRICE_VARIANCE invoice unit price", policy_dir)
    assert citations[0].policy_id == "price-policy"

    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "explanation": "The billed unit price is above the approved PO price.",
                        "recommended_action": "Check the approved PO or request a corrected invoice.",
                    }
                )
            )

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
    report = ExplanationAgent(
        llm_enabled=True,
        client=FakeClient(),
        policy_dir=policy_dir,
    ).explain(matching_batch())

    invoice_explanation = report.explanations[0]
    assert report.llm_rag_explanation_count == 2
    assert invoice_explanation.generation_mode == "llm_rag"
    assert invoice_explanation.policy_sources[0].policy_id == "price-policy"
    assert invoice_explanation.expected_amount == Decimal("100")
    assert invoice_explanation.found_amount == Decimal("110")
    assert "Expected amount: INR 100; found amount: INR 110." in (
        invoice_explanation.explanation
    )
    assert calls
    assert "retrieved_policy_sources" in calls[0]["input"]
