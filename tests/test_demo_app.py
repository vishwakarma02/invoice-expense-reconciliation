from datetime import date
from decimal import Decimal
import json

from fastapi.testclient import TestClient

from src.anomaly.agent import AnomalyReport
from src.demo import app as demo_app
from src.explanation.agent import ExplanationReport
from src.extraction.schemas import (
    DocumentExtraction,
    ExtractedField,
    ExtractionBatch,
    InvoiceRecord,
    LineItem,
    PurchaseOrderRecord,
    SourceEvidence,
)
from src.hitl.agent import HITLRouter
from src.matching.schemas import (
    InvoiceMatchResult,
    LineMatchResult,
    MatchIssue,
    MatchingBatch,
    MatchingConfig,
)


def extracted_field(value, document_id, source_file):
    return ExtractedField(
        value=value,
        raw_value=str(value) if value is not None else None,
        confidence=1.0 if value is not None else None,
        evidence=[
            SourceEvidence(
                document_id=document_id,
                source_file=source_file,
                quote=str(value) if value is not None else None,
            )
        ],
    )


def test_health_and_home_page():
    client = TestClient(demo_app.app)
    page = client.get("/").text

    assert client.get("/api/health").json()["email_sending_enabled"] is False
    assert "Run reconciliation pipeline" in page
    assert 'id="input-files"' in page
    assert 'id="stage-tabs"' in page
    assert "Expected amount:" in page
    assert "; found amount:" in page
    assert "Validation:" not in page
    assert "Audit request:" not in page
    assert "response.message" not in page
    assert 'id="decision-dialog"' in page
    assert 'id="decision-submit"' in page
    assert "Saving…" in page
    assert 'byId("decision-dialog").close();' in page
    assert "window.location.reload();" in page


def test_upload_runs_pipeline_and_exposes_each_stage(tmp_path, monkeypatch):
    database_path = tmp_path / "review.sqlite3"
    monkeypatch.setattr(demo_app, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(demo_app, "HITL_DATABASE", database_path)
    monkeypatch.setattr(demo_app, "INGESTION_OUTPUT", tmp_path / "ingestion.json")
    monkeypatch.setattr(demo_app, "EXTRACTION_OUTPUT", tmp_path / "extraction.json")
    monkeypatch.setattr(demo_app, "MATCHING_OUTPUT", tmp_path / "matching.json")
    monkeypatch.setattr(demo_app, "ANOMALY_OUTPUT", tmp_path / "anomaly.json")
    monkeypatch.setattr(demo_app, "EXPLANATION_OUTPUT", tmp_path / "explanation.json")

    def fake_pipeline(input_dir, output_dir):
        uploaded_files = [path for path in input_dir.rglob("*") if path.is_file()]
        assert len(uploaded_files) == 1
        assert uploaded_files[0].parent.name == "invoice"
        output_dir.mkdir(parents=True)
        (output_dir / "ingestion_output.json").write_text(
            '{"batch_id":"ingest-1","documents":[]}',
            encoding="utf-8",
        )
        (output_dir / "extraction_output.json").write_text(
            ExtractionBatch(
                extraction_id="extract-1",
                ingestion_batch_id="ingest-1",
            ).model_dump_json(),
            encoding="utf-8",
        )
        (output_dir / "matching_output.json").write_text(
            MatchingBatch(
                matching_id="matching-1",
                extraction_id="extract-1",
                config=MatchingConfig(),
            ).model_dump_json(),
            encoding="utf-8",
        )
        (output_dir / "anomaly_output.json").write_text(
            AnomalyReport(analyzed_invoice_count=0).model_dump_json(),
            encoding="utf-8",
        )
        (output_dir / "explanation_output.json").write_text(
            ExplanationReport(
                matching_id="matching-1",
                explanation_count=0,
            ).model_dump_json(),
            encoding="utf-8",
        )

    monkeypatch.setattr(demo_app, "run_pipeline", fake_pipeline)
    client = TestClient(demo_app.app)
    response = client.post(
        "/api/run",
        files=[("files", ("invoice-123.pdf", b"sample pdf bytes", "application/pdf"))],
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert response.json()["matching_id"] == "matching-1"
    for stage in ("ingestion", "extraction", "matching", "anomaly", "explanation"):
        stage_response = client.get(f"/api/outputs/{stage}?run_id={run_id}")
        assert stage_response.status_code == 200


def test_case_to_draft_api_uses_verified_match_data_and_never_sends(
    tmp_path, monkeypatch
):
    extraction_path = tmp_path / "extraction.json"
    matching_path = tmp_path / "matching.json"
    database_path = tmp_path / "review.sqlite3"
    invoice_id = "invoice-1"
    invoice_file = "invoices/invoice.pdf"
    po_id = "po-1"
    po_file = "master/purchase_orders.csv"

    po = PurchaseOrderRecord(
        po_number=extracted_field("PO-1", po_id, po_file),
        currency=extracted_field("INR", po_id, po_file),
        line_items=[
            LineItem(
                line_number=extracted_field("1", po_id, po_file),
                description=extracted_field("Widget", po_id, po_file),
                item_code=extracted_field("SKU-1", po_id, po_file),
                quantity=extracted_field(Decimal("10"), po_id, po_file),
                unit_price=extracted_field(Decimal("100"), po_id, po_file),
            )
        ],
    )
    invoice = InvoiceRecord(
        vendor_name=extracted_field("Example Vendor", invoice_id, invoice_file),
        invoice_number=extracted_field("INV-1", invoice_id, invoice_file),
        invoice_date=extracted_field(date(2026, 10, 1), invoice_id, invoice_file),
        po_number=extracted_field("PO-1", invoice_id, invoice_file),
        currency=extracted_field("INR", invoice_id, invoice_file),
        line_items=[
            LineItem(
                line_number=extracted_field("1", invoice_id, invoice_file),
                description=extracted_field("Widget", invoice_id, invoice_file),
                item_code=extracted_field("SKU-1", invoice_id, invoice_file),
                quantity=extracted_field(Decimal("10"), invoice_id, invoice_file),
                unit_price=extracted_field(Decimal("110"), invoice_id, invoice_file),
            )
        ],
        grand_total=extracted_field(Decimal("1298"), invoice_id, invoice_file),
    )
    extraction = ExtractionBatch(
        extraction_id="extract-1",
        ingestion_batch_id="ingest-1",
        documents=[
            DocumentExtraction(
                document_id=invoice_id,
                source_file=invoice_file,
                source_category="invoice",
                status="success",
                records=[invoice],
            ),
            DocumentExtraction(
                document_id=po_id,
                source_file=po_file,
                source_category="master",
                status="success",
                records=[po],
            ),
        ],
    )
    issue = MatchIssue(
        code="PRICE_VARIANCE",
        severity="exception",
        message="Invoice unit price 110 differs from PO price 100.",
        expected_amount=Decimal("100"),
        found_amount=Decimal("110"),
        currency="INR",
        evidence=[
            SourceEvidence(
                document_id=invoice_id,
                source_file=invoice_file,
                quote="Invoice unit price: 110",
            ),
            SourceEvidence(
                document_id=po_id,
                source_file=po_file,
                quote="PO unit price: 100",
            ),
        ],
    )
    matching = MatchingBatch(
        matching_id="matching-1",
        extraction_id="extract-1",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id=invoice_id,
                invoice_source_file=invoice_file,
                invoice_number="INV-1",
                purchase_order_number="PO-1",
                status="exception",
                lines=[
                    LineMatchResult(
                        invoice_line_number="1",
                        purchase_order_line_number="1",
                        item_description="Widget",
                        item_code="SKU-1",
                        invoiced_quantity=Decimal("10"),
                        ordered_quantity=Decimal("10"),
                        invoiced_unit_price=Decimal("110"),
                        ordered_unit_price=Decimal("100"),
                        issues=[issue],
                    )
                ],
                issues=[issue],
            )
        ],
    )
    explanation = ExplanationReport(
        matching_id="matching-1",
        explanation_count=1,
        explanations=[
            {
                "source_type": "invoice",
                "source_document_id": invoice_id,
                "source_file": invoice_file,
                "invoice_number": "INV-1",
                "issue_code": "PRICE_VARIANCE",
                "explanation": "Invoice price differs from the purchase order.",
                "recommended_action": "Request clarification.",
                "expected_amount": "100",
                "found_amount": "110",
                "currency": "INR",
            }
        ],
    )
    extraction_path.write_text(extraction.model_dump_json(), encoding="utf-8")
    matching_path.write_text(matching.model_dump_json(), encoding="utf-8")
    cases = HITLRouter(database_path).route(
        matching,
        explanation,
        AnomalyReport(analyzed_invoice_count=1),
    )

    monkeypatch.setattr(demo_app, "EXTRACTION_OUTPUT", extraction_path)
    monkeypatch.setattr(demo_app, "MATCHING_OUTPUT", matching_path)
    monkeypatch.setattr(demo_app, "HITL_DATABASE", database_path)

    client = TestClient(demo_app.app)
    response = client.post(
        f"/api/cases/{cases[0].case_id}/draft",
        json={"reviewed_source_evidence": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DRAFT_REVIEW_REQUIRED"
    assert body["draft"]["subject"].startswith("DRAFT - ")
    assert "PO-1" in body["draft"]["subject"]
    assert "INV-1" in body["draft"]["subject"]
    assert "NOT SENT" in body["draft"]["body"]
    assert body["audit_event"]["details"]["allow_send"] is False


def test_case_decision_endpoint_records_review(tmp_path, monkeypatch):
    database_path = tmp_path / "review.sqlite3"
    monkeypatch.setattr(demo_app, "HITL_DATABASE", database_path)
    client = TestClient(demo_app.app)

    response = client.post(
        "/api/cases/missing/decision",
        json={"decision": "approve", "reviewer": "Finance Clerk"},
    )

    assert response.status_code == 404
    assert json.loads(response.text)["detail"]


def test_case_decision_endpoint_persists_decision_and_returns_updated_case(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "review.sqlite3"
    monkeypatch.setattr(demo_app, "HITL_DATABASE", database_path)
    matching = MatchingBatch(
        matching_id="run-review",
        extraction_id="extract-review",
        config=MatchingConfig(),
        invoices=[
            InvoiceMatchResult(
                invoice_document_id="invoice-review",
                invoice_source_file="invoices/review.pdf",
                invoice_number="INV-REVIEW",
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
    case = HITLRouter(database_path).route(
        matching,
        ExplanationReport(matching_id="run-review", explanation_count=0),
        AnomalyReport(analyzed_invoice_count=1),
    )[0]
    client = TestClient(demo_app.app)

    response = client.post(
        f"/api/cases/{case.case_id}/decision",
        json={
            "decision": "request_info",
            "reviewer": "Finance Reviewer",
            "note": "Please verify the price.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "request_info"
    assert response.json()["reviewer"] == "Finance Reviewer"
    assert response.json()["decision_note"] == "Please verify the price."
    listed_cases = client.get("/api/cases", params={"run_id": "run-review"}).json()
    assert listed_cases[0]["status"] == "request_info"
