from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from src.anomaly.agent import AnomalyReport
from src.correspondence.agent import (
    CommunicationPreferences,
    DiscrepancyType,
    DisputeRequest,
    Evidence,
    EvidenceSourceType,
    Invoice as DisputeInvoice,
    InvoiceLine as DisputeInvoiceLine,
    PurchaseOrder as DisputePurchaseOrder,
    PurchaseOrderLine as DisputePurchaseOrderLine,
    Tone,
    Vendor as DisputeVendor,
    VendorDisputeInput,
    VendorDisputeResponse,
    execute_vendor_dispute_agent,
)
from src.extraction.schemas import (
    DocumentExtraction,
    ExtractionBatch,
    InvoiceRecord,
    PurchaseOrderRecord,
)
from src.explanation.agent import ExplanationReport
from src.hitl.agent import DEFAULT_DATABASE, HITLRouter, HumanDecision, ReviewCase
from src.matching.agent import normalize_currency, normalize_identifier
from src.matching.schemas import MatchingBatch
from run_pipeline import run_pipeline
from src.ingestion.agent import (
    SUPPORTED_CSV,
    SUPPORTED_IMAGES,
    SUPPORTED_PDF,
    infer_category,
)


logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
INGESTION_OUTPUT = OUTPUT_DIR / "ingestion_output.json"
EXTRACTION_OUTPUT = OUTPUT_DIR / "extraction_output.json"
MATCHING_OUTPUT = OUTPUT_DIR / "matching_output.json"
ANOMALY_OUTPUT = OUTPUT_DIR / "anomaly_output.json"
EXPLANATION_OUTPUT = OUTPUT_DIR / "explanation_output.json"
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
HITL_DATABASE = DEFAULT_DATABASE
UPLOAD_ROOT = OUTPUT_DIR / "demo_uploads"
MAX_UPLOAD_FILES = 50
MAX_UPLOAD_FILE_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 100 * 1024 * 1024


class DraftFromCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_source_evidence: Literal[True]
    sender_name: str = "Finance Team"
    sender_company: str = "Accounts Payable"
    tone: Tone = Tone.PROFESSIONAL_NEUTRAL


class PipelineRunResponse(BaseModel):
    status: Literal["completed"] = "completed"
    run_id: str
    extraction_id: str
    matching_id: str
    analyzed_invoices: int
    anomaly_findings: int
    explanations: int
    explanations_using_llm_rag: int
    review_cases: int
    pending_review_cases: int


app = FastAPI(
    title="Invoice Reconciliation Demo",
    version="1.0.0",
    description=(
        "Local demonstration of invoice ingestion, extraction, deterministic "
        "matching, anomaly detection, LLM/RAG explanations, human review, and "
        "draft-only vendor correspondence."
    ),
)


def _read_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline output does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Pipeline output must be a JSON object or array: {path}")
    return data


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_home() -> HTMLResponse:
    page = PROJECT_ROOT / "src" / "demo" / "static" / "index.html"
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str | bool | int]:
    llm_enabled = os.getenv("EXPLANATION_LLM_ENABLED", "false").strip().casefold()
    return {
        "status": "healthy",
        "service": "invoice-reconciliation-demo",
        "explanation_llm_rag_enabled": llm_enabled in {"1", "true", "yes", "on"},
        "policy_source_count": len(list(POLICY_DIR.glob("*.md"))),
        "email_sending_enabled": False,
        "payment_approval_enabled": False,
    }


@app.post("/api/run", response_model=PipelineRunResponse)
def run_demo_pipeline(files: list[UploadFile] = File(...)) -> PipelineRunResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one input file.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload no more than {MAX_UPLOAD_FILES} files per run.",
        )

    run_id = str(uuid.uuid4())
    run_dir = UPLOAD_ROOT / run_id
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=False)
    supported_extensions = SUPPORTED_PDF | SUPPORTED_IMAGES | SUPPORTED_CSV
    total_bytes = 0
    for index, upload in enumerate(files, start=1):
        filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or filename in {".", ".."}:
            raise HTTPException(status_code=400, detail="Every upload needs a filename.")
        if Path(filename).suffix.lower() not in supported_extensions:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {filename}",
            )
        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", filename).strip(" .")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid upload filename.")
        category = infer_category(Path(filename))
        category_dir = input_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        target = category_dir / safe_name
        if target.exists():
            target = category_dir / f"{index:02d}_{safe_name}"
        file_bytes = 0
        with target.open("wb") as saved_file:
            while chunk := upload.file.read(1024 * 1024):
                file_bytes += len(chunk)
                total_bytes += len(chunk)
                if file_bytes > MAX_UPLOAD_FILE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{filename} exceeds the 25 MB per-file limit.",
                    )
                if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="The combined upload exceeds the 100 MB limit.",
                    )
                saved_file.write(chunk)

    try:
        run_pipeline(input_dir, output_dir)
    except Exception as error:
        logger.exception("Uploaded-file reconciliation run failed: %s", run_id)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed for run {run_id}: {error}",
        ) from error

    global INGESTION_OUTPUT, EXTRACTION_OUTPUT, MATCHING_OUTPUT
    global ANOMALY_OUTPUT, EXPLANATION_OUTPUT
    INGESTION_OUTPUT = output_dir / "ingestion_output.json"
    EXTRACTION_OUTPUT = output_dir / "extraction_output.json"
    MATCHING_OUTPUT = output_dir / "matching_output.json"
    ANOMALY_OUTPUT = output_dir / "anomaly_output.json"
    EXPLANATION_OUTPUT = output_dir / "explanation_output.json"
    matching = MatchingBatch.model_validate(_read_json(MATCHING_OUTPUT))
    anomaly_data = AnomalyReport.model_validate(_read_json(ANOMALY_OUTPUT))
    explanation_data = ExplanationReport.model_validate(_read_json(EXPLANATION_OUTPUT))
    cases = HITLRouter(HITL_DATABASE).list_cases(matching.matching_id)
    return PipelineRunResponse(
        run_id=run_id,
        extraction_id=matching.extraction_id,
        matching_id=matching.matching_id,
        analyzed_invoices=anomaly_data.analyzed_invoice_count,
        anomaly_findings=len(anomaly_data.findings),
        explanations=explanation_data.explanation_count,
        explanations_using_llm_rag=explanation_data.llm_rag_explanation_count,
        review_cases=len(cases),
        pending_review_cases=sum(case.status == "pending_review" for case in cases),
    )


@app.get("/api/cases", response_model=list[ReviewCase])
def list_review_cases(run_id: str | None = None) -> list[ReviewCase]:
    return HITLRouter(HITL_DATABASE).list_cases(run_id)


@app.post("/api/cases/{case_id}/decision", response_model=ReviewCase)
def record_human_decision(
    case_id: str,
    decision: HumanDecision,
) -> ReviewCase:
    try:
        return HITLRouter(HITL_DATABASE).decide(case_id, decision)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _invoice_and_po(
    case: ReviewCase,
) -> tuple[
    MatchingBatch,
    DocumentExtraction,
    InvoiceRecord,
    DocumentExtraction,
    PurchaseOrderRecord,
]:
    matching = MatchingBatch.model_validate(_read_json(MATCHING_OUTPUT))
    extraction = ExtractionBatch.model_validate(_read_json(EXTRACTION_OUTPUT))

    if case.case_type == "invoice":
        invoice_match = next(
            (
                item
                for item in matching.invoices
                if item.invoice_document_id == case.source_document_id
            ),
            None,
        )
    else:
        payment = next(
            (
                item
                for item in matching.bank_payments
                if item.transaction_document_id == case.source_document_id
            ),
            None,
        )
        invoice_match = next(
            (
                item
                for item in matching.invoices
                if payment
                and item.invoice_document_id == payment.matched_invoice_document_id
            ),
            None,
        )

    if invoice_match is None or not invoice_match.purchase_order_number:
        raise HTTPException(
            status_code=422,
            detail="This case is not linked to both an invoice and a purchase order.",
        )

    invoice_document = next(
        (
            document
            for document in extraction.documents
            if document.document_id == invoice_match.invoice_document_id
        ),
        None,
    )
    invoice_record = next(
        (
            record
            for record in invoice_document.records
            if isinstance(record, InvoiceRecord)
        ),
        None,
    ) if invoice_document else None
    po_match = next(
        (
            (document, record)
            for document in extraction.documents
            for record in document.records
            if isinstance(record, PurchaseOrderRecord)
            and normalize_identifier(record.po_number.value)
            == normalize_identifier(invoice_match.purchase_order_number)
        ),
        None,
    )
    if invoice_document is None or invoice_record is None or po_match is None:
        raise HTTPException(
            status_code=422,
            detail="Could not find extracted invoice and PO records for this case.",
        )
    po_document, po_record = po_match
    if not invoice_record.invoice_number.value or not po_record.po_number.value:
        raise HTTPException(
            status_code=422,
            detail="Invoice or purchase-order number is missing from extracted records.",
        )
    return matching, invoice_document, invoice_record, po_document, po_record


def _build_correspondence_request(
    case: ReviewCase,
    request: DraftFromCaseRequest,
) -> VendorDisputeInput:
    (
        matching,
        invoice_document,
        invoice,
        po_document,
        purchase_order,
    ) = _invoice_and_po(case)
    invoice_match = next(
        item
        for item in matching.invoices
        if item.invoice_document_id == invoice_document.document_id
    )

    po_lines: list[DisputePurchaseOrderLine] = []
    po_line_numbers: set[str] = set()
    for index, line in enumerate(purchase_order.line_items, start=1):
        line_number = line.line_number.value or str(index)
        if (
            line.description.value
            and line.quantity.value is not None
            and line.unit_price.value is not None
        ):
            po_lines.append(
                DisputePurchaseOrderLine(
                    line_number=line_number,
                    sku=line.item_code.value or None,
                    description=line.description.value,
                    ordered_quantity=line.quantity.value,
                    unit_price=line.unit_price.value,
                )
            )
            po_line_numbers.add(line_number)
    if not po_lines:
        raise HTTPException(
            status_code=422,
            detail="The PO has no complete lines for a validated draft.",
        )

    matching_lines = {
        line.invoice_line_number: line
        for line in invoice_match.lines
        if line.invoice_line_number
    }
    invoice_lines: list[DisputeInvoiceLine] = []
    for index, line in enumerate(invoice.line_items, start=1):
        match = matching_lines.get(line.line_number.value or str(index))
        po_line_number = match.purchase_order_line_number if match else None
        if (
            po_line_number in po_line_numbers
            and line.quantity.value is not None
            and line.unit_price.value is not None
        ):
            invoice_lines.append(
                DisputeInvoiceLine(
                    po_line_number=po_line_number,
                    sku=line.item_code.value or None,
                    description=line.description.value or None,
                    invoiced_quantity=line.quantity.value,
                    unit_price=line.unit_price.value,
                )
            )
    if not invoice_lines:
        raise HTTPException(
            status_code=422,
            detail="No invoice lines have complete quantities, prices, and PO links.",
        )

    issue = next(
        (
            item
            for explanation in case.explanations
            for item in invoice_match.issues
            if item.code == explanation["issue_code"]
        ),
        None,
    )
    code = issue.code if issue else (case.issue_codes[0] if case.issue_codes else "")
    discrepancy_type = {
        "PRICE_VARIANCE": DiscrepancyType.PRICE_MISMATCH,
        "QUANTITY_EXCEEDS_ORDERED": DiscrepancyType.QUANTITY_MISMATCH,
        "QUANTITY_EXCEEDS_RECEIVED": DiscrepancyType.QUANTITY_MISMATCH,
        "TAX_AMOUNT_MISMATCH": DiscrepancyType.TAX_MISMATCH,
        "TAX_COMPONENT_AMOUNT_MISMATCH": DiscrepancyType.TAX_MISMATCH,
        "TAX_RATE_MISMATCH": DiscrepancyType.TAX_MISMATCH,
        "TAX_TYPE_MISMATCH": DiscrepancyType.TAX_MISMATCH,
    }.get(code, DiscrepancyType.OTHER)
    expected = issue.expected_amount if issue else None
    actual = issue.found_amount if issue else None
    if code.startswith("QUANTITY_") and invoice_match.lines:
        line_result = next(
            (
                item
                for item in invoice_match.lines
                if any(line_issue.code == code for line_issue in item.issues)
            ),
            None,
        )
        if line_result:
            if code == "QUANTITY_EXCEEDS_RECEIVED":
                expected = line_result.received_quantity
            else:
                expected = line_result.ordered_quantity
            actual = line_result.invoiced_quantity

    invoice_number = invoice.invoice_number.value
    po_number = purchase_order.po_number.value
    currency = (
        normalize_currency(invoice.currency.value)
        or normalize_currency(purchase_order.currency.value)
        or "INR"
    )
    evidence = [
        Evidence(
            source_type=EvidenceSourceType.PURCHASE_ORDER,
            source_id=po_number,
            verified=True,
            source_uri=po_document.source_file,
            evidence_note="Source record was included in deterministic matching.",
        ),
        Evidence(
            source_type=EvidenceSourceType.INVOICE,
            source_id=invoice_number,
            verified=True,
            source_uri=invoice_document.source_file,
            evidence_note="Source record was included in deterministic matching.",
        ),
    ]

    for duplicate in matching.invoices:
        if (
            duplicate.invoice_document_id == invoice_match.duplicate_of_document_id
            and duplicate.invoice_number
        ):
            evidence.append(
                Evidence(
                    source_type=EvidenceSourceType.INVOICE,
                    source_id=duplicate.invoice_number,
                    verified=True,
                    source_uri=duplicate.invoice_source_file,
                    evidence_note="Referenced duplicate invoice in matching results.",
                )
            )

    return VendorDisputeInput(
        request_id=f"demo-{uuid.uuid4().hex}",
        vendor=DisputeVendor(
            vendor_id=invoice.vendor_id.value or None,
            vendor_name=invoice.vendor_name.value or "Vendor",
        ),
        purchase_order=DisputePurchaseOrder(
            po_number=po_number,
            po_date=purchase_order.po_date.value,
            currency=currency,
            line_items=po_lines,
        ),
        invoice=DisputeInvoice(
            invoice_number=invoice_number,
            invoice_date=invoice.invoice_date.value,
            po_number=po_number,
            currency=currency,
            invoice_value=invoice.grand_total.value,
            line_items=invoice_lines,
        ),
        dispute=DisputeRequest(
            discrepancy_type=discrepancy_type,
            description=(
                issue.message if issue else case.summary
            )[:2000],
            requested_resolution=(
                "Please review the cited records and provide a corrected invoice "
                "or supporting clarification for the discrepancy."
            ),
            expected_value=expected if discrepancy_type != DiscrepancyType.OTHER else None,
            actual_value=actual if discrepancy_type != DiscrepancyType.OTHER else None,
            supplied_difference=(
                actual - expected
                if expected is not None
                and actual is not None
                and discrepancy_type != DiscrepancyType.OTHER
                else None
            ),
            affected_po_line_numbers=[
                line.purchase_order_line_number
                for line in invoice_match.lines
                if line.purchase_order_line_number in po_line_numbers
            ][:100],
        ),
        evidence=evidence,
        communication=CommunicationPreferences(
            tone=request.tone,
            sender_name=request.sender_name,
            sender_company=request.sender_company,
        ),
    )


@app.post(
    "/api/cases/{case_id}/draft",
    response_model=VendorDisputeResponse,
)
def draft_correspondence(
    case_id: str,
    request: DraftFromCaseRequest,
) -> VendorDisputeResponse:
    case = HITLRouter(HITL_DATABASE).get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Review case not found.")
    if case.status not in {"pending_review", "request_info"}:
        raise HTTPException(
            status_code=409,
            detail="Draft creation is available only for cases in human review.",
        )
    try:
        payload = _build_correspondence_request(case, request)
        return execute_vendor_dispute_agent(payload)
    except HTTPException:
        raise
    except (ValueError, OSError) as error:
        logger.exception("Could not prepare correspondence for case %s", case_id)
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/outputs/{name}")
def get_output(
    name: Literal["ingestion", "extraction", "matching", "anomaly", "explanation"],
    run_id: str | None = None,
):
    if run_id is not None:
        try:
            normalized_run_id = str(uuid.UUID(run_id))
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid run ID.") from error
        base_dir = UPLOAD_ROOT / normalized_run_id / "output"
        paths = {
            "ingestion": base_dir / "ingestion_output.json",
            "extraction": base_dir / "extraction_output.json",
            "matching": base_dir / "matching_output.json",
            "anomaly": base_dir / "anomaly_output.json",
            "explanation": base_dir / "explanation_output.json",
        }
        path = paths[name]
    else:
        path = {
            "ingestion": INGESTION_OUTPUT,
            "extraction": EXTRACTION_OUTPUT,
            "matching": MATCHING_OUTPUT,
            "anomaly": ANOMALY_OUTPUT,
            "explanation": EXPLANATION_OUTPUT,
        }[name]
    try:
        return _read_json(path)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
