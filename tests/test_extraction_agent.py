import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.extraction.agent import ExtractionAgent, run_extraction
from src.extraction.schemas import (
    ExtractedField,
    ExtractionBatch,
    InvoiceRecord,
)
from src.ingestion.schemas import (
    ExtractedTable,
    IngestedDocument,
    IngestionBatch,
)


def llm_field(
    value: str | None = None,
    *,
    raw: str | None = None,
    quote: str | None = None,
    confidence: float | None = 0.9,
) -> dict:
    return {
        "value": value,
        "raw_value": raw if raw is not None else value,
        "confidence": confidence,
        "evidence_quote": quote if quote is not None else raw or value,
    }


def valid_invoice_output() -> dict:
    return {
        "vendor_name": llm_field("Example Vendor", quote="Example Vendor"),
        "vendor_id": llm_field(),
        "vendor_gstin": llm_field("27ABCDE1234F1Z5", quote="GSTIN: 27ABCDE1234F1Z5"),
        "buyer_gstin": llm_field(
            "27AAKCS4821M1Z3", quote="Buyer GSTIN: 27AAKCS4821M1Z3"
        ),
        "invoice_number": llm_field("INV-0I", quote="Invoice No: INV-0I"),
        "invoice_date": llm_field(
            "2026-09-18", raw="18-09-2026", quote="Invoice Date: 18-09-2026"
        ),
        "po_number": llm_field("PO-1", quote="PO Ref: PO-1"),
        "currency": llm_field("INR", quote="₹"),
        "line_items": [
            {
                "line_number": llm_field("1", quote="1 Item A 2 50.00 100.00"),
                "description": llm_field("Item A", quote="Item A"),
                "item_code": llm_field(),
                "hsn_sac": llm_field(),
                "quantity": llm_field("2", quote="2"),
                "unit": llm_field("Nos", quote="Nos"),
                "unit_price": llm_field("50.00", quote="50.00"),
                "tax_rate": llm_field(),
                "line_total": llm_field("100.00", quote="100.00"),
            }
        ],
        "subtotal": llm_field("100.00", quote="Taxable Value 100.00"),
        "taxable_value": llm_field("100.00", quote="Taxable Value 100.00"),
        "taxes": [
            {
                "tax_type": llm_field("CGST", quote="CGST 9.00"),
                "rate": llm_field("9", quote="CGST 9.00"),
                "amount": llm_field("9.00", quote="CGST 9.00"),
            }
        ],
        "grand_total": llm_field("109.00", quote="Grand Total 109.00"),
    }


class FakeResponses:
    def __init__(self, output: str):
        self.output = output

    def create(self, **kwargs):
        return SimpleNamespace(output_text=self.output)


class FakeClient:
    def __init__(self, output: str):
        self.responses = FakeResponses(output)


def invoice_document(
    *,
    ocr_confidence: float | None = None,
    text: str | None = None,
) -> IngestedDocument:
    return IngestedDocument(
        document_id="doc-1",
        source_file="invoices/sample.pdf",
        document_type="pdf",
        category="invoice",
        content_hash="a" * 64,
        status="success",
        extracted_text=text
        or (
            "--- Page 1 ---\n"
            "Example Vendor\n"
            "Invoice No: INV-0I\n"
            "Invoice Date: 18-09-2026\n"
            "PO Ref: PO-1\n"
            "GSTIN: 27ABCDE1234F1Z5\n"
            "Buyer GSTIN: 27AAKCS4821M1Z3\n"
            "1 Item A 2 50.00 100.00\n"
            "Taxable Value 100.00\n"
            "CGST 9.00\n"
            "Grand Total 109.00\n₹"
        ),
        ocr_confidence=ocr_confidence,
    )


def test_invoice_extraction_keeps_verified_source_evidence():
    client = FakeClient(json.dumps(valid_invoice_output()))
    result = ExtractionAgent(client=client, model="test-model").extract_document(
        invoice_document(ocr_confidence=0.15)
    )

    assert result.status == "success"
    invoice = result.records[0]
    assert isinstance(invoice, InvoiceRecord)
    assert invoice.invoice_number.value == "INV-0I"
    assert invoice.invoice_number.raw_value == "INV-0I"
    assert invoice.invoice_number.evidence[0].page_number == 1
    assert invoice.invoice_number.evidence[0].source_file == "invoices/sample.pdf"
    assert invoice.invoice_number.confidence == 0.9
    assert invoice.invoice_number.confidence != 0.15
    assert invoice.invoice_date.value.isoformat() == "2026-09-18"
    assert invoice.buyer_gstin.value == "27AAKCS4821M1Z3"
    assert invoice.buyer_gstin.evidence[0].page_number == 1
    assert invoice.grand_total.value == Decimal("109.00")
    assert invoice.vendor_id.value is None


def test_image_receipt_extraction_normalizes_date_and_cites_source():
    receipt_text = "--- Page 1 ---\nHOTEL PANCHAVATI\nBill No: 4471\nDt: 14-10-2026"
    output = {
        "receipt_number": llm_field("4471", quote="Bill No: 4471"),
        "receipt_type": llm_field("expense receipt", quote="HOTEL PANCHAVATI"),
        "po_number": llm_field(),
        "receipt_date": llm_field(
            "2026-10-14", raw="14-10-2026", quote="Dt: 14-10-2026"
        ),
        "vendor_name": llm_field("HOTEL PANCHAVATI", quote="HOTEL PANCHAVATI"),
        "line_items": [],
    }
    document = IngestedDocument(
        document_id="receipt-1",
        source_file="receipts/hotel.png",
        document_type="image",
        category="receipt",
        content_hash="1" * 64,
        status="success",
        extracted_text=receipt_text,
        ocr_confidence=42.0,
    )

    record = ExtractionAgent(
        client=FakeClient(json.dumps(output)), model="test-model"
    ).extract_document(document).records[0]

    assert record.record_type == "receipt"
    assert record.receipt_number.value == "4471"
    assert record.receipt_date.value.isoformat() == "2026-10-14"
    assert record.receipt_date.evidence[0].page_number == 1
    assert record.receipt_date.confidence == 0.9


@pytest.mark.parametrize("ocr_confidence", [None, 12.5])
def test_ocr_confidence_is_not_used_as_field_confidence(ocr_confidence):
    client = FakeClient(json.dumps(valid_invoice_output()))
    invoice = ExtractionAgent(client=client, model="test-model").extract_document(
        invoice_document(ocr_confidence=ocr_confidence)
    ).records[0]

    assert invoice.invoice_number.confidence == 0.9
    assert invoice.invoice_number.evidence[0].page_number == 1


def test_unverified_llm_evidence_is_flagged_and_not_claimed_as_a_quote():
    output = valid_invoice_output()
    output["invoice_number"] = llm_field(
        "INV-FAKE", raw="INV-FAKE", quote="Invoice No: INV-FAKE"
    )
    invoice = ExtractionAgent(
        client=FakeClient(json.dumps(output)), model="test-model"
    ).extract_document(invoice_document()).records[0]

    assert invoice.invoice_number.value == "INV-FAKE"
    assert invoice.invoice_number.evidence[0].quote is None
    assert any("could not be verified" in warning for warning in invoice.invoice_number.warnings)


def test_incorrectly_recognized_tax_is_preserved_and_arithmetic_warning_added():
    output = valid_invoice_output()
    output["taxes"][0]["amount"] = llm_field("8.00", quote="CGST 9.00")
    invoice = ExtractionAgent(
        client=FakeClient(json.dumps(output)), model="test-model"
    ).extract_document(invoice_document()).records[0]

    assert invoice.taxes[0].amount.value == Decimal("8.00")
    assert any("differs from grand total" in warning for warning in invoice.warnings)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("₹1,234.50", Decimal("1234.50")),
        ("Rs. 1,234.50", Decimal("1234.50")),
        ("(1,234.50)", Decimal("-1234.50")),
    ],
)
def test_decimal_parser_normalizes_currency_and_grouping(raw, expected):
    table = ExtractedTable(
        source="bank.csv",
        headers=["txn_date", "narration", "debit", "credit", "balance"],
        rows=[["2026-10-01", "payment", raw, "", "2,000.00"]],
    )
    document = IngestedDocument(
        document_id="bank-1",
        source_file="bank/bank.csv",
        document_type="csv",
        category="bank",
        content_hash="b" * 64,
        status="success",
        tables=[table],
    )

    transaction = ExtractionAgent().extract_document(document).records[0]
    assert transaction.debit.value == expected
    assert transaction.balance.value == Decimal("2000.00")


def test_csv_records_preserve_empty_cells_and_row_evidence():
    table = ExtractedTable(
        source="purchase_orders.csv",
        headers=[
            "po_number",
            "vendor_id",
            "po_date",
            "line_no",
            "item_code",
            "description",
            "hsn_sac",
            "qty",
            "unit",
            "unit_price",
            "gst_rate",
        ],
        rows=[
            [
                "PO-1",
                "V1",
                "2026-10-01",
                "1",
                "ITEM-1",
                "Test item",
                "1234",
                "",
                "Nos",
                "1,200.00",
                "18",
            ]
        ],
    )
    document = IngestedDocument(
        document_id="po-1",
        source_file="master/purchase_orders.csv",
        document_type="csv",
        category="purchase_order",
        content_hash="c" * 64,
        status="success",
        tables=[table],
    )

    result = ExtractionAgent().extract_document(document)
    line = result.records[0].line_items[0]

    assert result.status == "success"
    assert line.quantity.value is None
    assert line.quantity.raw_value is None
    assert line.quantity.warnings
    assert line.unit_price.value == Decimal("1200.00")
    assert line.tax_rate.value == Decimal("18")
    assert line.unit_price.evidence[0].table_row == 1
    assert line.unit_price.evidence[0].column == "unit_price"


def test_csv_dates_parse_day_first_and_warn_when_ambiguous():
    table = ExtractedTable(
        source="expenses.csv",
        headers=["claim_id", "claim_date", "amount_claimed"],
        rows=[["EXP-1", "03/04/2026", "1,000.00"]],
    )
    document = IngestedDocument(
        document_id="expense-1",
        source_file="expenses/claims.csv",
        document_type="csv",
        category="expense",
        content_hash="d" * 64,
        status="success",
        tables=[table],
    )

    record = ExtractionAgent().extract_document(document).records[0]
    assert record.claim_date.value.isoformat() == "2026-04-03"
    assert "day-first" in record.claim_date.warnings[0]
    assert record.amount_claimed.value == Decimal("1000.00")


def test_missing_openai_configuration_fails_unstructured_extraction_explicitly():
    result = ExtractionAgent(model="").extract_document(invoice_document())
    assert result.status == "failed"
    assert "OPENAI_API_KEY" in result.errors[0]


@pytest.mark.parametrize(
    ("response", "error_fragment"),
    [
        ("{", "JSON"),
        ('{"invoice_number":"bad"}', "Field required"),
    ],
)
def test_malformed_or_incomplete_llm_output_is_a_document_failure(
    response, error_fragment
):
    result = ExtractionAgent(
        client=FakeClient(response), model="test-model"
    ).extract_document(invoice_document())

    assert result.status == "failed"
    assert not result.records
    assert error_fragment.casefold() in result.errors[0].casefold()


def test_unknown_category_and_unrecognized_table_are_not_silently_dropped():
    document = IngestedDocument(
        document_id="other-1",
        source_file="misc/unknown.csv",
        document_type="csv",
        category="other",
        content_hash="e" * 64,
        status="success",
        tables=[ExtractedTable(source="unknown.csv", headers=["foo"], rows=[["bar"]])],
    )

    result = ExtractionAgent().extract_document(document)
    assert result.status == "unsupported"
    assert not result.records
    assert result.warnings


def test_extraction_schema_rejects_invalid_confidence_and_financial_values():
    with pytest.raises(ValidationError):
        ExtractedField[str](value="invoice", confidence=1.1)
    with pytest.raises(ValidationError):
        InvoiceRecord(
            grand_total=ExtractedField[Decimal](value="not-a-decimal"),
        )


def test_run_extraction_reads_batch_and_atomically_writes_validated_output(tmp_path):
    ingestion_file = tmp_path / "ingestion.json"
    output_file = tmp_path / "output" / "extraction.json"
    ingestion_file.write_text(
        json.dumps(
            IngestionBatch(
                batch_id="batch-1",
                documents=[
                    IngestedDocument(
                        document_id="vendor-1",
                        source_file="master/vendors.csv",
                        document_type="csv",
                        category="other",
                        content_hash="f" * 64,
                        status="success",
                        tables=[
                            ExtractedTable(
                                source="vendors.csv",
                                headers=["vendor_id", "vendor_name", "gstin"],
                                rows=[["V1", "Example Vendor", "27ABCDE1234F1Z5"]],
                            )
                        ],
                    )
                ],
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )

    batch = run_extraction(ingestion_file, output_file)
    written = ExtractionBatch.model_validate_json(output_file.read_text(encoding="utf-8"))

    assert batch.ingestion_batch_id == "batch-1"
    assert written.documents[0].records[0].record_type == "vendor"
    assert not output_file.with_suffix(".json.tmp").exists()
