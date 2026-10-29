from datetime import date
from decimal import Decimal
import json

from src.extraction.schemas import ExtractionBatch
from src.extraction.schemas import (
    BankTransactionRecord,
    DocumentExtraction,
    ExtractedField,
    ExtractionBatch,
    InvoiceRecord,
    LineItem,
    PurchaseOrderRecord,
    ReceiptRecord,
    SourceEvidence,
    TaxLine,
    VendorRecord,
)
from src.matching.agent import MatchingAgent, normalize_identifier, run_matching
from src.matching.schemas import MatchingConfig


def field(value, *, document_id="doc", source_file="source.csv"):
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


def line_item(
    *,
    description="Item A",
    item_code="ITEM-A",
    quantity="10",
    unit_price="100",
    line_total="1000",
    tax_rate="18",
    line_number="1",
    source="source.csv",
    document_id="doc",
):
    return LineItem(
        line_number=field(line_number, document_id=document_id, source_file=source),
        description=field(description, document_id=document_id, source_file=source),
        item_code=field(item_code, document_id=document_id, source_file=source),
        hsn_sac=field(None, document_id=document_id, source_file=source),
        quantity=field(Decimal(quantity), document_id=document_id, source_file=source),
        unit=field("Nos", document_id=document_id, source_file=source),
        unit_price=field(
            Decimal(unit_price), document_id=document_id, source_file=source
        ),
        tax_rate=field(Decimal(tax_rate), document_id=document_id, source_file=source),
        line_total=field(Decimal(line_total), document_id=document_id, source_file=source),
    )


def invoice_record(
    *,
    number="INV/2026/001",
    vendor_id=None,
    vendor_gstin="27ABCDE1234F1Z5",
    buyer_gstin="27AAKCS4821M1Z3",
    po_number="PO-1",
    quantity="10",
    unit_price="100",
    line_total="1000",
    taxable="1000",
    tax_amount="180",
    grand_total="1180",
    tax_types=("CGST", "SGST"),
    tax_rates=("9", "9"),
    document_id="invoice-1",
    source_file="invoices/invoice.pdf",
    line_description="Item A",
    line_code="ITEM-A",
):
    invoice = InvoiceRecord(
        vendor_name=field("Example Vendor", document_id=document_id, source_file=source_file),
        vendor_id=field(vendor_id, document_id=document_id, source_file=source_file),
        vendor_gstin=field(
            vendor_gstin, document_id=document_id, source_file=source_file
        ),
        buyer_gstin=field(buyer_gstin, document_id=document_id, source_file=source_file),
        invoice_number=field(number, document_id=document_id, source_file=source_file),
        invoice_date=field(
            date(2026, 10, 1), document_id=document_id, source_file=source_file
        ),
        po_number=field(po_number, document_id=document_id, source_file=source_file),
        currency=field("INR", document_id=document_id, source_file=source_file),
        line_items=[
            line_item(
                description=line_description,
                item_code=line_code,
                quantity=quantity,
                unit_price=unit_price,
                line_total=line_total,
                document_id=document_id,
                source=source_file,
            )
        ],
        subtotal=field(
            Decimal(taxable), document_id=document_id, source_file=source_file
        ),
        taxable_value=field(
            Decimal(taxable), document_id=document_id, source_file=source_file
        ),
        taxes=[
            TaxLine(
                tax_type=field(
                    tax_type, document_id=document_id, source_file=source_file
                ),
                rate=field(
                    Decimal(rate), document_id=document_id, source_file=source_file
                ),
                amount=field(
                    Decimal(tax_amount) / len(tax_types),
                    document_id=document_id,
                    source_file=source_file,
                ),
            )
            for tax_type, rate in zip(tax_types, tax_rates)
        ],
        grand_total=field(
            Decimal(grand_total), document_id=document_id, source_file=source_file
        ),
    )
    return invoice


def document(document_id, record, category, source_file=None, status="success"):
    return DocumentExtraction(
        document_id=document_id,
        source_file=source_file or f"{category}/{document_id}.csv",
        source_category=category,
        status=status,
        records=[record],
    )


def base_batch(*, invoice=None, invoice_document_id="invoice-1", extra_documents=None):
    invoice = invoice or invoice_record(document_id=invoice_document_id)
    po = PurchaseOrderRecord(
        po_number=field("PO-1", document_id="po-doc"),
        vendor_id=field("V001", document_id="po-doc"),
        po_date=field(date(2026, 9, 1), document_id="po-doc"),
        line_items=[line_item(document_id="po-doc", source="master/po.csv")],
    )
    receipt = ReceiptRecord(
        receipt_number=field("GRN-1", document_id="receipt-doc"),
        receipt_type=field("GOODS_RECEIPT", document_id="receipt-doc"),
        po_number=field("PO-1", document_id="receipt-doc"),
        receipt_date=field(date(2026, 9, 30), document_id="receipt-doc"),
        line_items=[
            line_item(
                quantity="10",
                document_id="receipt-doc",
                source="master/grn.csv",
            )
        ],
    )
    vendor = VendorRecord(
        vendor_id=field("V001", document_id="vendor-doc"),
        vendor_name=field("Example Vendor", document_id="vendor-doc"),
        gstin=field("27ABCDE1234F1Z5", document_id="vendor-doc"),
        state_code=field("27", document_id="vendor-doc"),
        city=field("Pune", document_id="vendor-doc"),
        payment_terms_days=field(30, document_id="vendor-doc"),
    )
    documents = [
        document(
            invoice_document_id,
            invoice,
            "invoice",
            source_file=f"invoices/{invoice_document_id}.pdf",
        ),
        document("po-doc", po, "purchase_order"),
        document("receipt-doc", receipt, "receipt"),
        document("vendor-doc", vendor, "other"),
    ]
    documents.extend(extra_documents or [])
    return ExtractionBatch(extraction_id="extract-1", ingestion_batch_id="ingest-1", documents=documents)


def issue_codes(result):
    return {issue.code for issue in result.issues}


def test_identifier_normalization_handles_punctuation_and_leading_zeroes():
    assert normalize_identifier("DOS/2026/0417") == "DOS2026417"
    assert normalize_identifier("DOS-2026-417") == "DOS2026417"


def test_clean_invoice_three_way_match_preserves_provenance():
    result = MatchingAgent().match_batch(base_batch()).invoices[0]

    assert result.status == "matched"
    assert result.vendor_id == "V001"
    assert result.purchase_order_document_id == "po-doc"
    assert result.receipt_document_ids == ["receipt-doc"]
    assert result.lines[0].received_quantity == Decimal("10")
    assert result.lines[0].unit_price_variance_percent == Decimal("0")
    assert not result.issues
    assert {item.document_id for item in result.lines[0].evidence} >= {
        "invoice-1",
        "po-doc",
        "receipt-doc",
    }


def test_unit_price_within_tolerance_is_not_flagged():
    invoice = invoice_record(
        unit_price="100.50",
        line_total="1005",
        taxable="1005",
        tax_amount="180.90",
        grand_total="1185.90",
    )
    result = MatchingAgent().match_batch(base_batch(invoice=invoice)).invoices[0]

    assert result.status == "matched"
    assert "PRICE_VARIANCE" not in issue_codes(result)
    assert result.lines[0].unit_price_variance_percent == Decimal("0.50")


def test_price_and_quantity_variances_are_explicit_exceptions():
    invoice = invoice_record(
        unit_price="110",
        quantity="11",
        line_total="1210",
        taxable="1210",
        tax_amount="217.80",
        grand_total="1427.80",
    )
    result = MatchingAgent().match_batch(base_batch(invoice=invoice)).invoices[0]

    assert result.status == "exception"
    assert "PRICE_VARIANCE" in issue_codes(result)
    assert "QUANTITY_EXCEEDS_ORDERED" in issue_codes(result)
    assert "QUANTITY_EXCEEDS_RECEIVED" in issue_codes(result)


def test_invoice_number_duplicate_with_formatting_is_detected():
    first = invoice_record(
        number="DOS/2026/0417",
        vendor_id="V001",
        document_id="invoice-first",
    )
    duplicate = invoice_record(
        number="DOS-2026-417",
        vendor_id="V001",
        document_id="invoice-duplicate",
    )
    batch = base_batch(
        invoice=first,
        invoice_document_id="invoice-first",
        extra_documents=[
            document(
                "invoice-duplicate",
                duplicate,
                "invoice",
                source_file="invoices/duplicate.pdf",
            )
        ],
    )

    results = MatchingAgent().match_batch(batch).invoices
    duplicate_result = next(
        result
        for result in results
        if result.invoice_document_id == "invoice-duplicate"
    )

    assert duplicate_result.status == "exception"
    assert duplicate_result.duplicate_of_document_id == "invoice-first"
    assert "DUPLICATE_INVOICE" in issue_codes(duplicate_result)


def test_missing_po_is_an_exception_and_never_treated_as_matched():
    invoice = invoice_record(po_number=None)
    result = MatchingAgent().match_batch(base_batch(invoice=invoice)).invoices[0]

    assert result.status == "exception"
    assert "MISSING_PO" in issue_codes(result)
    assert result.purchase_order_document_id is None


def test_vendor_and_tax_jurisdiction_mismatches_are_reported():
    invoice = invoice_record(
        vendor_gstin="27ABCDE1234F1Z5",
        buyer_gstin="29AAKCS4821M1Z3",
    )
    result = MatchingAgent().match_batch(base_batch(invoice=invoice)).invoices[0]

    assert "TAX_TYPE_MISMATCH" in issue_codes(result)


def test_tax_component_amounts_are_checked_separately():
    invoice = invoice_record()
    invoice.taxes[0].amount.value = Decimal("10")
    invoice.taxes[1].amount.value = Decimal("8")
    result = MatchingAgent().match_batch(base_batch(invoice=invoice)).invoices[0]

    assert "TAX_COMPONENT_AMOUNT_MISMATCH" in issue_codes(result)


def test_invoice_total_arithmetic_uses_configured_rounding_tolerance():
    invoice = invoice_record(grand_total="1182")
    result = MatchingAgent().match_batch(
        base_batch(invoice=invoice),
    ).invoices[0]

    assert result.status == "exception"
    assert "INVOICE_TOTAL_MISMATCH" in issue_codes(result)

    within_tolerance = invoice_record(grand_total="1180.50")
    result = MatchingAgent(
        MatchingConfig(amount_tolerance=Decimal("1.00"))
    ).match_batch(base_batch(invoice=within_tolerance)).invoices[0]
    assert "INVOICE_TOTAL_MISMATCH" not in issue_codes(result)


def bank_document(
    *,
    document_id,
    narration,
    debit,
    credit=None,
):
    record = BankTransactionRecord(
        transaction_date=field(date(2026, 10, 12), document_id=document_id),
        narration=field(narration, document_id=document_id),
        debit=field(Decimal(debit) if debit is not None else None, document_id=document_id),
        credit=field(Decimal(credit) if credit is not None else None, document_id=document_id),
        balance=field(Decimal("10000"), document_id=document_id),
    )
    return document(document_id, record, "bank")


def test_bank_debit_matches_invoice_reference_and_reports_payment_variance():
    bank = bank_document(
        document_id="bank-1",
        narration="NEFT DR-EXAMPLE VENDOR-INV/2026/001",
        debit="1000",
    )
    batch = base_batch(extra_documents=[bank])

    result = MatchingAgent().match_batch(batch).bank_payments[0]

    assert result.status == "exception"
    assert result.matched_invoice_document_id == "invoice-1"
    assert result.amount_variance == Decimal("-180")
    assert {issue.code for issue in result.issues} == {"PAYMENT_VARIANCE"}


def test_unmatched_debit_and_non_debits_are_distinguished():
    unmatched = bank_document(
        document_id="bank-unmatched",
        narration="UPI DR-UNKNOWN MERCHANT-PAYMENT",
        debit="18500",
    )
    credit = bank_document(
        document_id="bank-credit",
        narration="CUSTOMER PAYMENT",
        debit=None,
        credit="500",
    )
    result = MatchingAgent().match_batch(
        base_batch(extra_documents=[unmatched, credit])
    ).bank_payments

    assert result[0].status == "exception"
    assert result[0].issues[0].code == "UNMATCHED_PAYMENT"
    assert result[1].status == "ignored"


def test_matching_output_schema_records_review_for_low_confidence():
    invoice = invoice_record()
    invoice.invoice_number.confidence = 0.2
    result = MatchingAgent().match_batch(base_batch(invoice=invoice)).invoices[0]

    assert result.status == "review"
    assert "LOW_EXTRACTION_CONFIDENCE" in issue_codes(result)


def test_run_matching_reads_extraction_json_and_writes_validated_output(tmp_path):
    input_file = tmp_path / "extraction.json"
    output_file = tmp_path / "output" / "matching.json"
    extraction = base_batch()
    input_file.write_text(extraction.model_dump_json(), encoding="utf-8")

    result = run_matching(input_file, output_file)
    written = json.loads(output_file.read_text(encoding="utf-8"))
    validated = MatchingAgent().match_batch(
        ExtractionBatch.model_validate_json(input_file.read_text(encoding="utf-8"))
    )

    assert result.extraction_id == "extract-1"
    assert written["extraction_id"] == validated.extraction_id
    assert written["invoices"][0]["status"] == "matched"
