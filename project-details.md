# Project Handoff Document: Invoice & Expense Reconciliation Agentic AI System

**Document purpose:** Transfer project context to another AI development agent so it can continue implementation without rediscovering requirements, changing established architectural decisions, or breaking working code.

**Project status:** Ingestion, Extraction, and Deterministic Matching Agents are implemented. `run_pipeline.py` serializes and runs all three agents in order. Next milestone: Anomaly Agent.

**Primary language:** Python  
**Development environment:** Windows, PowerShell, Python virtual environment  
**Project directory:** `C:\Agentic-AI-Training\Project\invoice-expense-reconciliation`  
**Currency:** INR  
**LLM integration:** OpenAI API permitted  
**Core principle:** Use LLMs for unstructured document interpretation and explanations; use deterministic Python for financial calculations, matching, and approval rules.

---

# 1. Project Overview

## 1.1 What are we building?

We are building an agentic AI system for automated invoice and expense reconciliation.

The system processes business documents, extracts structured financial information, compares invoices against purchase orders and receipts, identifies discrepancies and suspicious transactions, explains exceptions, and routes cases for automated clearance or human review.

The system must support multiple document formats and realistic document-quality issues, including:

- Text-based PDF invoices.
- Scanned PDF invoices.
- Image-based documents such as JPG and PNG.
- CSV files containing financial or transaction data.
- Inconsistent date and number formats.
- Missing information.
- OCR errors and poorly formatted tables.
- Duplicate invoices with slightly different invoice-number formatting.
- Quantity, unit-price, tax, and total mismatches.
- Missing purchase orders or receipts.
- Bank payment variances and unmatched transactions.
- Expense claims that do not satisfy expected rules.

The objective is not simply to extract text or match amounts. It is to produce an **auditable reconciliation workflow** in which every decision can be traced to source evidence and deterministic rules.

## 1.2 Target users

The primary users are finance and accounts-payable personnel who need to:

- Process supplier invoices faster.
- Reduce manual reconciliation effort.
- Identify payment risks before invoices are cleared.
- Understand why a transaction was flagged.
- Review exceptions with supporting evidence.
- Maintain an auditable record of automated and human decisions.

## 1.3 Expected business outcome

The system should produce structured reconciliation results with clear statuses, reasons, evidence, confidence indicators, and recommended actions.

It should distinguish between:

1. Documents successfully processed.
2. Financial records successfully extracted.
3. Transactions successfully matched.
4. Transactions that have anomalies or discrepancies.
5. Transactions safe to clear under configured rules.
6. Transactions requiring human review.

A document being successfully ingested does **not** mean its invoice is valid, reconciled, or safe to pay.

---

# 2. Architecture and Development Approach

The project uses a seven-agent architecture. Agents should have well-defined responsibilities and communicate through validated data contracts.

## Agent 1: Ingestion Agent — Implemented

**Responsibility:** Read supported files, extract their available text and tables, run OCR when needed, and preserve file-level evidence.

It supports PDFs, images, and CSV files.

It identifies document categories based on folder names and filenames, calculates SHA-256 hashes, records processing status and errors, and produces a JSON batch output.

It does not make financial decisions or determine whether invoices should be paid.

## Agent 2: Extraction Agent — Implemented

**Responsibility:** Convert the Ingestion Agent's output into structured, validated financial records.

It should extract information such as:

- Vendor name and vendor ID.
- Invoice number and invoice date.
- Purchase-order number.
- Currency.
- Invoice line items.
- Item descriptions, quantities, unit prices, and line totals.
- Tax type, rate, and amount.
- Subtotal, taxable value, and grand total.
- Receipt or goods-received information.
- Bank transaction details.
- Expense-claim information.

The Extraction Agent must handle missing fields, varied date formats, inconsistent layouts, OCR mistakes, and ambiguous values.

Use Pydantic models for strict output validation. Use OpenAI where interpretation of unstructured text is required.

Every important extracted field should ideally include:

- Extracted value.
- Confidence score.
- Source document.
- Page or table reference.
- Supporting text or evidence.
- Validation warnings, if any.

Do not silently invent missing values. Represent unknown information explicitly and flag ambiguous values.

## Agent 3: Matching Agent — Implemented

**Responsibility:** Perform deterministic financial reconciliation.

The Matching Agent must not use an LLM to make the final matching decision. It should use Python rules and reproducible calculations.

Implemented capabilities include:

- Three-way matching among purchase orders, invoices, and receipts.
- Vendor matching through vendor IDs, GSTINs, and exact normalized names.
- Invoice-number normalization.
- Quantity validation.
- Unit-price validation.
- Tax and total validation.
- Configurable tolerances, implemented with `Decimal`.
- Missing-document detection.
- Duplicate invoice detection for formatting variations.
- Bank debit-to-invoice matching with amount variance and unmatched-payment findings.
- Explainable match results with source evidence.

The matching logic must distinguish acceptable differences within tolerance from genuine discrepancies.
Matching results are not payment approvals. Expense claim-to-receipt matching is not yet implemented.

## Agent 4: Anomaly Agent

**Responsibility:** Independently identify risks that ordinary matching may not detect.

Initial anomaly categories include:

- Duplicate invoices.
- Duplicate invoice numbers with formatting variations.
- Unusual amounts compared with vendor history.
- Policy violations.
- Suspicious or unexpected financial values.
- Other configured business-rule anomalies.

The Anomaly Agent must operate independently of the Matching Agent. A transaction may match its purchase order and receipt correctly but still have a duplicate-invoice or policy risk.

## Agent 5: Explanation Agent

**Responsibility:** Explain reconciliation outcomes in plain language for finance personnel.

For each exception, it should explain:

- What was expected.
- What was found.
- The calculated difference.
- The rule or evidence that triggered the exception.
- The financial amount at risk, where calculable.
- What a reviewer should investigate next.

Explanations must be grounded in structured results and source evidence. The agent must not invent discrepancies or override deterministic calculations.

## Agent 6: Human-in-the-Loop (HITL) Router

**Responsibility:** Decide whether a case may be automatically cleared or must be routed for human review.

Routing should use configurable criteria, including:

- Extraction confidence.
- OCR quality.
- Severity of discrepancies.
- Amount at risk.
- Missing or ambiguous evidence.
- Anomaly results.
- Configured business policies.

The router should generate an approval recommendation and review payload. It must not mark a transaction as safe merely because extraction succeeded.

Maintain a record of the routing decision, reasons, and relevant evidence. Human approval and final payment authorization should remain separate from the ingestion process.

## Agent 7: Correspondence Agent

**Responsibility:** Draft vendor correspondence when discrepancies need clarification or correction.

Examples include:

- Requesting a corrected invoice.
- Asking about a price variance.
- Requesting a missing purchase-order reference.
- Clarifying tax discrepancies.
- Requesting supporting documents.

The agent should use verified reconciliation findings and source evidence to prepare a draft email.

**Important boundary:** It must not automatically send emails. A human must review and approve correspondence before it is sent.

---

# 3. Technology Stack

The current project uses the following technologies:

| Component | Technology | Purpose |
|---|---|---|
| Runtime | Python 3.11+ | Main application |
| Virtual environment | Python `venv` | Dependency isolation |
| LLM | OpenAI API | Document interpretation and explanations |
| Data validation | Pydantic | Structured data contracts |
| PDF processing | PyMuPDF | PDF text extraction and page rendering |
| OCR engine | Tesseract OCR | Recognizing text in scanned documents |
| OCR integration | pytesseract | Calling Tesseract from Python |
| Image processing | Pillow | Image preparation for OCR |
| Tabular data | pandas | CSV ingestion and processing |
| Configuration | python-dotenv | Loading environment variables |
| Testing | pytest | Unit and integration tests |
| Persistence | SQLite planned | Structured records, decisions, and audit history |

Dependencies are maintained in `requirements.txt`. Keep the existing project dependency versions and working environment unless there is a demonstrated compatibility issue.

### PyMuPDF import note

The environment previously crashed when running `import fitz`. The ingestion implementation was updated to use:

```python
import pymupdf as fitz
```

This retains the existing `fitz.open()` and `fitz.Matrix()` call style while using the modern import name.

Do not add the unrelated `fitz` distribution to `requirements.txt`. The required package is `PyMuPDF`.

### Tesseract configuration

Tesseract is installed separately from Python packages. Its executable path is configured through `.env`, for example:

```dotenv
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

`run_ingestion.py` should call `load_dotenv()` before the ingestion agent runs.

Do not commit `.env` or API secrets to source control.

---

# 4. Current Project Structure

The project uses the following structure. Some folders may be empty until their corresponding agents are implemented.

```text
invoice-expense-reconciliation/
│
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── run_ingestion.py
├── run_extraction.py
├── run_matching.py
├── run_pipeline.py
│
├── data/
│   ├── input/
│   │   ├── invoices/
│   │   ├── purchase_orders/
│   │   ├── receipts/
│   │   ├── bank/
│   │   └── expenses/
│   │
│   ├── ground_truth/
│   │   └── ground_truth.json
│   │
│   ├── processed/
│   └── output/
│       ├── ingestion_output.json
│       ├── extraction_output.json
│       └── matching_output.json
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── schemas.py
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── schemas.py
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── schemas.py
│   ├── anomaly/
│   ├── explanation/
│   ├── hitl/
│   └── correspondence/
│
├── tests/
│   ├── test_extraction_agent.py
│   ├── test_matching_agent.py
│   ├── test_pipeline.py
│
└── logs/
```

Treat this as the intended architecture, not a claim that every listed file already exists.

The working ingestion implementation is located at:

- `src/ingestion/agent.py`
- `src/ingestion/schemas.py`
- `run_ingestion.py`

The ingestion output is written to:

`data/output/ingestion_output.json`

The extraction output is written to:

`data/output/extraction_output.json`

The matching output is written to:

`data/output/matching_output.json`

Run all three agents in sequence from the project directory:

```powershell
python run_pipeline.py
```

The pipeline runs `run_ingestion`, `run_extraction`, then `run_matching`.
Each stage reads the previous stage's serialized JSON output. Keep each agent's
JSON contract as the integration boundary when adding later agents.

---

# 5. Ingestion Agent: Current Implementation

## 5.1 Supported formats

The ingestion agent discovers supported files recursively beneath `data/input/`.

| Format | Processing approach |
|---|---|
| Text-based PDF | Extract text using PyMuPDF |
| Scanned PDF | Render page to image and run Tesseract OCR when the text layer is insufficient |
| JPG, JPEG, PNG, TIFF, BMP, WebP | Image preprocessing followed by OCR |
| CSV | Read with pandas and preserve headers and rows |

For PDFs, the current implementation uses a short-text threshold to decide when OCR should be used. This is a practical fallback, not a guarantee that every page with a text layer has been correctly extracted.

Table extraction is attempted when supported by the installed PyMuPDF version. A failed table-extraction attempt is logged and should not automatically invalidate otherwise usable page text.

## 5.2 Other implemented features

The agent currently provides:

- Recursive input-file discovery.
- Document-category inference from folder names and filenames.
- Unique document IDs.
- SHA-256 content hashes.
- Page-level extraction evidence.
- OCR confidence at page and document levels.
- Processing status: `success`, `partial`, or `failed`.
- Error collection.
- Atomic JSON output writing using a temporary file followed by replacement.
- Terminal summaries showing discovered, successful, partial, and failed document counts.

The ingestion layer must remain a document-processing component. Do not add invoice matching, financial clearance, or payment approval logic to it.

---

# 6. Ingestion Output Contract

The output is a JSON object representing a complete ingestion batch.

The Pydantic schemas currently define the following logical structure.

## 6.1 Batch-level structure

```json
{
  "batch_id": "generated-uuid",
  "documents": []
}
```

- `batch_id`: Unique identifier for the ingestion run.
- `documents`: Array containing one result per discovered input document.

The UUIDs shown in examples are illustrative.

## 6.2 Document-level structure

A document result contains:

| Field | Type | Meaning |
|---|---|---|
| `document_id` | String | Unique ID for this processing result |
| `source_file` | String | Path relative to the input directory, using forward slashes |
| `document_type` | String | `pdf`, `image`, `csv`, or `unknown` |
| `category` | String | `invoice`, `purchase_order`, `receipt`, `bank`, `expense`, or `other` |
| `content_hash` | String | SHA-256 hash of the source file |
| `status` | String | `success`, `partial`, or `failed` |
| `extracted_text` | String | Extracted text, including page markers for PDFs |
| `tables` | Array | Extracted tables and their source metadata |
| `pages` | Array | Page-level extraction evidence |
| `ocr_confidence` | Number or null | Document-level OCR confidence |
| `errors` | Array of strings | Errors encountered during processing |

### Document status semantics

- `success`: No processing errors were recorded.
- `partial`: Some output was produced, but one or more processing errors were recorded.
- `failed`: Processing errors occurred and no usable extracted text, tables, or page evidence were produced.

These statuses describe ingestion success, not business validity.

## 6.3 Page-level structure

Each page-evidence entry has this schema:

```json
{
  "page_number": 1,
  "extraction_method": "ocr",
  "character_count": 1250,
  "ocr_confidence": 86.4
}
```

Fields:

- `page_number`: One-based page number or the logical row for non-paginated sources.
- `extraction_method`: `text_layer`, `ocr`, `csv`, or `none`.
- `character_count`: Number of characters in the extracted text for the page or logical source.
- `ocr_confidence`: Mean valid Tesseract word-confidence score, or `null`.

For CSV, the current implementation uses page number `1` as a logical source identifier.

## 6.4 Table structure

Each extracted table has this general structure:

```json
{
  "source": "invoices/INV-01_example.pdf",
  "page_number": 1,
  "headers": [
    "Item",
    "Quantity",
    "Unit Price",
    "Amount"
  ],
  "rows": [
    ["Material A", "10", "100.00", "1000.00"]
  ]
}
```

- `source`: Source file reference.
- `page_number`: PDF page number, or `null` where the source is not paginated.
- `headers`: Column names.
- `rows`: Table rows, represented as arrays.

For CSV files, pandas supplies the column names and rows.

## 6.5 OCR confidence semantics

OCR confidence is a recognition-quality indicator, not a probability that a financial field is correct.

The implementation calculates the arithmetic mean of valid word-level Tesseract confidence scores from pages where OCR was used.

The current implementation does not weight scores by word importance or document size.

Interpret the field as follows:

- Numeric value: OCR returned valid word-level scores.
- `null`: OCR was not used, or no valid word-level confidence values were returned.

A text-based PDF can have `null` OCR confidence even when its extracted text is complete. CSV files should also have `null`.

Do not convert `null` to zero. Do not use OCR confidence alone to clear or reject an invoice.

## 6.6 Example output

The following example illustrates a scanned invoice. Values are illustrative and do not represent an actual test result.

```json
{
  "batch_id": "batch-uuid",
  "documents": [
    {
      "document_id": "document-uuid",
      "source_file": "invoices/INV-10_scanned.pdf",
      "document_type": "pdf",
      "category": "invoice",
      "content_hash": "example-sha256-hash",
      "status": "success",
      "extracted_text": "--- Page 1 ---\nExample Supplier Invoice",
      "tables": [],
      "pages": [
        {
          "page_number": 1,
          "extraction_method": "ocr",
          "character_count": 34,
          "ocr_confidence": 86.4
        }
      ],
      "ocr_confidence": 86.4,
      "errors": []
    }
  ]
}
```

The downstream Extraction Agent should consume this contract rather than directly reopening original files unnecessarily.

However, it must preserve `source_file` and evidence references so it can request additional source information when extraction is ambiguous.

---

# 7. Ground Truth and Evaluation Dataset

A separate ground-truth dataset has been defined for evaluating the reconciliation system.

Its intended location is:

`data/ground_truth/ground_truth.json`

The dataset includes:

- Generation date and currency.
- Configured matching tolerances.
- Eleven invoice cases.
- Bank transactions.
- Expense claims.
- Expected statuses and exception labels.
- Explanations of expected discrepancies.
- Notes identifying unpaid invoices that must not be cleared for payment as-is.

## 7.1 Ground-truth metadata

The provided configuration includes:

```json
{
  "generated": "2026-10-27",
  "currency": "INR",
  "tolerances": {
    "unit_price_pct": 1.0,
    "total_rounding_inr": 1.0,
    "qty": "billed qty must not exceed received qty"
  }
}
```

Use the actual ground-truth file as the authoritative source for all fixture data. Do not recreate missing records from this excerpt.

## 7.2 Important invoice test cases

The dataset intentionally includes multiple classes of behavior:

- Valid invoices that should clear.
- Price variance.
- Quantity mismatch.
- Duplicate invoice.
- Missing purchase order.
- Tax error, including an incorrect tax type.
- Incorrect invoice total.
- A price difference within configured tolerance that should not be flagged.
- A duplicate invoice whose number has formatting differences.
- An unpaid-invoice note that must be respected during clearance decisions.

Some known example relationships:

- `INV-04` tests a price difference within tolerance and helps detect false positives.
- `INV-06` duplicates `INV-04` with a differently formatted invoice number.
- `INV-08` tests an incorrect tax type, with IGST versus CGST and SGST.
- `INV-11` tests a discrepancy between the printed total and the arithmetic correct total.

The complete fixture definitions and expected statuses must be read from the actual JSON file.

## 7.3 Bank transactions and expense claims

The bank dataset includes matched transactions, a payment variance, an unmatched payment, and out-of-scope credits or charges.

The expense dataset includes a clear claim (`EXP-301`) and a mismatch (`EXP-302`).

The system should not force unrelated bank credits or charges into invoice matches merely because their amounts are similar.

## 7.4 Preventing evaluation leakage

**Ground truth must remain separate from runtime agent inputs.**

The runtime agents must not read expected statuses, expected exception labels, or explanatory answers when processing an invoice.

Instead:

1. Agents process source documents normally.
2. They generate extraction and reconciliation results.
3. A separate evaluation layer compares the results with ground truth.
4. Evaluation reports identify correct decisions, missed exceptions, false positives, and extraction errors.

This separation is essential for meaningful evaluation.

Ground truth should be used to measure system quality, not to tell the agents what answer to produce.

---

# 8. Financial and Safety Constraints

These constraints are mandatory across the project.

## 8.1 Deterministic financial decisions

Do not use an LLM as the final authority for:

- Arithmetic.
- Tax calculations.
- Quantity and unit-price comparisons.
- Configured tolerance checks.
- Duplicate detection rules.
- Payment matching.
- Approval thresholds.

An LLM may interpret unstructured information, normalize candidate values, or explain a result. Deterministic Python must calculate and validate the financial result.

## 8.2 Evidence and auditability

Every exception should eventually be traceable to:

- Source document.
- Extracted field or table.
- Calculated expected value.
- Observed value.
- Difference.
- Rule applied.
- Resulting status.
- Explanation and review decision.

Never discard extraction warnings simply because the overall document status is `success`.

## 8.3 Payment safety

An invoice is not safe to pay simply because it was extracted successfully or matched one related document.

The system must consider outstanding exceptions, duplicates, missing evidence, and applicable approval rules.

The ground-truth unpaid-invoice note identifies `INV-03`, `INV-05`, `INV-06`, `INV-07`, `INV-08`, and `INV-11` as unpaid and not to be cleared for payment as-is.

Treat these as evaluation constraints. Do not expose them as runtime hints to the agents.

## 8.4 Human oversight

Human review is required for cases that fail configured clearance criteria.

The Correspondence Agent only drafts messages; it does not send them automatically.

---

# 9. Current Development Status

## Completed

- Windows project setup and Python virtual environment.
- Environment configuration using `.env`.
- OpenAI API connection test.
- Initial project dependencies and directory structure.
- Ingestion schemas for documents, tables, pages, and batches.
- Recursive discovery of supported PDFs, images, and CSVs.
- PDF text extraction.
- OCR fallback for scanned PDFs.
- Image OCR.
- CSV ingestion.
- Document categorization.
- SHA-256 hashing.
- Page-level OCR confidence.
- Document-level OCR confidence.
- JSON batch output.
- Basic error handling and terminal processing summary.
- Resolution of the duplicate `PageEvidence` schema definition that caused Pydantic validation errors.
- Successful execution of the ingestion workflow.
- Pydantic extraction contracts for invoices, purchase orders, receipts,
  bank transactions, expense claims, and vendor records.
- Deterministic CSV extraction for bank statements, expense claims, purchase
  orders, goods receipts/service entries, and vendor master rows. CSV record
  type is identified from table headers as well as the ingestion category,
  because the vendors file is currently categorized as `other`.
- Structured OpenAI extraction for invoice documents and image-based receipts.
- Field-level raw values, normalized values, confidence estimates, source
  references, evidence quotes/locations, and validation warnings.
- Deterministic date/decimal parsing and invoice arithmetic checks using
  `Decimal`; extraction warnings do not make silent corrections to source data.
- Extraction runner `run_extraction.py`, which validates the ingestion JSON
  and atomically writes `data/output/extraction_output.json`.
- Three-agent orchestration in `run_pipeline.py`, which runs ingestion,
  extraction, then matching using each stage's JSON output.
- Deterministic matching schemas and `MatchingAgent` for PO/invoice/receipt
  comparisons, duplicate invoice detection, and bank payment matching.
- `run_matching.py` for standalone matching and `matching_output.json` as the
  serialized match-results contract.
- Three-agent orchestration in `run_pipeline.py`: ingestion JSON ->
  extraction JSON -> matching JSON. Later agents should consume the preceding
  stage's validated JSON contract.
- Matching checks include vendor/PO references, line identity, ordered and
  received quantities, unit-price tolerance, tax amounts/rates and (when
  buyer GSTIN is available) GST jurisdiction, invoice arithmetic, duplicate
  invoice numbers, bank reference/vendor evidence, and payment amount variance.
- Configurable matching defaults: INR currency, 1% unit-price tolerance,
  INR 1.00 total/tax/payment tolerance, and minimum critical-field confidence
  of 0.6.
- Automated extraction, matching, and pipeline tests. The full current test
  suite passes: `python -m pytest -q` (39 tests at the last verification).
- A generated `data/output/extraction_output.json` was inspected: 18 source
  documents; 18 document-level extraction statuses marked `success`; record
  types/counts were 11 invoices, 11 receipts, 9 purchase orders, 9 bank
  transactions, 2 expense claims, and 5 vendors.
- The current serialized `data/output/matching_output.json` was inspected:
  11 invoice outcomes and 9 bank-payment outcomes were generated from the
  extraction batch. Matching output is an exception report, not a payment
  authorization.

Run the end-to-end three-agent sequence with:

```powershell
python run_pipeline.py
```

The OpenAI API key and model from `.env` are needed for invoice and image
receipt interpretation. Structured CSV documents can be extracted without
calling the LLM. If unstructured extraction needs the LLM but it is unavailable,
the document is explicitly marked `failed`; it is not silently omitted.

## Known extraction limitations and next validation

- LLM invoice/receipt extraction has unit-test coverage using a fake client;
  no live OpenAI integration test is part of the test suite. Live API behavior
  depends on the configured model supporting the Responses API JSON Schema
  format.
- LLM confidence is heuristic, not calibrated. OCR confidence remains a
  separate ingestion signal.
- LLM-supplied evidence quotes are checked against ingested text; unverifiable
  quotes are removed and warned about. Continue evaluating field accuracy
  against source evidence before using extracted data for matching.
- Invoice arithmetic checks report inconsistencies; they do not modify
  extracted amounts or decide whether an invoice should be paid.
- Document-level `success` means extraction produced validated records, not
  that every financial value is correct or that a document is safe to pay.
- The extraction output is an intermediate contract. It does not contain
  matching, anomaly, clearance, or payment decisions.
- Invoice extraction now includes optional `buyer_gstin` so matching can
  validate IGST versus CGST/SGST using supplier and buyer state codes. The
  current extraction and matching output files were generated after this
  field was added.

## Known implementation considerations

- PyMuPDF must be imported using the working import style `import pymupdf as fitz` in this environment.
- Tesseract must be available to Python through `TESSERACT_CMD` or the system PATH.
- OCR confidence currently uses an unweighted mean of valid word-level scores.
- The current OCR helper performs separate OCR calls to obtain confidence data and formatted text. This can be optimized later.
- PDF table extraction depends on support in the installed PyMuPDF version.
- OCR confidence is not the same as extraction confidence for an individual invoice field.
- A successful ingestion result does not imply that every expected field has been extracted correctly.

Do not destabilize working ingestion, extraction, or matching behavior while
building the next agent. Changes should be incremental and covered by tests.

---

# 10. Extraction Agent: Current Implementation

The Extraction Agent milestone is implemented. Its serialized interface is
`data/output/ingestion_output.json` in and
`data/output/extraction_output.json` out. The next milestone is the Anomaly
Agent; do not redo extraction unless evaluation reveals a concrete defect.

## 10.1 Objective and record types

The agent consumes the validated `IngestionBatch` and produces an
`ExtractionBatch` with a document result for each source document. The
implemented financial record types are:

- Invoice.
- Purchase order.
- Receipt/goods receipt/service entry.
- Bank transaction.
- Expense claim.
- Vendor master record.

Missing or ambiguous values remain null/unknown and carry warnings rather than
being invented.

## 10.2 Implementation files and run sequence

Implemented files:

- `src/extraction/schemas.py`
- `src/extraction/agent.py`
- `src/extraction/__init__.py`
- `run_extraction.py` for extraction-only execution.
- `run_pipeline.py` to run ingestion, extraction, and matching serially.
- `tests/test_extraction_agent.py`
- `tests/test_pipeline.py`

Run all three agents together from PowerShell in the repository root:

```powershell
python run_pipeline.py
```

The runner writes `data/output/ingestion_output.json`, passes it to
`run_extraction` to produce `data/output/extraction_output.json`, then passes
that file to `run_matching` to write `data/output/matching_output.json`.
Individual stages can be run with `python run_ingestion.py`,
`python run_extraction.py`, or `python run_matching.py`.

## 10.3 Extraction data contract

The schemas use strict Pydantic models with extra fields forbidden, an
`ExtractionBatch`/`DocumentExtraction` wrapper, and discriminated financial
record types. Important extracted values use `Decimal` or `date` as
appropriate. The generic `ExtractedField` includes:

- Normalized `value`, nullable when missing/unparseable.
- `raw_value` as found or returned by the extractor.
- A nullable confidence estimate from 0 to 1.
- Source document, source path, optional quote, page/table row, and column.
- Validation and ambiguity warnings.

CSV row values use deterministic parsing and high confidence when successfully
parsed from a source cell; empty or malformed cells remain unknown and warn.
Invoice and receipt values are extracted through OpenAI structured JSON Schema
output. LLM values are Pydantic-validated, converted to typed values, and
evidence quotes are checked against ingested document text. OCR confidence is
not copied into field confidence.

## 10.4 Validation and remaining work

The extraction tests cover valid and malformed structured outputs, missing
fields, date and decimal normalization, OCR-confidence separation, evidence
retention/verification, arithmetic warnings, CSV row provenance, unsupported
categories, and atomic output. Run:

```powershell
python -m pytest -q
```

The last full verification completed with 39 passing tests. Before relying on
the results, inspect and evaluate the actual extracted invoice records against
their source documents; a successful extraction status is not a correctness or
payment-clearance guarantee.

---

# 11. Matching Agent: Current Implementation

The Matching Agent consumes the validated
`data/output/extraction_output.json` contract and writes
`data/output/matching_output.json`. It is deterministic and does not use an
LLM for financial decisions.

## 11.1 Implemented functionality

- Pydantic contracts: `src/matching/schemas.py`.
- Matching logic: `src/matching/agent.py`.
- Standalone runner: `run_matching.py`.
- Pipeline stage order: ingestion -> extraction -> matching in
  `run_pipeline.py`.
- Matching contract includes invoice outcomes and bank-payment outcomes, with
  explicit issues, statuses, source evidence, and configuration used.
- Invoice outcomes are `matched`, `exception`, or `review`; bank transaction
  outcomes can also be `ignored` for credits/opening-balance rows and bank
  charges outside AP payment matching.
- PO and receipt items are matched by normalized item code, then exact
  normalized description. Line-number fallback is only used when descriptions
  are absent; no fuzzy/LLM matching is done.
- Invoice number normalization removes punctuation and normalizes leading
  zeros in digit components (e.g. `DOS/2026/0417` and `DOS-2026-417`).
- Matching reports PO/vendor mismatch, missing PO/receipt, item and quantity
  discrepancy, price variance beyond tolerance, tax amount/rate/type mismatch,
  invoice arithmetic mismatch, duplicate invoice, and bank-payment variance or
  unmatched debit.
- GST jurisdiction is validated only when seller and buyer GSTIN/state codes
  are available. Invoice extraction schema now exposes `buyer_gstin`; old
  output JSON must be regenerated to take advantage of it.
- The matching result is evidence and exception reporting only. It never
  authorizes payment or treats a match as payment approval.

Default configuration is declared in `MatchingConfig`: currency INR, 1%
unit-price tolerance, INR 1.00 amount tolerance, and a 0.6 minimum confidence
for critical fields. Callers may provide a different config. Runtime matching
does not read ground-truth expected statuses or exception labels.

Tests: `tests/test_matching_agent.py`. Run the full suite with:

```powershell
python -m pytest -q
```

The current saved matching output contains 11 invoice outcomes (4 matched,
7 exceptions) and 9 bank-payment outcomes (4 matched, 2 exceptions, 3 ignored).
The invoice exceptions include price variance, quantity exceeding receipt,
duplicate invoice, missing PO, invoice-total arithmetic mismatch, and tax
type/rate/component findings. Review the source document and extracted tax
evidence for the tax findings before treating them as confirmed discrepancies;
matching is deterministic over extracted values and does not correct an
extraction error. Expense claim-to-receipt amount reconciliation is still
outstanding.

## Known matching limitations

- Expense claims are extracted but not yet reconciled against receipt totals.
- Exact normalized item descriptions are used when item codes are unavailable;
  semantically equivalent but differently worded descriptions are not guessed.
- Vendor names in bank narrations use a conservative token overlap heuristic.
  Ambiguous candidates are sent to review; evaluate it against additional bank
  statement formats before relying on it broadly.
- GST rate calculations use PO rates and invoice taxable bases. A match does
  not establish tax compliance beyond the checks represented by these fields.

---

# 12. Subsequent Development Roadmap

The next milestone is the Anomaly Agent. Continue in this order:

## Milestone 4: Anomaly Agent

- Implement duplicate detection independently of matching.
- Add configurable policy rules and historical outlier checks.
- Return anomaly types, evidence, severity, and relevant financial amounts.

## Milestone 5: Explanation Agent

- Generate finance-friendly explanations grounded in structured findings.
- Include expected versus observed amounts and calculated differences.
- Validate that explanations agree with the deterministic results.

## Milestone 6: HITL Router

- Define configurable clearance and review criteria.
- Combine extraction confidence, OCR quality, discrepancy severity, and financial risk.
- Produce auditable routing decisions and review payloads.
- Enforce the unpaid-invoice and exception constraints.

## Milestone 7: Correspondence Agent

- Draft vendor clarification and dispute emails from verified findings.
- Include relevant invoice identifiers and discrepancy details.
- Require human approval before sending.

## Milestone 8: Integration and Evaluation

- Connect all agents through stable, validated JSON contracts.
- Compare outputs with the separate ground-truth dataset.
- Measure extraction/matching accuracy, false positives, missed exceptions,
  and clearance safety.
- Add end-to-end tests and audit logging.
- Optimize performance after correctness is established.

---

# 13. Instructions for the Next AI Development Agent

You are continuing an existing project. Do not treat this as a greenfield implementation.

Follow these instructions:

1. **Continue with the Anomaly Agent.** Ingestion, extraction, and matching
   are implemented, and `run_pipeline.py` serializes their outputs in order.
2. Read the current extraction and matching schemas, runners, and output
   contracts before designing the next agent.
3. Use `data/output/matching_output.json` as the Anomaly Agent input; preserve
   source and field evidence in anomaly outcomes.
4. Inspect the real ground-truth schema and source documents for evaluation,
   but never use expected statuses or exception labels from ground truth to
   drive runtime matching decisions.
5. Keep anomaly detection independent of matching; matching already identifies
   deterministic duplicate invoice numbers and financial discrepancies.
6. Preserve extraction warnings, match issues, and confidence signals; a
   successful match is not a clearance decision.
7. Use `Decimal` for financial arithmetic where relevant and add focused tests.
8. Extend `run_pipeline.py` only after the new agent has a stable, validated
   input/output contract. Keep JSON contracts as inter-agent boundaries.
9. Preserve existing ingestion/extraction/matching behavior and unrelated working-tree
   changes. Do not commit credentials, send email, initiate payment, or add
   automatic payment-approval behavior.
10. Run `python -m pytest -q` and validate output contracts before
    declaring the next milestone complete.
11. Keep the project suitable for a three-developer MVP; prefer a correct,
    auditable end-to-end flow over premature complexity.

**Current success criterion:** Build and test an Anomaly Agent that reads
matching output independently, identifies non-matching risks with evidence,
and does not approve or pay invoices.