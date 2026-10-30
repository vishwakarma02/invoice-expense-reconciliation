Invoice reconciliation input use cases
======================================

Each numbered folder is a separate upload scenario. In the demo UI, choose all
files from ONE scenario folder, then click Run reconciliation pipeline.
Do not combine scenario folders: each contains its own invoice set and shared
reference CSVs.

Scenarios
---------
01-clean-three-way-match
  One sample invoice intended to pass the invoice/PO/receipt matching checks.

02-matching-tax-exception
  One sample invoice with tax-related matching exceptions.

03-duplicate-invoice
  Two sample invoices sharing a normalized invoice reference; exercises
  matching duplicate detection and independent anomaly detection.

04-policy-limit-and-price-variance
  One sample invoice with a PO price variance and an amount above the default
  INR 100,000 anomaly policy threshold.

Every folder includes purchase_orders.csv, goods_receipts.csv, and vendors.csv.
The invoices and reference CSVs are copied from this repository's data/input
sample set. No ground-truth file or generated output is included.

Run requirements
----------------
- Set OPENAI_API_KEY and OPENAI_MODEL in the project .env for PDF invoice
  extraction.
- These scenarios use text-layer PDFs, so they do not require Tesseract OCR.
- The UI accepts PDF, image, and CSV files. Select all files from only one
  scenario folder for each run.
- Outputs remain in the UI's run-specific output directory.
