# Invoice Anomaly Agent

The deterministic anomaly agent consumes the Extraction Agent batch JSON (or
legacy results array) and returns a report with `analyzed_invoice_count` and
structured `findings`. It flags:

- **Duplicate invoices** (`DUPLICATE_INVOICE`, high severity): vendor, currency,
  and normalized invoice number match; or, when invoice numbers differ, vendor,
  date, currency, total, and the complete line-item identity/quantity set match.
  Line-item identity prefers `item_code` (also accepts `product_id` and `sku`);
  if all code fields are null or blank, it falls back to normalized description.
  Description text is not compared when a usable product code is present.
- **Vendor-history outliers** (`VENDOR_AMOUNT_OUTLIER`, medium severity):
  compares an invoice to supplied historical totals for the same normalized
  vendor and currency. With at least three history values by default, it
  calculates the median and median absolute deviation (MAD), then flags a
  modified z-score above 3.5. If MAD is zero, any amount different from the
  median is flagged.
- **Policy-limit breaches** (`ABOVE_POLICY_LIMIT`, high severity): flags totals
  strictly greater than the configured currency limit. The example default is
  INR 100,000; set the limit according to your organization's approved policy.

## Vendor History

Vendor history is optional. The preferred input is a vendor-to-currency-to-list
mapping, with at least three trusted historical invoice totals per currency:

```json
{
  "Deccan Office Supplies": {
    "INR": [48000, 51000, 49500]
  }
}
```

The legacy form `{"Vendor Name": [48000, 51000, 49500]}` is also accepted; its
values are treated as the candidate invoice's currency. History should be
curated from known valid invoices and exclude duplicates, reversals, credits,
and exceptional one-off transactions. The implementation uses the history
provided for each run; it does not persist or independently validate history.

## Run the Agent

Run these commands from the repository root. Regenerate extraction results if
the extraction schema has changed so the output includes available item codes:

```powershell
python -m src.extraction_agent.batch_cli
python -m src.anomaly.batch_cli `
  --input data/output/extraction_output.json `
  --output data/output/anomaly_results.json `
  --vendor-history data/input/vendor_history.json `
  --max-invoice-amount 100000 `
  --log-level INFO
```

For the default input and policy limit, run the root-level convenience script:

```powershell
python run_anomaly.py
```

It reads `data/output/extraction_output.json` and writes
`data/output/anomaly_results.json`.

The report defaults to `data/output/anomaly_results.json`. Omit
`--vendor-history` when history is unavailable; duplicate and policy checks
still run. Configure policy programmatically with `AnomalyPolicy`, for example:

```python
from decimal import Decimal

from src.anomaly import AnomalyAgent, AnomalyPolicy

policy = AnomalyPolicy(
    max_invoice_amount_by_currency={
        "INR": Decimal("100000"),
        "USD": Decimal("2500"),
    },
    outlier_modified_z_threshold=Decimal("3.5"),
    min_vendor_history=5,
)
report = AnomalyAgent(policy).analyze(extraction_results, vendor_history)
```

The Python API is available as
`AnomalyAgent().analyze(extraction_results, vendor_history)`.

## Logging

The CLI defaults to `INFO`; use `--log-level` with `DEBUG`, `WARNING`, or
`ERROR` to adjust verbosity. Logs include run counts, finding categories, and
invoice identifiers when a finding occurs. They do not include vendor names
or invoice amounts. When using the Python API, configure logging in the calling
application; the module does not configure global logging handlers.

## Tests

Run the anomaly tests or the entire suite from the repository root:

```powershell
python -m pytest tests\test_anomaly_agent.py -q
python -m pytest tests -q
```

## Limitations and Review

Vendor-history detection is a statistical screening rule, not a payment
decision. Its effectiveness depends on history quality, vendor identity
resolution, and policy calibration. The default three-record minimum and 3.5
modified-z threshold are configurable through `AnomalyPolicy`, but should be
validated against representative labeled invoices before production use.
Findings should be reviewed alongside extraction confidence and supporting
documents; do not treat an anomaly finding alone as proof of fraud or a reason
to automatically approve or reject payment.
