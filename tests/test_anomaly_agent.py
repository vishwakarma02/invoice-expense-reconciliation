import logging
from decimal import Decimal

import pytest

from src.anomaly import AnomalyAgent, AnomalyPolicy


def extracted_invoice(
    invoice_id: str,
    vendor: str,
    number: str,
    amount: str,
    invoice_date: str = "2026-10-01",
    *,
    currency: str = "INR",
    line_items: list[dict] | None = None,
) -> dict:
    if line_items is None:
        line_items = [
            {
                "item_code": {"value": "SKU-001"},
                "description": {"value": "Widget A"},
                "quantity": {"value": "2"},
                "unit_price": {"value": "50"},
                "total": {"value": "100"},
            }
        ]
    return {
        "status": "success",
        "document_id": invoice_id,
        "record": {
            "document_type": "invoice",
            "invoice_number": {"value": number},
            "invoice_date": {"value": invoice_date},
            "vendor_name": {"value": vendor},
            "currency": {"value": currency},
            "grand_total": {"value": amount},
            "line_items": line_items,
        },
    }


def test_detects_duplicate_invoice_with_reformatted_reference():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice(
                "INV-04", "Deccan Office Supplies", "DOS/2026/0417", "54832.40"
            ),
            extracted_invoice(
                "INV-06", "Deccan Office Supplies", "DOS-2026-417", "54832.40"
            ),
        ]
    )

    assert report.analyzed_invoice_count == 2
    assert len(report.findings) == 1
    assert report.findings[0].anomaly_type == "DUPLICATE_INVOICE"
    assert report.findings[0].duplicate_of == "INV-04"


def test_duplicate_requires_same_vendor_for_matching_invoice_reference():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice("INV-1", "Vendor A", "2026/001", "100"),
            extracted_invoice("INV-2", "Vendor B", "2026-001", "100"),
        ]
    )

    assert report.findings == []


def test_same_description_with_different_item_code_is_not_duplicate():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "100"),
            extracted_invoice(
                "INV-2",
                "Acme",
                "A-2",
                "100",
                line_items=[
                    {
                        "item_code": {"value": "SKU-002"},
                        "description": {"value": "Widget A"},
                        "quantity": {"value": "2"},
                    }
                ],
            ),
        ]
    )

    assert report.findings == []


def test_same_vendor_date_and_total_but_different_quantity_is_not_duplicate():
    first = extracted_invoice("INV-1", "Acme", "A-1", "100")
    second = extracted_invoice(
        "INV-2",
        "Acme",
        "A-2",
        "100",
        line_items=[
            {
                "item_code": {"value": "SKU-001"},
                "description": {"value": "Widget A"},
                "quantity": {"value": "3"},
            }
        ],
    )

    assert AnomalyAgent().analyze([first, second]).findings == []


def test_detail_match_ignores_line_order_and_normalizes_item_codes():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice(
                "INV-1",
                "Acme Ltd.",
                "A-1",
                "300",
                line_items=[
                    {
                        "item_code": {"value": "SKU-A"},
                        "description": {"value": "Widget-A"},
                        "quantity": {"value": "2"},
                    },
                    {
                        "item_code": {"value": "SKU-B"},
                        "description": {"value": "Widget B"},
                        "quantity": {"value": "1"},
                    },
                ],
            ),
            extracted_invoice(
                "INV-2",
                "ACME, LTD",
                "A-2",
                "300",
                line_items=[
                    {
                        "item_code": {"value": "sku-b"},
                        "description": {"value": "unrelated description"},
                        "quantity": {"value": "1.0"},
                    },
                    {
                        "item_code": {"value": "sku-a"},
                        "description": {"value": "another description"},
                        "quantity": {"value": "2.00"},
                    },
                ],
            ),
        ]
    )

    assert len(report.findings) == 1
    assert report.findings[0].anomaly_type == "DUPLICATE_INVOICE"
    assert report.findings[0].duplicate_of == "INV-1"


@pytest.mark.parametrize("code_field", ["product_id", "sku"])
def test_detail_match_accepts_common_product_code_field_names(code_field):
    report = AnomalyAgent().analyze(
        [
            extracted_invoice(
                "INV-1",
                "Acme",
                "A-1",
                "100",
                line_items=[
                    {
                        code_field: {"value": "SKU-001"},
                        "quantity": {"value": "2"},
                    }
                ],
            ),
            extracted_invoice(
                "INV-2",
                "Acme",
                "A-2",
                "100",
                line_items=[
                    {
                        "item_code": {"value": "sku001"},
                        "quantity": {"value": "2"},
                    }
                ],
            ),
        ]
    )

    assert len(report.findings) == 1
    assert report.findings[0].duplicate_of == "INV-1"


@pytest.mark.parametrize(
    ("second_date", "second_currency", "second_total"),
    [
        ("2026-10-02", "INR", "100"),
        ("2026-10-01", "USD", "100"),
        ("2026-10-01", "INR", "101"),
    ],
)
def test_detail_match_requires_same_date_currency_and_total(
    second_date, second_currency, second_total
):
    report = AnomalyAgent().analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "100"),
            extracted_invoice(
                "INV-2",
                "Acme",
                "A-2",
                second_total,
                second_date,
                currency=second_currency,
            ),
        ]
    )

    assert report.findings == []


def test_missing_quantity_or_absent_line_items_do_not_confirm_detail_match():
    first = extracted_invoice("INV-1", "Acme", "A-1", "100")
    without_lines = extracted_invoice("INV-2", "Acme", "A-2", "100", line_items=[])
    missing_item_code = extracted_invoice(
        "INV-3",
        "Acme",
        "A-3",
        "100",
        line_items=[
            {
                "item_code": {"value": "SKU-001"},
                "description": {"value": "Widget A"},
            }
        ],
    )
    missing_quantity = extracted_invoice(
        "INV-4",
        "Acme",
        "A-4",
        "100",
        line_items=[
            {
                "description": {"value": "Widget A"},
                "quantity": {"value": "2"},
            }
        ],
    )

    assert AnomalyAgent().analyze(
        [first, without_lines, missing_quantity, missing_item_code]
    ).findings == []


@pytest.mark.parametrize("empty_item_code", [None, ""])
def test_missing_item_code_falls_back_to_matching_description(empty_item_code):
    first = extracted_invoice(
        "INV-1",
        "Acme",
        "A-1",
        "100",
        line_items=[
            {
                "item_code": {"value": empty_item_code},
                "description": {"value": "Widget A"},
                "quantity": {"value": "2"},
            }
        ],
    )
    second = extracted_invoice(
        "INV-2",
        "Acme",
        "A-2",
        "100",
        line_items=[
            {
                "item_code": {"value": empty_item_code},
                "description": {"value": "widget-a"},
                "quantity": {"value": "2.0"},
            }
        ],
    )

    report = AnomalyAgent().analyze([first, second])

    assert len(report.findings) == 1
    assert report.findings[0].duplicate_of == "INV-1"


def test_missing_item_code_with_different_description_is_not_duplicate():
    first = extracted_invoice(
        "INV-1",
        "Acme",
        "A-1",
        "100",
        line_items=[
            {
                "item_code": {"value": None},
                "description": {"value": "Widget A"},
                "quantity": {"value": "2"},
            }
        ],
    )
    second = extracted_invoice(
        "INV-2",
        "Acme",
        "A-2",
        "100",
        line_items=[
            {
                "item_code": {"value": ""},
                "description": {"value": "Widget B"},
                "quantity": {"value": "2"},
            }
        ],
    )

    assert AnomalyAgent().analyze([first, second]).findings == []


def test_matching_invoice_reference_is_duplicate_even_if_line_items_differ():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "100"),
            extracted_invoice(
                "INV-2",
                "ACME",
                "A/1",
                "200",
                line_items=[
                    {
                        "item_code": {"value": "SKU-OTHER"},
                        "description": {"value": "Different item"},
                        "quantity": {"value": "7"},
                    }
                ],
            ),
        ]
    )

    assert len(report.findings) == 1
    assert report.findings[0].duplicate_of == "INV-1"


def test_duplicate_detection_does_not_match_different_currencies():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "100", currency="INR"),
            extracted_invoice("INV-2", "Acme", "A-1", "100", currency="USD"),
        ]
    )

    assert report.findings == []


def test_detects_amount_outlier_against_vendor_history():
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "50000")],
        {" acme ": ["9000", "10000", "11000", "10000"]},
    )

    assert [finding.anomaly_type for finding in report.findings] == [
        "VENDOR_AMOUNT_OUTLIER"
    ]


def test_outlier_detected_when_historical_median_absolute_deviation_is_zero():
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "12000")],
        {"Acme": ["10000", "10000", "10000"]},
    )

    assert [finding.anomaly_type for finding in report.findings] == [
        "VENDOR_AMOUNT_OUTLIER"
    ]


def test_amount_near_historical_median_is_not_an_outlier():
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "10500")],
        {"Acme": ["9000", "10000", "11000", "10000"]},
    )

    assert report.findings == []


def test_currency_specific_vendor_history_only_applies_to_matching_currency():
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "50000", currency="USD")],
        {"Acme": {"INR": ["9000", "10000", "11000"]}},
    )

    assert report.findings == []


def test_legacy_history_array_applies_to_invoice_currency():
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "50000", currency="USD")],
        {"Acme": ["9000", "10000", "11000", "10000"]},
    )

    assert [finding.anomaly_type for finding in report.findings] == [
        "VENDOR_AMOUNT_OUTLIER"
    ]


def test_flags_only_amounts_strictly_above_policy_limit():
    agent = AnomalyAgent(
        AnomalyPolicy(max_invoice_amount_by_currency={"INR": Decimal("100000")})
    )
    report = agent.analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "100000"),
            extracted_invoice(
                "INV-2", "Other Vendor", "B-1", "100000.01", "2026-10-02"
            ),
        ]
    )

    assert [finding.invoice_id for finding in report.findings] == ["INV-2"]
    assert report.findings[0].anomaly_type == "ABOVE_POLICY_LIMIT"


def test_logs_analysis_summary_and_findings_without_invoice_amounts_or_vendor_names(
    caplog,
):
    with caplog.at_level(logging.INFO, logger="src.anomaly.agent"):
        report = AnomalyAgent().analyze(
            [extracted_invoice("INV-1", "Private Vendor", "A-1", "100001")]
        )

    assert len(report.findings) == 1
    assert "Starting invoice anomaly analysis" in caplog.text
    assert "Invoice exceeds the configured policy limit" in caplog.text
    assert "Completed invoice anomaly analysis" in caplog.text
    assert "invoice_id=INV-1" in caplog.text
    assert "Private Vendor" not in caplog.text
    assert "100001" not in caplog.text


@pytest.mark.parametrize("amount", ["99999.99", "100000"])
def test_amount_at_or_below_policy_limit_is_not_flagged(amount):
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", amount)]
    )

    assert report.findings == []


def test_policy_limit_can_be_set_for_multiple_currencies():
    agent = AnomalyAgent(
        AnomalyPolicy(
            max_invoice_amount_by_currency={
                "INR": Decimal("100000"),
                "USD": Decimal("1000"),
            }
        )
    )
    report = agent.analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "1001", currency="USD"),
            extracted_invoice("INV-2", "Acme", "A-2", "999", currency="USD"),
            extracted_invoice("INV-3", "Acme", "A-3", "100001", currency="INR"),
        ]
    )

    assert [
        (finding.invoice_id, finding.anomaly_type) for finding in report.findings
    ] == [
        ("INV-1", "ABOVE_POLICY_LIMIT"),
        ("INV-3", "ABOVE_POLICY_LIMIT"),
    ]


def test_invoice_without_a_policy_for_its_currency_is_not_flagged():
    report = AnomalyAgent(
        AnomalyPolicy(max_invoice_amount_by_currency={"INR": Decimal("100000")})
    ).analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "999999", currency="USD")]
    )

    assert report.findings == []


@pytest.mark.parametrize(
    "policy_values",
    [
        {"max_invoice_amount_by_currency": {"INR": Decimal("-1")}},
        {"max_invoice_amount_by_currency": {"INR": Decimal("Infinity")}},
        {"outlier_modified_z_threshold": Decimal("0")},
        {"min_vendor_history": 2},
        {"unexpected_option": True},
    ],
)
def test_invalid_policy_is_rejected(policy_values):
    with pytest.raises(ValueError):
        AnomalyPolicy(**policy_values)


def test_ignores_non_invoice_records_and_requires_complete_invoice_data():
    report = AnomalyAgent().analyze(
        [{"record": {"document_type": "generic_table"}}]
    )
    assert report.analyzed_invoice_count == 0
    assert report.findings == []

    with pytest.raises(ValueError, match="invoice_number"):
        AnomalyAgent().analyze(
            [{"record": {"document_type": "invoice", "grand_total": {"value": "5"}}}]
        )


@pytest.mark.parametrize(
    ("amount", "message"),
    [
        ("not-a-number", "valid number"),
        ("-1", "non-negative"),
        ("NaN", "finite non-negative"),
        ("Infinity", "finite non-negative"),
    ],
)
def test_invalid_invoice_amount_is_rejected(amount, message):
    with pytest.raises(ValueError, match=message):
        AnomalyAgent().analyze(
            [extracted_invoice("INV-1", "Acme", "A-1", amount)]
        )


def test_failed_extractions_are_skipped():
    failed = extracted_invoice("INV-1", "Acme", "A-1", "100")
    failed["status"] = "extraction_failed"

    report = AnomalyAgent().analyze([failed])

    assert report.analyzed_invoice_count == 0
    assert report.findings == []


def test_malformed_extraction_input_is_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        AnomalyAgent().analyze([None])
    with pytest.raises(ValueError, match="array of objects"):
        AnomalyAgent().analyze({"record": "not-an-array"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("invoice_date", "2026-13-01", "YYYY-MM-DD"),
        ("invoice_date", "20261001", "YYYY-MM-DD"),
        ("invoice_number", "", "non-empty text"),
        ("vendor_name", 7, "non-empty text"),
    ],
)
def test_invalid_invoice_identity_fields_are_rejected(field, value, message):
    invoice = extracted_invoice("INV-1", "Acme", "A-1", "100")
    invoice["record"][field] = {"value": value}

    with pytest.raises(ValueError, match=message):
        AnomalyAgent().analyze([invoice])


def test_duplicate_invoice_ids_are_rejected():
    with pytest.raises(ValueError, match="identifiers must be unique"):
        AnomalyAgent().analyze(
            [
                extracted_invoice("INV-1", "Acme", "A-1", "100"),
                extracted_invoice("INV-1", "Acme", "A-2", "100"),
            ]
        )


def test_invalid_line_item_quantity_is_rejected():
    invoice = extracted_invoice(
        "INV-1",
        "Acme",
        "A-1",
        "100",
        line_items=[
            {
                "item_code": {"value": "SKU-001"},
                "description": {"value": "Widget A"},
                "quantity": {"value": "-2"},
            }
        ],
    )

    with pytest.raises(ValueError, match="non-negative"):
        AnomalyAgent().analyze([invoice])


@pytest.mark.parametrize(
    "history",
    [
        {"Acme": "10000"},
        {"Acme": {"INR": "10000"}},
        {" ": ["10000"]},
        {"Acme": ["NaN"]},
        {"Acme": ["-1"]},
    ],
)
def test_malformed_vendor_history_is_rejected(history):
    with pytest.raises(ValueError):
        AnomalyAgent().analyze([], history)


def test_does_not_apply_outlier_rule_with_insufficient_history():
    report = AnomalyAgent().analyze(
        [extracted_invoice("INV-1", "Acme", "A-1", "50000")],
        {"Acme": ["10000", "11000"]},
    )

    assert report.findings == []


def test_duplicate_detection_keeps_one_finding_per_invoice():
    report = AnomalyAgent().analyze(
        [
            extracted_invoice("INV-1", "Acme", "A-1", "100"),
            extracted_invoice("INV-2", "Acme", "A-1", "100"),
            extracted_invoice("INV-3", "Acme", "A-1", "100"),
        ]
    )

    assert len(report.findings) == 2
    assert [finding.duplicate_of for finding in report.findings] == [
        "INV-1",
        "INV-1",
    ]
