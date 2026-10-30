from types import SimpleNamespace

from src.extraction.schemas import ExtractionBatch
from src.ingestion.schemas import IngestionBatch
import run_pipeline


def test_pipeline_runs_ingestion_before_extraction_using_its_json(monkeypatch, tmp_path):
    ingestion_output = tmp_path / "ingestion_output.json"
    extraction_output = tmp_path / "extraction_output.json"
    matching_output = tmp_path / "matching_output.json"
    anomaly_output = tmp_path / "anomaly_output.json"
    explanation_output = tmp_path / "explanation_output.json"
    hitl_database = tmp_path / "hitl.sqlite3"
    explanation_output = tmp_path / "explanation_output.json"
    events: list[tuple[str, object]] = []
    anomaly_report = SimpleNamespace(findings=[])

    monkeypatch.setattr(run_pipeline, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(run_pipeline, "INGESTION_OUTPUT", ingestion_output)
    monkeypatch.setattr(run_pipeline, "EXTRACTION_OUTPUT", extraction_output)
    monkeypatch.setattr(run_pipeline, "MATCHING_OUTPUT", matching_output)
    monkeypatch.setattr(run_pipeline, "ANOMALY_OUTPUT", anomaly_output)
    monkeypatch.setattr(run_pipeline, "EXPLANATION_OUTPUT", explanation_output)
    monkeypatch.setattr(run_pipeline, "HITL_DATABASE", hitl_database)
    monkeypatch.setattr(run_pipeline, "EXPLANATION_OUTPUT", explanation_output)

    def fake_ingestion(input_dir, output_file):
        events.append(("ingestion", (input_dir, output_file)))
        output_file.write_text("{}", encoding="utf-8")
        return IngestionBatch(batch_id="batch-1")

    def fake_extraction(input_file, output_file):
        events.append(("extraction", (input_file, output_file)))
        return ExtractionBatch(
            extraction_id="extraction-1",
            ingestion_batch_id="batch-1",
        )

    def fake_matching(input_file, output_file):
        events.append(("matching", (input_file, output_file)))
        return SimpleNamespace(
            invoices=[],
            bank_payments=[],
            extraction_id="extraction-1",
            matching_id="matching-1",
        )

    def fake_anomaly(input_file, output_file):
        events.append(("anomaly", (input_file, output_file)))
        return anomaly_report

    def fake_explanation(input_file, output_file, anomalies):
        events.append(("explanation", (input_file, output_file, anomalies)))
        return SimpleNamespace(explanation_count=0)

    class FakeHITLRouter:
        def __init__(self, database_path):
            assert database_path == hitl_database

        def route(self, matching, explanations, anomalies, extraction):
            events.append(("hitl", (matching, explanations, anomalies, extraction)))
            return []

    def fake_explanation(input_file, output_file, anomalies):
        events.append(("explanation", (input_file, output_file, anomalies)))
        return SimpleNamespace(explanation_count=0)

    monkeypatch.setattr(run_pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(run_pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(run_pipeline, "run_matching", fake_matching)
    monkeypatch.setattr(run_pipeline, "run_anomaly", fake_anomaly)
    monkeypatch.setattr(run_pipeline, "run_explanation", fake_explanation)
    monkeypatch.setattr(run_pipeline, "HITLRouter", FakeHITLRouter)
    monkeypatch.setattr(run_pipeline, "run_explanation", fake_explanation)

    run_pipeline.run_pipeline()

    assert [event[0] for event in events] == [
        "ingestion",
        "extraction",
        "matching",
        "anomaly",
        "explanation",
        "hitl",
    ]
    assert events[1][1] == (ingestion_output, extraction_output)
    assert events[2][1] == (extraction_output, matching_output)
    assert events[3][1] == (extraction_output, anomaly_output)
    assert events[4][1] == (
        matching_output,
        explanation_output,
        anomaly_report,
    )


def test_pipeline_does_not_start_extraction_without_ingestion_output(
    monkeypatch, tmp_path
):
    ingestion_output = tmp_path / "missing.json"
    extraction_started = False

    monkeypatch.setattr(run_pipeline, "INGESTION_OUTPUT", ingestion_output)
    monkeypatch.setattr(
        run_pipeline,
        "run_ingestion",
        lambda *_: IngestionBatch(batch_id="batch-1"),
    )

    def fake_extraction(*_):
        nonlocal extraction_started
        extraction_started = True
        return SimpleNamespace()

    monkeypatch.setattr(run_pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(run_pipeline, "run_matching", lambda *_: SimpleNamespace())
    monkeypatch.setattr(run_pipeline, "run_explanation", lambda *_: SimpleNamespace())

    try:
        run_pipeline.run_pipeline()
    except FileNotFoundError as error:
        assert str(ingestion_output) in str(error)
    else:
        raise AssertionError("Pipeline should fail if ingestion output is absent.")

    assert not extraction_started
