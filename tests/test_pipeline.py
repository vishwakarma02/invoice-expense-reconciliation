from types import SimpleNamespace

from src.extraction.schemas import ExtractionBatch
from src.ingestion.schemas import IngestionBatch
import run_pipeline


def test_pipeline_runs_ingestion_before_extraction_using_its_json(monkeypatch, tmp_path):
    ingestion_output = tmp_path / "ingestion_output.json"
    extraction_output = tmp_path / "extraction_output.json"
    matching_output = tmp_path / "matching_output.json"
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(run_pipeline, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(run_pipeline, "INGESTION_OUTPUT", ingestion_output)
    monkeypatch.setattr(run_pipeline, "EXTRACTION_OUTPUT", extraction_output)
    monkeypatch.setattr(run_pipeline, "MATCHING_OUTPUT", matching_output)

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
        )

    monkeypatch.setattr(run_pipeline, "run_ingestion", fake_ingestion)
    monkeypatch.setattr(run_pipeline, "run_extraction", fake_extraction)
    monkeypatch.setattr(run_pipeline, "run_matching", fake_matching)

    run_pipeline.run_pipeline()

    assert [event[0] for event in events] == ["ingestion", "extraction", "matching"]
    assert events[1][1] == (ingestion_output, extraction_output)
    assert events[2][1] == (extraction_output, matching_output)


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

    try:
        run_pipeline.run_pipeline()
    except FileNotFoundError as error:
        assert str(ingestion_output) in str(error)
    else:
        raise AssertionError("Pipeline should fail if ingestion output is absent.")

    assert not extraction_started
