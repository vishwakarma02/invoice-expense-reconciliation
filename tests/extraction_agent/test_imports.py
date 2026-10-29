from src.extraction_agent.extractor import extract_invoice, extract_receipt
from src.extraction_agent.ingestion_adapter import process_ingestion_file


def test_repo_imports():
    assert callable(extract_invoice)
    assert callable(extract_receipt)
    assert callable(process_ingestion_file)
