import pytest
from scripts.benchmark_rag import threshold_status, validate_test_database


def test_failed_measurement_cannot_be_reported_as_pass():
    assert threshold_status(70, 85) == "FAIL"
    assert threshold_status(85, 85) == "PASS"
    assert threshold_status(100, 100, less_than=True) == "FAIL"
    assert threshold_status(None, 95) == "NOT MEASURED"


@pytest.mark.parametrize("dsn", ["", "postgresql://localhost/production", "postgresql://remote/forge_hub_test",
    "postgresql://localhost/forge_hub_test?options=unsafe", "postgresql://localhost/forge_hub_test#fragment"])
def test_benchmark_refuses_application_or_remote_database(dsn):
    with pytest.raises(ValueError):
        validate_test_database(dsn)


def test_benchmark_accepts_only_explicit_local_disposable_database():
    dsn = "postgresql://synthetic:synthetic@127.0.0.1:12345/forge_hub_test_run"
    assert validate_test_database(dsn) == dsn


@pytest.mark.asyncio
async def test_benchmark_ingestion_uses_validated_application_upload_contract():
    from scripts.benchmark_rag import ingest_benchmark_dataset, TECHNICAL_PAGES
    from app.config import get_settings
    from app import db
    document_id, count = await ingest_benchmark_dataset("benchmark-contract", get_settings())
    assert count > 0
    assert db._memory_store.documents[document_id]["total_pages"] == len(TECHNICAL_PAGES)
    assert len(db._memory_store.embeddings) == count
