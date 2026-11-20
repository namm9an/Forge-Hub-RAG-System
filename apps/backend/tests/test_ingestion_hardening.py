"""Ingestion/storage contracts with synthetic PDFs and explicit fake tokenizers/models."""

import ast
import asyncio
from contextlib import nullcontext
import io
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import List

import httpx
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile


class CharacterTokenizer:
    """Character offsets are a test double, not a semantic/model tokenizer evaluation."""

    model_max_length = 384

    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(range(len(text) + (2 if add_special_tokens else 0)))

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text, add_special_tokens=False),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_token_count_includes_model_special_tokens():
    from app.documents import count_tokens

    assert count_tokens("abcd", tokenizer=CharacterTokenizer()) == 6


def test_oversized_word_is_split_without_losing_text():
    from app.documents import recursive_token_chunk

    text = "x-" * 400
    chunks = recursive_token_chunk(text, max_tokens=64, overlap_tokens=0, tokenizer=CharacterTokenizer())
    assert "".join(content for content, _ in chunks) == text
    assert all(tokens <= 64 for _, tokens in chunks)


def test_overlap_cannot_exceed_the_model_budget():
    from app.documents import recursive_token_chunk

    text = "lead " * 40 + "\n\n" + "target " * 350
    chunks = recursive_token_chunk(text, max_tokens=350, overlap_tokens=50, tokenizer=CharacterTokenizer())
    assert len(chunks) > 1
    assert all(tokens <= 350 for _, tokens in chunks)
    assert all(content in text for content, _ in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [np.full((1, 768), np.nan), np.ones((1, 2)), np.zeros((1, 768)), np.full((1, 768), 2.0)])
async def test_invalid_embedding_vectors_are_rejected(vectors):
    from app.embeddings import _embed_texts

    class InvalidEncoder:
        def encode(self, texts, **kwargs):
            return vectors

    with pytest.raises(ValueError):
        await _embed_texts(InvalidEncoder(), ["synthetic"])


@pytest.mark.asyncio
async def test_blocking_model_work_runs_off_the_event_loop():
    from app import embeddings

    event_thread = threading.get_ident()
    assert await embeddings.run_blocking(threading.get_ident) != event_thread


@pytest.mark.asyncio
async def test_cancelled_request_keeps_worker_slot_until_work_finishes(monkeypatch):
    from app import embeddings

    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(embedding_workers=1))
    entered, release, second_entered = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def first_work():
        loop.call_soon_threadsafe(entered.set)
        release.wait(timeout=2)

    first = asyncio.create_task(embeddings.run_blocking(first_work))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(embeddings.run_blocking(second_entered.set))
        await asyncio.sleep(0.02)
        assert not second_entered.is_set()
    finally:
        release.set()
        if second is not None:
            await asyncio.wait_for(second, timeout=1)
        if not first.done():
            await asyncio.wait_for(first, timeout=1)


@pytest.fixture
def pdf_bytes():
    source = Path(__file__).resolve().parents[3] / "scripts/benchmark_rag.py"
    definitions = [n for n in ast.parse(source.read_text(encoding="utf-8")).body
                   if isinstance(n, ast.FunctionDef) and n.name in {"_escape_pdf_text", "generate_benchmark_pdf"}]
    namespace = {"List": List}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["generate_benchmark_pdf"](["Synthetic document evidence for isolated ingestion contracts."])


@pytest.fixture
def authenticated_app():
    from app import app
    from app.auth import Principal, require_user

    app.dependency_overrides[require_user] = lambda: Principal(user_id="owner-a")
    return app


async def upload(client, pdf_bytes, session="ingestion-session", **kwargs):
    return await client.post("/api/documents/upload", params={"session_id": session},
                             files=[("files", ("synthetic.pdf", pdf_bytes, "application/pdf"))], **kwargs)


@pytest.mark.asyncio
async def test_query_session_upload_is_visible_and_ready(authenticated_app, pdf_bytes):
    from app import db

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        response = await upload(client, pdf_bytes)
        assert response.status_code == 200, response.text
        assert response.json()["session_id"] == "ingestion-session"
        listing = await client.get("/api/documents", params={"session_id": "ingestion-session"})
        assert len(listing.json()["documents"]) == 1
    assert db._memory_store.chunks
    assert set(db._memory_store.embeddings) == set(db._memory_store.chunks)


@pytest.mark.asyncio
async def test_missing_or_conflicting_upload_session_is_rejected(authenticated_app, pdf_bytes):
    from app import db

    files = [("files", ("synthetic.pdf", pdf_bytes, "application/pdf"))]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        missing = await client.post("/api/documents/upload", files=files)
        assert missing.status_code == 422
        conflict = await client.post("/api/documents/upload?session_id=one", data={"session_id": "two"}, files=files)
        assert conflict.status_code in {400, 422}
    assert not db._memory_store.documents
    assert not db._memory_store.sessions


@pytest.mark.asyncio
async def test_invalid_later_file_does_not_leave_partial_rows(authenticated_app, pdf_bytes):
    from app import db

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        response = await client.post("/api/documents/upload?session_id=partial", files=[
            ("files", ("valid.pdf", pdf_bytes, "application/pdf")),
            ("files", ("invalid.txt", b"invalid", "text/plain")),
        ])
        assert response.status_code in {400, 422}
    assert not db._memory_store.documents
    assert not db._memory_store.chunks
    assert not db._memory_store.embeddings


@pytest.mark.asyncio
async def test_embedding_failure_is_visible_and_stores_nothing(authenticated_app, pdf_bytes, monkeypatch):
    from app import db, documents

    async def failed_embedding(*args, **kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(documents, "_embed_texts", failed_embedding)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        response = await upload(client, pdf_bytes)
        assert response.status_code == 503
    assert not db._memory_store.documents
    assert not db._memory_store.chunks
    assert not db._memory_store.embeddings


@pytest.mark.asyncio
async def test_later_embedding_batch_failure_rolls_back_earlier_batches(authenticated_app, pdf_bytes, monkeypatch):
    from app import db, documents

    chunks = [{"content": f"synthetic evidence {i}", "content_hash": f"hash-{i}", "page_number": 1,
               "page_numbers": [1], "token_count": 4, "chunk_index": i} for i in range(17)]
    monkeypatch.setattr(documents, "_prepare_pdf", lambda *args: {
        "filename": "synthetic.pdf", "file_size": len(pdf_bytes), "total_pages": 1, "chunks": chunks,
    })
    original, calls = documents._embed_texts, []

    async def fail_second(model, texts, **kwargs):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("synthetic second batch failure")
        return await original(model, texts, **kwargs)

    monkeypatch.setattr(documents, "_embed_texts", fail_second)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        response = await upload(client, pdf_bytes)
        assert response.status_code == 503
    assert calls == [16, 1]
    assert not db._memory_store.sessions
    assert not db._memory_store.documents
    assert not db._memory_store.chunks
    assert not db._memory_store.embeddings


@pytest.mark.asyncio
async def test_persistence_failure_rolls_back_the_entire_batch(authenticated_app, pdf_bytes, monkeypatch):
    from app import db, documents

    original = documents.execute

    async def failed_insert(statement, *args):
        if "INSERT INTO embeddings" in statement:
            raise RuntimeError("synthetic database failure")
        return await original(statement, *args)

    monkeypatch.setattr(documents, "execute", failed_insert)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app, raise_app_exceptions=False), base_url="http://test") as client:
        response = await upload(client, pdf_bytes)
        assert response.status_code >= 500
    assert not db._memory_store.documents
    assert not db._memory_store.chunks
    assert not db._memory_store.embeddings


@pytest.mark.asyncio
async def test_cumulative_document_quota_is_race_safe(authenticated_app, pdf_bytes):
    from app.config import get_settings

    settings = get_settings().model_copy(update={"max_files_per_session": 1})
    authenticated_app.dependency_overrides[get_settings] = lambda: settings
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        first, second = await asyncio.gather(upload(client, pdf_bytes), upload(client, pdf_bytes))
    assert sorted([first.status_code, second.status_code]) == [200, 413]


def test_duplicate_page_content_preserves_all_page_attribution(monkeypatch):
    from app import documents
    from app.config import get_settings

    monkeypatch.setattr(documents, "_extract_pdf", lambda *args: ([(1, "identical evidence"), (2, "identical evidence")], 2))
    prepared = documents._prepare_pdf(b"%PDF-synthetic", "repeated.pdf", get_settings(), CharacterTokenizer(), 350)
    assert len(prepared["chunks"]) == 1
    assert prepared["chunks"][0]["page_number"] == 1
    assert prepared["chunks"][0]["page_numbers"] == [1, 2]


@pytest.mark.asyncio
async def test_actual_read_limit_does_not_trust_upload_size_metadata(pdf_bytes):
    from app.documents import _read_file

    file = UploadFile(filename="synthetic.pdf", file=io.BytesIO(pdf_bytes), size=0)
    with pytest.raises(HTTPException) as error:
        await _read_file(file, remaining_bytes=32)
    assert error.value.status_code == 413


@pytest.mark.parametrize("filename", ["../escape.pdf", "broken[Doc].pdf", "newline\n.pdf"])
def test_unsafe_filename_is_rejected(filename):
    from app.documents import _validate_file_metadata

    with pytest.raises(HTTPException) as error:
        _validate_file_metadata(SimpleNamespace(filename=filename, content_type="application/pdf"))
    assert error.value.status_code == 400


@pytest.mark.parametrize("limit", ["pages", "characters"])
def test_pdf_resource_limits_stop_processing(monkeypatch, limit):
    from app import documents
    from app.config import get_settings

    settings = get_settings().model_copy(update={"max_pages_per_document": 1, "max_extracted_chars": 3})
    page = SimpleNamespace(edges=[], extract_text=lambda **options: "more than three characters")
    pages = [page, page] if limit == "pages" else [page]
    monkeypatch.setattr(documents.pdfplumber, "open", lambda *args: nullcontext(SimpleNamespace(pages=pages)))
    with pytest.raises(HTTPException) as error:
        documents._extract_pdf(b"%PDF-synthetic", settings)
    assert error.value.status_code == 413


def test_two_column_pdf_keeps_continuation_sentences_together(pdf_bytes):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject
    from app.config import get_settings
    from app.documents import _extract_pdf

    writer = PdfWriter(clone_from=io.BytesIO(pdf_bytes))
    content = DecodedStreamObject()
    content.set_data(b"""BT /F1 10 Tf
1 0 0 1 50 720 Tm (Audit logs must be kept) Tj
1 0 0 1 50 706 Tm (for thirty-five days.) Tj
1 0 0 1 350 720 Tm (Backups must be mirrored) Tj
1 0 0 1 350 706 Tm (at the secondary site.) Tj
ET""")
    writer.pages[0].replace_contents(content)
    output = io.BytesIO()
    writer.write(output)
    pages, _ = _extract_pdf(output.getvalue(), get_settings())
    text = " ".join(pages[0][1].split())
    assert "Audit logs must be kept for thirty-five days." in text
    assert "Backups must be mirrored at the secondary site." in text


def test_ruled_pdf_table_keeps_wrapped_cells_with_their_rows(pdf_bytes):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject
    from app.config import get_settings
    from app.documents import _extract_pdf

    writer = PdfWriter(clone_from=io.BytesIO(pdf_bytes))
    content = DecodedStreamObject()
    content.set_data(b"""0.5 w
40 630 m 550 630 l 550 750 l 40 750 l h S
210 630 m 210 720 l S 410 630 m 410 720 l S
40 720 m 550 720 l S 40 690 m 550 690 l S 40 660 m 550 660 l S
BT /F1 11 Tf
1 0 0 1 50 770 Tm (Apply these rules after recovery.) Tj
1 0 0 1 220 733 Tm (Recovery Operations) Tj
1 0 0 1 50 703 Tm (Environment) Tj
1 0 0 1 50 677 Tm (Staging) Tj
1 0 0 1 50 646 Tm (Production) Tj
1 0 0 1 220 703 Tm (Recovery Window) Tj
1 0 0 1 220 677 Tm (15 minutes) Tj
1 0 0 1 220 646 Tm (8 hours) Tj
1 0 0 1 420 703 Tm (Required Action) Tj
1 0 0 1 420 677 Tm (Run tabletop) Tj
1 0 0 1 420 667 Tm (exercise) Tj
1 0 0 1 420 646 Tm (Verify checksum) Tj
1 0 0 1 50 600 Tm (Log the outcome.) Tj
ET""")
    writer.pages[0].replace_contents(content)
    output = io.BytesIO()
    writer.write(output)

    pages, page_count = _extract_pdf(output.getvalue(), get_settings())

    assert page_count == len(pages) == 1
    page_number, text = pages[0]
    assert page_number == 1
    lines = text.strip().splitlines()
    assert lines[0] == "Apply these rules after recovery."
    assert lines[-1] == "Log the outcome."
    assert text.count("Recovery Operations") == 1
    assert "Environment | Recovery Window | Required Action" in lines
    assert "Staging | 15 minutes | Run tabletop exercise" in lines
    assert "Production | 8 hours | Verify checksum" in lines


def test_ragged_table_does_not_erase_prose_inside_its_outer_bounds(pdf_bytes):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject
    from app.config import get_settings
    from app.documents import _extract_pdf

    writer = PdfWriter(clone_from=io.BytesIO(pdf_bytes))
    content = DecodedStreamObject()
    content.set_data(b"""0.5 w
40 650 m 300 650 l 300 700 l 550 700 l 550 750 l 40 750 l h S
40 700 m 300 700 l S 300 700 m 300 750 l S
BT /F1 11 Tf
1 0 0 1 50 725 Tm (Primary label) Tj
1 0 0 1 320 725 Tm (Primary value) Tj
1 0 0 1 50 675 Tm (Secondary label) Tj
1 0 0 1 320 675 Tm (Outside table label) Tj
ET""")
    writer.pages[0].replace_contents(content)
    output = io.BytesIO()
    writer.write(output)

    pages, _ = _extract_pdf(output.getvalue(), get_settings())
    assert pages[0][1].count("Outside table label") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty-text", "truncated", "non-pdf"])
async def test_rejected_document_formats_leave_no_session_or_content(authenticated_app, pdf_bytes, kind):
    from app import db
    from scripts.benchmark_rag import generate_benchmark_pdf

    content = {
        "empty-text": generate_benchmark_pdf([""]),
        "truncated": pdf_bytes[:40],
        "non-pdf": b"<document>ordinary non-PDF text</document>",
    }[kind]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        response = await client.post("/api/documents/upload", params={"session_id": "invalid-pdf"},
                                     files={"files": ("invalid.pdf", content, "application/pdf")})
    assert response.status_code == 422
    assert "invalid-pdf" not in db._memory_store.sessions
    assert not db._memory_store.documents


@pytest.mark.asyncio
async def test_embedding_backfill_cannot_process_another_owner(authenticated_app, pdf_bytes):
    from app import db
    from app.auth import Principal, require_user

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        owned = await upload(client, pdf_bytes, session="session-a")
        assert owned.status_code == 200, owned.text
        authenticated_app.dependency_overrides[require_user] = lambda: Principal(user_id="owner-b")
        other = await upload(client, pdf_bytes, session="session-b")
        assert other.status_code == 200, other.text
        other_doc = other.json()["documents"][0]["document_id"]
        missing = [cid for cid, row in db._memory_store.chunks.items() if row["document_id"] == other_doc]
        for cid in missing:
            del db._memory_store.embeddings[cid]
        authenticated_app.dependency_overrides[require_user] = lambda: Principal(user_id="owner-a")
        denied = await client.post("/api/embeddings/generate", params={"session_id": "session-b"})
        assert denied.status_code in {403, 404}
        own_backfill = await client.post("/api/embeddings/generate", params={"session_id": "session-a"})
        assert own_backfill.status_code == 200, own_backfill.text
    assert all(cid not in db._memory_store.embeddings for cid in missing)


@pytest.mark.asyncio
async def test_missing_authentication_is_rejected_before_model_work(pdf_bytes, monkeypatch):
    from app import app, documents

    attempted = []

    async def unexpected_model(*args):
        attempted.append(True)
        raise AssertionError("Authentication must precede model work")

    monkeypatch.setattr(documents, "get_model_async", unexpected_model)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        denied = await upload(client, pdf_bytes)
        assert denied.status_code == 401
    assert not attempted


@pytest.mark.asyncio
async def test_other_owner_is_rejected_before_model_work(authenticated_app, pdf_bytes, monkeypatch):
    from app import documents
    from app.auth import Principal, require_user
    from app.sessions import authorize_session

    await authorize_session("private-session", "owner-b", create=True)
    attempted = []

    async def unexpected_model(*args):
        attempted.append(True)
        raise AssertionError("Ownership must precede model work")

    monkeypatch.setattr(documents, "get_model_async", unexpected_model)
    authenticated_app.dependency_overrides[require_user] = lambda: Principal(user_id="owner-a")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=authenticated_app), base_url="http://test") as client:
        denied = await upload(client, pdf_bytes, session="private-session")
        assert denied.status_code == 404
    assert not attempted
