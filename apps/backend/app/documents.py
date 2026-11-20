from bisect import bisect_right
import hashlib
import io
import math
import re
import unicodedata
from typing import List, Optional, Tuple
from uuid import UUID

import pdfplumber
from pypdf import PdfReader
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from loguru import logger

from .auth import Principal, require_user
from .config import get_settings, Settings
from .db import execute, fetch, fetchval, transaction
from .embeddings import _as_pgvector, _embed_texts, get_model, get_model_async, run_blocking
from .schemas import DocumentsResponse, DocumentInfo
from .sessions import authorize_session

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _tokenizer(tokenizer=None):
    if tokenizer is None:
        tokenizer = getattr(get_model(get_settings()), "tokenizer", None)
    if tokenizer is None:
        raise ValueError("The embedding model must expose its tokenizer")
    return tokenizer


def count_tokens(text: str, tokenizer=None) -> int:
    """Count the actual model encoding, including its special tokens."""
    if not text:
        return 0
    return len(_tokenizer(tokenizer).encode(text, add_special_tokens=True, truncation=False))


def recursive_token_chunk(text: str, max_tokens: int = 350, overlap_tokens: int = 50, tokenizer=None) -> List[Tuple[str, int]]:
    """Use tokenizer offsets so oversized subwords can split without discarding text."""
    if not text or not text.strip():
        return []
    tokenizer = _tokenizer(tokenizer)
    max_tokens = min(max_tokens, int(getattr(tokenizer, "model_max_length", max_tokens)))
    special = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_tokens - special
    if capacity < 1 or overlap_tokens < 0:
        raise ValueError("Token budget must admit content and overlap must be nonnegative")
    if count_tokens(text, tokenizer) <= max_tokens:
        return [(text, count_tokens(text, tokenizer))]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = list(encoded["offset_mapping"])
    if not offsets or any(not (0 <= start < end <= len(text)) for start, end in offsets):
        raise ValueError("A fast tokenizer with source offsets is required")
    ends = [end for _, end in offsets]
    chunks = []
    start_index, start_char = 0, 0
    while start_index < len(offsets):
        end_index = min(start_index + capacity, len(offsets))
        end_char = offsets[end_index][0] if end_index < len(offsets) else len(text)
        if end_index < len(offsets):
            segment = text[start_char:end_char]
            boundaries = list(re.finditer(r"\n\s*\n|(?<=[.!?])\s+", segment))
            if not boundaries:
                boundaries = list(re.finditer(r"\s+", segment))
            if boundaries and segment[:boundaries[-1].end()].strip():
                end_char = start_char + boundaries[-1].end()
        # Re-encoding a subword boundary can change its tokenization; verify every result.
        while end_char > start_char and count_tokens(text[start_char:end_char], tokenizer) > max_tokens:
            end_index -= 1
            end_char = min(end_char, offsets[end_index][0])
        if end_char <= start_char:
            raise ValueError("The token budget cannot represent the next source character")
        content = text[start_char:end_char]
        chunks.append((content, count_tokens(content, tokenizer)))
        if end_char == len(text):
            break
        consumed = bisect_right(ends, end_char)
        next_index = max(start_index + 1, consumed - min(overlap_tokens, capacity - 1))
        if overlap_tokens == 0:
            next_index, next_char = consumed, end_char
        else:
            next_char = offsets[next_index][0] if next_index < len(offsets) else end_char
            while next_char < end_char and next_char > 0 and not text[next_char - 1].isspace() and not text[next_char].isspace():
                next_char += 1
            next_index = max(next_index, bisect_right(ends, next_char))
        if next_char <= start_char:
            next_index, next_char = consumed, end_char
        start_index, start_char = next_index, next_char
    return chunks


def _full_ruled_table(table):
    left, top, right, bottom = table.bbox
    cell_area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in table.cells)
    return (len(table.rows) >= 2 and len(table.columns) >= 2
            and math.isclose(cell_area, (right - left) * (bottom - top)))


def _extract_page_text(page):
    # shortcut: handle clearly ruled tables with bounded geometry; review
    # layout detection separately for borderless or drawing-heavy pages.
    if not 4 <= len(page.edges) <= 256:
        return page.extract_text(use_text_flow=True) or ""
    tables = [table for table in page.find_tables() if _full_ruled_table(table)]
    if not tables:
        return page.extract_text(use_text_flow=True) or ""

    def table_index(char):
        x = (char["x0"] + char["x1"]) / 2
        y = (char["top"] + char["bottom"]) / 2
        return next((index for index, (left, top, right, bottom) in enumerate(boxes)
                     if left <= x < right and top <= y < bottom), None)

    boxes = [table.bbox for table in tables]
    blocks, prose, emitted = [], [], set()
    for char in page.chars:
        index = table_index(char)
        if index is None:
            prose.append(char)
        elif index not in emitted:
            if prose:
                blocks.append(pdfplumber.utils.extract_text(prose, use_text_flow=True))
                prose.clear()
            for row in tables[index].extract():
                cells = [" ".join(cell.split()) if cell else "" for cell in row]
                while cells and not cells[-1]:
                    cells.pop()
                if cells:
                    blocks.append(" | ".join(cells))
            emitted.add(index)
    if prose:
        blocks.append(pdfplumber.utils.extract_text(prose, use_text_flow=True))
    return "\n".join(block for block in blocks if block)


def _extract_pdf(file_bytes: bytes, settings: Settings):
    max_pages = getattr(settings, "max_pages_per_document", 500)
    max_chars = getattr(settings, "max_extracted_chars", 2_000_000)

    def collect(pages, extract_text):
        if len(pages) > max_pages:
            raise HTTPException(status_code=413, detail="PDF page limit exceeded")
        extracted, total_chars = [], 0
        for number, page in enumerate(pages, start=1):
            text = extract_text(page) or ""
            total_chars += len(text)
            if total_chars > max_chars:
                raise HTTPException(status_code=413, detail="PDF extracted text limit exceeded")
            if text.strip():
                extracted.append((number, text))
        return extracted, len(pages)

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            result = collect(pdf.pages, _extract_page_text)
            if result[0]:
                return result
    except HTTPException:
        raise
    except Exception:
        logger.debug("Primary PDF extractor could not parse the document")
    try:
        result = collect(PdfReader(io.BytesIO(file_bytes)).pages, lambda page: page.extract_text())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Corrupt or unreadable PDF") from exc
    if not result[0]:
        raise HTTPException(status_code=422, detail="PDF contains no extractable text")
    return result


def _prepare_pdf(file_bytes, filename, settings, tokenizer, token_budget):
    pages, total_pages = _extract_pdf(file_bytes, settings)
    records, by_hash, raw_chunks = [], {}, 0
    for page_number, text in pages:
        chunks = recursive_token_chunk(text, max_tokens=token_budget, overlap_tokens=settings.chunk_overlap_tokens, tokenizer=tokenizer)
        raw_chunks += len(chunks)
        if raw_chunks > getattr(settings, "max_chunks_per_document", 10_000):
            raise HTTPException(status_code=413, detail="PDF chunk limit exceeded")
        for content, token_count in chunks:
            content_hash = _hash_text(content)
            if content_hash in by_hash:
                record = by_hash[content_hash]
                if page_number not in record["page_numbers"]:
                    record["page_numbers"].append(page_number)
                continue
            record = {"content": content, "content_hash": content_hash, "page_number": page_number,
                      "page_numbers": [page_number], "token_count": token_count, "chunk_index": len(records)}
            records.append(record)
            by_hash[content_hash] = record
    if not records:
        raise HTTPException(status_code=422, detail="PDF contains no usable text")
    return {"filename": filename, "file_size": len(file_bytes), "total_pages": total_pages, "chunks": records}


def _session_identifier(*identifiers):
    supplied = [value for value in identifiers if value is not None]
    if not supplied or any(not value.strip() for value in supplied):
        raise HTTPException(status_code=422, detail="An explicit session_id is required")
    if len(set(supplied)) != 1:
        raise HTTPException(status_code=400, detail="Conflicting session identifiers")
    return supplied[0]


def _validate_file_metadata(file):
    name = file.filename or ""
    if (not name.lower().endswith(".pdf") or len(name) > 255 or any(c in name for c in "/\\[]")
            or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in name)):
        raise HTTPException(status_code=400, detail="A safe PDF filename is required")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")


async def _read_file(file: UploadFile, remaining_bytes: int) -> bytes:
    if remaining_bytes < 0:
        raise HTTPException(status_code=413, detail="Upload byte limit exceeded")
    pieces, size = [], 0
    while True:
        piece = await file.read(min(65536, remaining_bytes - size + 1))
        if not piece:
            break
        pieces.append(piece)
        size += len(piece)
        if size > remaining_bytes:
            raise HTTPException(status_code=413, detail="Upload byte limit exceeded")
    content = b"".join(pieces)
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Invalid PDF signature")
    return content


@router.get("", response_model=DocumentsResponse)
async def list_documents(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None),
    principal: Principal = Depends(require_user),
):
    sid = _session_identifier(session_id, x_session_id)
    await authorize_session(sid, principal.user_id)
    rows = await fetch(
        """
        SELECT d.id, d.filename, d.file_size, d.total_pages, d.session_id, d.created_at,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunks
        FROM documents d JOIN sessions s ON s.session_id = d.session_id
        WHERE d.session_id = $1 AND s.user_id = $2 ORDER BY d.created_at DESC
        """, sid, principal.user_id,
    )
    return DocumentsResponse(documents=[
        DocumentInfo(id=str(row["id"]), filename=row["filename"], pages=row["total_pages"],
                     chunks=row["chunks"], file_size=row["file_size"], session_id=row["session_id"],
                     created_at=str(row["created_at"]) if row["created_at"] else None)
        for row in rows
    ])


@router.delete("/{document_id}")
async def delete_document(document_id: UUID, principal: Principal = Depends(require_user)):
    async with transaction():
        rows = await fetch(
            "SELECT d.session_id FROM documents d JOIN sessions s ON s.session_id = d.session_id "
            "WHERE d.id = $1 AND s.user_id = $2",
            document_id, principal.user_id,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Document not found")
        sid = rows[0]["session_id"]
        await authorize_session(sid, principal.user_id)
        deleted = await execute("DELETE FROM documents WHERE id = $1 AND session_id = $2", document_id, sid)
        if deleted != "DELETE 1":
            raise HTTPException(status_code=404, detail="Document not found")
        await execute("UPDATE sessions SET total_documents = GREATEST(total_documents - 1, 0) WHERE session_id = $1", sid)
    return {"status": "ok", "deleted_document_id": str(document_id)}


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id_form: Optional[str] = Form(None, alias="session_id"),
    session_id_query: Optional[str] = Query(None, alias="session_id"),
    x_session_id: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
    principal: Principal = Depends(require_user),
):
    sid = _session_identifier(session_id_query, session_id_form, x_session_id)
    if not files or len(files) > settings.max_files_per_session:
        raise HTTPException(status_code=400, detail="Upload file count limit exceeded")
    for file in files:
        _validate_file_metadata(file)
    async with transaction():
        await authorize_session(sid, principal.user_id, create=True)
        usage = (await fetch(
            "SELECT COUNT(*) AS count, COALESCE(SUM(file_size), 0) AS total_bytes FROM documents WHERE session_id = $1", sid
        ))[0]
        if usage["count"] + len(files) > settings.max_files_per_session or usage["total_bytes"] >= settings.max_total_bytes:
            raise HTTPException(status_code=413, detail="Session document quota exceeded")
        inputs, total_bytes = [], 0
        for file in files:
            content = await _read_file(file, settings.max_total_bytes - usage["total_bytes"] - total_bytes)
            total_bytes += len(content)
            inputs.append((file.filename, content))
        try:
            model = await get_model_async(settings)
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("Embedding tokenizer is unavailable")
            token_budget = min(settings.chunk_size_tokens, int(getattr(model, "max_seq_length", settings.chunk_size_tokens)))
            prepared = []
            for filename, content in inputs:
                prepared.append(await run_blocking(_prepare_pdf, content, filename, settings, tokenizer, token_budget))
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Document preparation failed: {}", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Document embedding preparation is unavailable") from exc

        results = []
        for document in prepared:
            doc_id = await fetchval(
                "INSERT INTO documents (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
                sid, document["filename"], document["file_size"], document["total_pages"],
            )
            for start in range(0, len(document["chunks"]), 16):
                batch = document["chunks"][start:start + 16]
                try:
                    vectors = await _embed_texts(model, [chunk["content"] for chunk in batch])
                except Exception as exc:
                    logger.warning("Document embedding failed: {}", type(exc).__name__)
                    raise HTTPException(status_code=503, detail="Document embedding preparation is unavailable") from exc
                for chunk, vector in zip(batch, vectors):
                    chunk_id = await fetchval(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content, content_hash, page_number, token_count, page_numbers)
                        VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id
                        """,
                        doc_id, chunk["chunk_index"], chunk["content"], chunk["content_hash"],
                        chunk["page_number"], chunk["token_count"], chunk["page_numbers"],
                    )
                    await execute(
                        "INSERT INTO embeddings (chunk_id, embedding, model_version) VALUES ($1, $2::vector, $3)",
                        chunk_id, _as_pgvector(vector), settings.embedding_model,
                    )
            results.append({"document_id": str(doc_id), "filename": document["filename"],
                            "total_pages": document["total_pages"], "chunks_created": len(document["chunks"])})
        await execute(
            "UPDATE sessions SET total_documents = total_documents + $2, last_activity = NOW() WHERE session_id = $1",
            sid, len(prepared),
        )
    return {"status": "success", "session_id": sid, "documents": results}
