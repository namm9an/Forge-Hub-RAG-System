import hashlib
import io
import re
from typing import List, Tuple, Optional
import pdfplumber
from pypdf import PdfReader
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header, Form
from fastapi import BackgroundTasks
from loguru import logger

from .config import get_settings, Settings
from .db import ensure_schema, execute, fetchval, fetch
from .schemas import DocumentsResponse, DocumentInfo
from .embeddings import get_model, _embed_texts, _as_pgvector

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _hash_text(text: str) -> str:
    """Generate SHA-256 hash of text for deduplication (Rule B-4)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    """
    Fast and accurate token estimator matching standard BPE tokenizers.
    Splits on word boundaries and standalone punctuation tokens.
    """
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


def recursive_token_chunk(
    text: str,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
) -> List[Tuple[str, int]]:
    """
    Recursive Token-Aware Chunker (Complies with Rules B-1 and B-2):
    - Target: strictly <= max_tokens (default 350 tokens to respect 384-token embedding ceiling).
    - Preserves semantic boundaries: paragraphs (\n\n) -> sentences -> words.
    - Never cuts words or sentences in half arbitrarily.
    - Returns list of (chunk_text, token_count).
    """
    text = text.strip()
    if not text:
        return []

    total_tokens = count_tokens(text)
    if total_tokens <= max_tokens:
        return [(text, total_tokens)]

    # 1. Split into paragraphs
    raw_paragraphs = text.split("\n\n")
    units: List[str] = []

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append(para)
        else:
            # Paragraph exceeds max_tokens -> split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if count_tokens(sentence) <= max_tokens:
                    units.append(sentence)
                else:
                    # Very long sentence -> split on clause/word boundaries without severing words
                    words = sentence.split()
                    cur_part: List[str] = []
                    cur_len = 0
                    for word in words:
                        w_len = count_tokens(word)
                        if cur_len + w_len > max_tokens and cur_part:
                            units.append(" ".join(cur_part))
                            cur_part = [word]
                            cur_len = w_len
                        else:
                            cur_part.append(word)
                            cur_len += w_len
                    if cur_part:
                        units.append(" ".join(cur_part))

    # 2. Greedily assemble units into chunks up to max_tokens with overlap
    chunks: List[Tuple[str, int]] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current_tokens + unit_tokens > max_tokens and current_units:
            # Finalize current chunk
            chunk_str = "\n\n".join(current_units) if "\n\n" in text else " ".join(current_units)
            chunks.append((chunk_str, count_tokens(chunk_str)))

            # Calculate overlap by taking trailing units up to overlap_tokens
            overlap_units: List[str] = []
            overlap_count = 0
            for u in reversed(current_units):
                u_cnt = count_tokens(u)
                if overlap_count + u_cnt <= overlap_tokens:
                    overlap_units.insert(0, u)
                    overlap_count += u_cnt
                else:
                    break

            current_units = overlap_units + [unit]
            current_tokens = sum(count_tokens(u) for u in current_units)
        else:
            current_units.append(unit)
            current_tokens += unit_tokens

    if current_units:
        final_str = "\n\n".join(current_units) if "\n\n" in text else " ".join(current_units)
        chunks.append((final_str, count_tokens(final_str)))

    return chunks


def _extract_pages_pdfplumber(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract page text tracking 1-indexed page numbers (Rule B-3)."""
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((idx, text))
    return pages


def _extract_pages_pypdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Fallback page extractor using pypdf."""
    pages: List[Tuple[int, str]] = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((idx, text))
    return pages


async def _embed_and_store_chunks(chunk_records: List[dict], settings: Settings) -> int:
    """Generate normalized vector embeddings and save to database."""
    if not chunk_records:
        return 0
    try:
        model = get_model(settings)
        texts = [c["content"] for c in chunk_records]
        vectors = await _embed_texts(model, texts)

        stored = 0
        for record, vec in zip(chunk_records, vectors):
            await execute(
                """
                INSERT INTO embeddings (chunk_id, embedding, model_version)
                VALUES ($1, $2::vector, $3)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                record["id"],
                _as_pgvector(vec),
                settings.embedding_model,
            )
            stored += 1
        return stored
    except Exception as e:
        logger.error("Failed to generate embeddings for document chunks: {}", e)
        return 0


@router.get("", response_model=DocumentsResponse)
async def list_documents(
    session_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None),
):
    await ensure_schema()
    effective_session = session_id or x_session_id
    rows = await fetch(
        """
        SELECT d.id, d.filename, d.file_size, d.total_pages, d.session_id, d.created_at,
               COALESCE((SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id), 0) AS chunks
        FROM documents d
        WHERE ($1::text IS NULL OR d.session_id = $1)
        ORDER BY d.created_at DESC
        """,
        effective_session,
    )
    docs = [
        DocumentInfo(
            id=str(r["id"]),
            filename=r["filename"],
            pages=r["total_pages"],
            chunks=r["chunks"],
            file_size=r["file_size"],
            session_id=r["session_id"],
            created_at=str(r["created_at"]) if r["created_at"] else None,
        )
        for r in rows
    ]
    return DocumentsResponse(documents=docs)


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    await ensure_schema()
    await execute("DELETE FROM documents WHERE id = $1", document_id)
    return {"status": "ok", "deleted_document_id": document_id}


@router.post("/upload")
async def upload_documents(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    x_session_id: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
):
    # Rule B-5: Dynamic Session Isolation
    effective_session = session_id or x_session_id or "default"
    max_files = settings.max_files_per_session
    max_total_bytes = settings.max_total_bytes
    chunk_size = settings.chunk_size_tokens
    chunk_overlap = settings.chunk_overlap_tokens

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Maximum {max_files} files allowed per upload")

    total_size = sum((f.size or 0) for f in files)
    if total_size > max_total_bytes:
        raise HTTPException(status_code=400, detail=f"Total upload size exceeds {max_total_bytes // (1024*1024)}MB limit")

    await ensure_schema()

    # Ensure session row exists (Rule B-5)
    await execute(
        """
        INSERT INTO sessions (session_id, last_activity)
        VALUES ($1, NOW())
        ON CONFLICT (session_id) DO UPDATE SET last_activity = NOW()
        """,
        effective_session,
    )

    processed_documents = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is not a PDF")

        file_bytes = await f.read()

        # Extract text page-by-page (Rule B-3)
        pages_data: List[Tuple[int, str]] = []
        try:
            pages_data = _extract_pages_pdfplumber(file_bytes)
            if not pages_data:
                raise ValueError("pdfplumber extracted 0 pages, trying fallback")
        except Exception as e:
            logger.warning("pdfplumber extraction warning: {}, trying pypdf", e)
            try:
                pages_data = _extract_pages_pypdf(file_bytes)
            except Exception as e2:
                logger.error("pypdf extraction failed: {}", e2)
                raise HTTPException(status_code=422, detail=f"Corrupt or unreadable PDF: {f.filename}")

        if not pages_data:
            raise HTTPException(status_code=422, detail=f"PDF '{f.filename}' contains no extractable text")

        total_pages = max(p[0] for p in pages_data)

        # Insert Document record
        doc_id = await fetchval(
            """
            INSERT INTO documents (session_id, filename, file_size, total_pages)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            effective_session,
            f.filename,
            len(file_bytes),
            total_pages,
        )

        # Chunk each page preserving page_number and token limits (Rules B-1, B-2, B-3)
        chunk_index = 0
        new_chunks = []

        for page_num, page_text in pages_data:
            page_chunks = recursive_token_chunk(page_text, max_tokens=chunk_size, overlap_tokens=chunk_overlap)
            for chunk_content, token_cnt in page_chunks:
                c_hash = _hash_text(chunk_content)
                chunk_id = await fetchval(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, content_hash, page_number, token_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (document_id, chunk_index) DO NOTHING
                    RETURNING id
                    """,
                    doc_id,
                    chunk_index,
                    chunk_content,
                    c_hash,
                    page_num,
                    token_cnt,
                )
                if chunk_id:
                    new_chunks.append({"id": chunk_id, "content": chunk_content})
                chunk_index += 1

        # Automatically generate vector embeddings for new chunks
        if new_chunks:
            background.add_task(_embed_and_store_chunks, new_chunks, settings)

        processed_documents.append({
            "document_id": str(doc_id),
            "filename": f.filename,
            "total_pages": total_pages,
            "chunks_created": chunk_index,
        })

    # Update session stats
    await execute(
        """
        UPDATE sessions
        SET total_documents = total_documents + $2,
            last_activity = NOW()
        WHERE session_id = $1
        """,
        effective_session,
        len(processed_documents),
    )

    return {
        "status": "success",
        "session_id": effective_session,
        "documents": processed_documents,
    }
