from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi import BackgroundTasks
from typing import List
from loguru import logger
import pdfplumber
from PyPDF2 import PdfReader
import hashlib
import io
from .config import get_settings
from .db import ensure_schema, execute, fetchval, fetch
from .schemas import DocumentsResponse, DocumentInfo

router = APIRouter(prefix="/api/documents", tags=["documents"]) 


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def _extract_text_pdfplumber(file_bytes: bytes) -> tuple[str, int]:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text), len(pdf.pages)


def _extract_text_pypdf2(file_bytes: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text), len(reader.pages)


@router.get("", response_model=DocumentsResponse)
async def list_documents(session_id: str | None = None):
    await ensure_schema()
    rows = await fetch(
        """
        SELECT d.id, d.filename, d.file_size, d.total_pages,
               COALESCE((SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id), 0) AS chunks
        FROM documents d
        WHERE ($1::text IS NULL OR d.session_id = $1)
        ORDER BY d.created_at DESC
        """,
        session_id,
    )
    docs = [
        DocumentInfo(
            id=str(r["id"]),
            filename=r["filename"],
            pages=r["total_pages"],
            chunks=r["chunks"],
            file_size=r["file_size"],
        )
        for r in rows
    ]
    return DocumentsResponse(documents=docs)


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    await ensure_schema()
    # delete cascades to chunks and embeddings per FK
    await execute("DELETE FROM documents WHERE id = $1", document_id)
    return {"status": "ok"}


@router.post("/upload")
async def upload_documents(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    settings=Depends(get_settings),
):
    # Limits from config (avoid hard-coding)
    max_files = int(getattr(settings, "MAX_FILES_PER_SESSION", 3))
    max_total_bytes = int(getattr(settings, "MAX_TOTAL_BYTES", 50 * 1024 * 1024))
    chunk_size = int(getattr(settings, "CHUNK_SIZE", 1000))
    overlap = int(getattr(settings, "CHUNK_OVERLAP", 200))

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Maximum {max_files} files allowed")

    total_size = 0
    for f in files:
        if f.content_type not in ("application/pdf",):
            raise HTTPException(status_code=400, detail="Only PDF files are supported for now")
        total_size += f.size or 0
    if total_size > max_total_bytes:
        raise HTTPException(status_code=400, detail="Total upload size exceeds limit")

    await ensure_schema()

    stored_docs = []
    for f in files:
        file_bytes = await f.read()

        # Extract text using preferred pipeline with fallback
        text = ""
        pages = 0
        try:
            text, pages = _extract_text_pdfplumber(file_bytes)
            if not text.strip():
                raise ValueError("Empty text via pdfplumber, fallback")
        except Exception as e:
            logger.warning("pdfplumber failed: {}", e)
            try:
                text, pages = _extract_text_pypdf2(file_bytes)
            except Exception as e2:
                logger.error("PyPDF2 failed: {}", e2)
                raise HTTPException(status_code=422, detail="Failed to process PDF. It may be corrupted.")

        # Store document row
        doc_id = await fetchval(
            """
            INSERT INTO documents (session_id, filename, file_size, total_pages) 
            VALUES ($1, $2, $3, $4) RETURNING id
            """,
            "anonymous",  # TODO: replace when session handling is wired
            f.filename,
            len(file_bytes),
            pages,
        )

        # Chunk and store chunks
        chunks = _split_chunks(text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            await execute(
                """
                INSERT INTO chunks (document_id, chunk_index, content, content_hash)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                """,
                doc_id,
                idx,
                chunk,
                _hash_text(chunk),
            )

        stored_docs.append({
            "id": str(doc_id),
            "filename": f.filename,
            "pages": pages,
            "chunks": len(chunks)
        })

    return {"status": "ok", "documents": stored_docs}
