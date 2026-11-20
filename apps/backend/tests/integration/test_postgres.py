"""Real PostgreSQL semantics for both actual DDL sources; never memory substitutes."""

import json
import uuid

import asyncpg
import pytest

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


async def seed(connection):
    sid = "synthetic-" + uuid.uuid4().hex
    await connection.execute("INSERT INTO sessions(session_id) VALUES($1)", sid)
    doc = await connection.fetchval(
        "INSERT INTO documents(session_id, filename, file_size) VALUES($1, 'synthetic.pdf', 100) RETURNING id", sid
    )
    chunk = await connection.fetchval(
        "INSERT INTO chunks(document_id, chunk_index, content, content_hash) "
        "VALUES($1, 0, 'PostgreSQL durability uses write ahead logging.', 'synthetic-hash') RETURNING id", doc
    )
    vector = "[1," + ",".join(["0"] * 767) + "]"
    await connection.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES($1, $2::vector)", chunk, vector)
    await connection.execute(
        "INSERT INTO chat_history(session_id, user_message, assistant_message, processing_time_ms) "
        "VALUES($1, 'synthetic question', 'synthetic answer', 1)", sid
    )
    return sid, doc, chunk, vector


async def test_actual_foreign_keys_reject_orphans(postgres_connection):
    connection = postgres_connection
    cases = [
        ("INSERT INTO documents(session_id, filename, file_size) VALUES($1, 'orphan.pdf', 1)", "missing-session"),
        ("INSERT INTO chunks(document_id, chunk_index, content, content_hash) VALUES($1, 0, 'orphan', 'hash')", uuid.uuid4()),
        ("INSERT INTO embeddings(chunk_id, embedding) VALUES($1, array_fill(0.1::real, ARRAY[768])::vector)", uuid.uuid4()),
        ("INSERT INTO chat_history(session_id, user_message, assistant_message, processing_time_ms) VALUES($1, 'q', 'a', 1)", "missing-session"),
    ]
    for statement, missing_parent in cases:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with connection.transaction():
                await connection.execute(statement, missing_parent)


async def test_actual_unique_constraints_reject_duplicates(postgres_connection):
    connection = postgres_connection
    sid, doc, chunk, vector = await seed(connection)
    cases = [
        ("INSERT INTO sessions(session_id) VALUES($1)", (sid,)),
        ("INSERT INTO chunks(document_id, chunk_index, content, content_hash) VALUES($1, 0, 'duplicate', 'other')", (doc,)),
        ("INSERT INTO embeddings(chunk_id, embedding) VALUES($1, $2::vector)", (chunk, vector)),
    ]
    for statement, args in cases:
        with pytest.raises(asyncpg.UniqueViolationError):
            async with connection.transaction():
                await connection.execute(statement, *args)


@pytest.mark.parametrize("delete_parent", ["document", "session"])
async def test_actual_cascades_remove_dependents(postgres_connection, delete_parent):
    connection = postgres_connection
    sid, doc, chunk, _ = await seed(connection)
    if delete_parent == "document":
        await connection.execute("DELETE FROM documents WHERE id = $1", doc)
    else:
        await connection.execute("DELETE FROM sessions WHERE session_id = $1", sid)
    assert await connection.fetchval("SELECT COUNT(*) FROM documents") == 0
    assert await connection.fetchval("SELECT COUNT(*) FROM chunks") == 0
    assert await connection.fetchval("SELECT COUNT(*) FROM embeddings") == 0
    expected_remaining = 1 if delete_parent == "document" else 0
    assert await connection.fetchval("SELECT COUNT(*) FROM sessions") == expected_remaining
    assert await connection.fetchval("SELECT COUNT(*) FROM chat_history") == expected_remaining


async def test_generated_full_text_search_and_gin_index(postgres_connection):
    connection = postgres_connection
    _, _, chunk, _ = await seed(connection)
    statement = "SELECT id FROM chunks WHERE content_tsv @@ websearch_to_tsquery('english', $1)"
    assert await connection.fetchval(statement, "durability") == chunk
    assert await connection.fetchval(statement, "astronomy") is None
    await connection.execute("SET LOCAL enable_seqscan = off")
    plan = await connection.fetchval("EXPLAIN (FORMAT JSON) " + statement, "durability")
    if isinstance(plan, str):
        plan = json.loads(plan)
    assert "idx_chunks_content_tsv" in json.dumps(plan)


async def test_hnsw_cosine_index_and_real_vector_ordering(postgres_connection):
    connection = postgres_connection
    _, doc, first_chunk, vector = await seed(connection)
    second_chunk = await connection.fetchval(
        "INSERT INTO chunks(document_id, chunk_index, content, content_hash) "
        "VALUES($1, 1, 'Orthogonal synthetic vector.', 'second-hash') RETURNING id", doc
    )
    orthogonal = "[0,1," + ",".join(["0"] * 766) + "]"
    await connection.execute("INSERT INTO embeddings(chunk_id, embedding) VALUES($1, $2::vector)", second_chunk, orthogonal)
    metadata = await connection.fetchrow(
        "SELECT am.amname, i.indisvalid, opc.opcname FROM pg_class idx "
        "JOIN pg_index i ON i.indexrelid = idx.oid JOIN pg_am am ON am.oid = idx.relam "
        "JOIN pg_opclass opc ON opc.oid = i.indclass[0] "
        "WHERE idx.relnamespace = current_schema()::regnamespace AND idx.relname = 'idx_embeddings_vector_cosine'"
    )
    assert metadata is not None
    assert metadata["amname"] == "hnsw"
    assert metadata["indisvalid"] is True
    assert metadata["opcname"] == "vector_cosine_ops"
    statement = "SELECT chunk_id FROM embeddings ORDER BY embedding <=> $1::vector LIMIT 2"
    rows = await connection.fetch(statement, vector)
    assert [row["chunk_id"] for row in rows] == [first_chunk, second_chunk]
    await connection.execute("SET LOCAL enable_seqscan = off")
    plan = await connection.fetchval("EXPLAIN (FORMAT JSON) " + statement, vector)
    if isinstance(plan, str):
        plan = json.loads(plan)
    assert "idx_embeddings_vector_cosine" in json.dumps(plan)
