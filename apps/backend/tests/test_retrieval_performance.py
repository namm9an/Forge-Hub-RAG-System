"""Deterministic retrieval scheduling checks without running live models."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app import embeddings


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(tuple(texts))
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, int(text)] = 1.0
        return vectors


async def test_warmed_model_lookup_does_not_wait_for_a_worker(monkeypatch):
    def unavailable_worker(*args, **kwargs):
        pytest.fail("Looking up a warmed model did not need a blocking worker")

    model = RecordingEncoder()
    monkeypatch.setattr(embeddings, "_model", model)
    monkeypatch.setattr(embeddings, "run_blocking", unavailable_worker)
    assert await embeddings.get_model_async(SimpleNamespace()) is model


async def test_query_embeddings_batch_concurrent_distinct_inputs_in_order(monkeypatch):
    model = RecordingEncoder()
    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(embedding_workers=2))

    vectors = await asyncio.gather(*(embeddings.embed_query(model, str(index)) for index in range(19)))

    assert [np.argmax(vector) for vector in vectors] == list(range(19))
    assert sorted(len(call) for call in model.calls) == [3, 8, 8]
    assert sorted(int(text) for call in model.calls for text in call) == list(range(19))


async def test_cancelled_queued_query_does_not_reach_encoder(monkeypatch):
    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(embedding_workers=1))
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    class SlowEncoder(RecordingEncoder):
        def encode(self, texts, **kwargs):
            if texts == ["1"]:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(timeout=3)
            return super().encode(texts, **kwargs)

    model = SlowEncoder()
    first = asyncio.create_task(embeddings.embed_query(model, "1"))
    await asyncio.wait_for(started.wait(), timeout=2)
    cancelled = asyncio.create_task(embeddings.embed_query(model, "2"))
    second = asyncio.create_task(embeddings.embed_query(model, "3"))
    try:
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert list(embeddings._query_batchers[loop].pending.values()) == ["3"]
    finally:
        release.set()

    vectors = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
    assert [np.argmax(vector) for vector in vectors] == [1, 3]
    assert model.calls == [("1",), ("3",)]


async def test_cancelled_formed_query_batch_does_not_start_on_cpu(monkeypatch):
    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(embedding_workers=1))
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def occupy_worker():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=3)

    worker = asyncio.create_task(embeddings.run_blocking(occupy_worker))
    await asyncio.wait_for(started.wait(), timeout=2)
    model = RecordingEncoder()
    cancelled = asyncio.create_task(embeddings.embed_query(model, "1"))
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not embeddings._query_batchers[loop].pending
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
    finally:
        release.set()
        await asyncio.wait_for(worker, timeout=2)

    result = await asyncio.wait_for(embeddings.embed_query(model, "2"), timeout=2)
    assert np.argmax(result) == 2
    assert model.calls == [("2",)]


async def test_query_batch_error_does_not_fail_a_valid_neighbor():
    class Encoder(RecordingEncoder):
        def encode(self, texts, **kwargs):
            if "bad" in texts:
                raise ValueError("synthetic encoding failure")
            return super().encode(texts, **kwargs)

    model = Encoder()
    good, bad = await asyncio.gather(embeddings.embed_query(model, "1"), embeddings.embed_query(model, "bad"),
                                    return_exceptions=True)

    assert np.argmax(good) == 1
    assert isinstance(bad, ValueError)


async def test_waiting_reranker_does_not_take_an_encoder_slot(monkeypatch):
    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(embedding_workers=2))
    started, release, second_started = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    def first_rank():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=3)

    first = asyncio.create_task(embeddings.run_reranking(first_rank))
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(embeddings.run_reranking(second_started.set))
        result = await asyncio.wait_for(embeddings.embed_query(RecordingEncoder(), "1"), timeout=2)
        assert np.argmax(result) == 1
        assert not second_started.is_set()
    finally:
        release.set()
        if second is not None:
            await asyncio.wait_for(second, timeout=2)
    assert second_started.is_set()
