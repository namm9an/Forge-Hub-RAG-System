from types import SimpleNamespace
import json
import httpx
import pytest


@pytest.mark.asyncio
async def test_gemini_stream_uses_only_google_endpoint_and_selected_key(monkeypatch):
    from app import chat
    requests = []
    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-google-key"
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    real_client = httpx.AsyncClient
    monkeypatch.setattr(chat.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    cfg = SimpleNamespace(llm_provider="gemini", gemini_api_key="synthetic-google-key",
        openrouter_api_key="must-not-be-used", llm_model="synthetic-gemini-model",
        max_answer_chars=100, provider_timeout_seconds=5)
    assert [token async for token in chat._provider_tokens(cfg, [])] == ["hello"]
    assert len(requests) == 1


def test_provider_keys_are_redacted_from_settings_representation():
    from app.config import Settings
    settings = Settings(gemini_api_key="synthetic-private-google-value", openrouter_api_key="synthetic-private-router-value")
    assert "synthetic-private-google-value" not in repr(settings)
    assert "synthetic-private-router-value" not in repr(settings)


@pytest.mark.asyncio
async def test_live_evaluation_can_bound_reasoning_without_changing_normal_generation(monkeypatch):
    from app import chat
    from scripts.evaluate_quality import budgeted_completion
    payloads = []
    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'synthetic answer'}}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(chat.httpx, 'AsyncClient', lambda **kwargs: real_client(transport=httpx.MockTransport(handler),**kwargs))
    reservations = []
    budget = SimpleNamespace(reserve=lambda label, model: reservations.append((label, model)))
    settings = SimpleNamespace(llm_provider="gemini", gemini_api_key="synthetic", llm_model="gemini-flash",
                               provider_timeout_seconds=5)
    assert await budgeted_completion(settings, [], budget, "test") == "synthetic answer"
    assert payloads[0]["reasoning_effort"] == "low"
    assert reservations == [("test", "gemini-flash")]
    await chat._llm_complete_openrouter('synthetic',[], 'gemini-flash')
    assert 'reasoning_effort' not in payloads[1]
