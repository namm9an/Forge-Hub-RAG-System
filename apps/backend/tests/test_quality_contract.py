import json

import httpx
import pytest

from scripts.evaluate_quality import CallBudget, load_live_settings, summarize, validate_judgment


def test_call_budget_persists_failed_attempts_and_never_exceeds_approval(tmp_path):
    path = tmp_path/'budget.json'
    path.write_text(json.dumps({'limit':2,'used':0,'attempts':[]}))
    assert CallBudget(path).reserve('first','gemini-flash') == 1
    assert CallBudget(path).reserve('failed request also counts','gemini-flash') == 2
    with pytest.raises(ValueError, match='exhausted'):
        CallBudget(path).reserve('third','gemini-flash')
    assert json.loads(path.read_text())['used'] == 2
    assert not path.with_suffix('.json.lock').exists()


def test_budget_fails_closed_if_another_process_holds_lock(tmp_path):
    path = tmp_path/'budget.json'
    path.write_text(json.dumps({'limit':10,'used':0,'attempts':[]}))
    path.with_suffix('.json.lock').touch()
    with pytest.raises(FileExistsError):
        CallBudget(path).reserve('concurrent','gemini-flash')
    assert json.loads(path.read_text())['used'] == 0


def cases():
    return [{'id':'q1','sources':[{'id':'s1','text':'fact'},{'id':'s2','text':'noise'}],
             'expected_facts':[{'id':'f1','fact':'fact'}], 'answer':'fact'}]


def item():
    return {'id':'q1','relevant_source_ids':['s1'],'missing_fact_ids':[],
            'faithfulness':1.0,'answer_relevance':0.9}


@pytest.mark.parametrize('change', [
    {'id':'unknown'}, {'relevant_source_ids':['invented']}, {'missing_fact_ids':['invented']},
    {'relevant_source_ids':['s1','s1']}, {'faithfulness':float('nan')},
    {'answer_relevance':1.1}, {'faithfulness':True}, {'answer_relevance':None},
])
def test_judgment_rejects_unknown_evidence_and_invalid_scores(change):
    with pytest.raises(ValueError):
        validate_judgment(cases(), {'items':[dict(item(),**change)]})


def test_missing_duplicate_cases_cannot_be_silently_averaged():
    with pytest.raises(ValueError):
        validate_judgment(cases(), {'items':[]})
    with pytest.raises(ValueError):
        validate_judgment(cases(), {'items':[item(),item()]})


def test_metrics_distinguish_context_precision_fact_recall_and_missing_generation():
    judged = validate_judgment(cases(), {'items':[item()]})
    result = summarize(cases(),judged)
    assert result['source_context_precision'] == 0.5
    assert result['fact_recall'] == 1
    assert result['faithfulness'] == 1
    assert result['answer_relevance'] == 0.9
    assert result['backend_release_gate'] == 'INCOMPLETE'
    no_answer = cases()
    no_answer[0].pop('answer')
    judged = validate_judgment(no_answer, {'items':[dict(item(),faithfulness=None,answer_relevance=None)]})
    assert summarize(no_answer,judged)['faithfulness'] is None


def test_absent_answer_cannot_receive_faithfulness_or_relevance_scores():
    dataset = cases()
    dataset[0].pop('answer')
    with pytest.raises(ValueError):
        validate_judgment(dataset,{'items':[item()]})


def test_explicit_credentials_never_interpolate_ambient_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('FORGE_REVIEW_SENTINEL','synthetic-key-from-ambient')
    path = tmp_path/'evaluation.env'
    path.write_text('GEMINI_API_KEY=${FORGE_REVIEW_SENTINEL}\nLLM_MODEL=gemini-flash\nALLOW_LIVE_EVALUATION=true\n')
    with pytest.raises(ValueError):
        load_live_settings(path)


def synthetic_claim_scores(answer, sources, expected):
    # These authored fixtures use period-separated claims. Real model scoring needs independent labels.
    answer = answer.removeprefix("Evidence excerpts (local extractive fallback):")
    for source in sources:
        answer = answer.replace(
            f"[Doc: {source['filename']}, Page: {source['page_number']}, Chunk: {source['id']}]", "",
        )
    claims = [claim.strip() for claim in answer.split(".") if claim.strip()]
    supported = {claim.strip() for source in sources for claim in source["content"].split(".") if claim.strip()}
    faithfulness = sum(claim in supported for claim in claims) / len(claims) if claims else 0
    relevance = 2 * len(set(claims) & set(expected)) / (len(claims) + len(expected))
    return faithfulness, relevance


def test_synthetic_claim_metrics_distinguish_unsupported_and_irrelevant_statements():
    sources = [{"filename": "notes.pdf", "page_number": 1, "id": "one",
                "content": "Keep logs for 35 days. The rack is blue."}]
    tag = "[Doc: notes.pdf, Page: 1, Chunk: one]"
    expected = ("Keep logs for 35 days",)
    assert synthetic_claim_scores(f"Keep logs for 35 days. Use the spare drive. {tag}", sources, expected) == (0.5, 2 / 3)
    assert synthetic_claim_scores(f"Keep logs for 35 days. The rack is blue. {tag}", sources, expected) == (1.0, 2 / 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("provider_status", [200, 400, 503], ids=["mock-provider", "fallback-400", "fallback-503"])
async def test_synthetic_answer_quality_contract(streaming, provider_status, monkeypatch):
    from app import app, chat
    from app.auth import Principal, require_user
    from app.config import Settings, get_settings
    from app.schemas import SearchResultItem

    labeled = {
        "How long are audit logs retained?": ("Audit logs are retained for 35 days",),
        "Which protocol protects archive replication?": ("Archive replication uses mutual TLS",),
        "Where is the archive manifest stored and how long is it retained?": (
            "The archive manifest is stored in cold storage", "The archive manifest is retained for 90 days",
        ),
    }
    settings = Settings(_env_file=None, openrouter_api_key="synthetic-provider-key", max_history_turns=0)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[require_user] = lambda: Principal("quality-fixture-owner")

    async def retrieve(query, session_id, top_k, config, *, user_id):
        text = ". ".join(labeled[query]) + "."
        return [SearchResultItem(chunk_id="fact", document_id="synthetic", filename="runbook.pdf",
                                 page_number=3, content=text, score=0.02, dense_score=0.95, sparse_score=0.8)]

    class Ranker:
        def rerank(self, query, candidates, **kwargs):
            return [dict(row.model_dump(), rerank_score=0.99) for row in candidates]

    monkeypatch.setattr(chat, "_perform_hybrid_retrieval", retrieve)
    monkeypatch.setattr(chat, "get_reranker", lambda *args: Ranker())
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        prompt = json.loads(payload["messages"][1]["content"])
        requests.append((request.url.path, payload, prompt))
        if provider_status != 200:
            return httpx.Response(provider_status)
        text = ". ".join(labeled[prompt["question"]]) + ". [Doc: runbook.pdf, Page: 3, Chunk: fact]"
        if not streaming:
            return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})
        event = {"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(chat.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(provider), **kw))
    expected_backend = "provider" if provider_status == 200 else "extractive_fallback"
    answers = []
    async with real_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for question, expected in labeled.items():
            body = {"query": question, "session_id": "quality-fixture"}
            if streaming:
                response = await client.post("/api/chat/stream", json=body, headers={"Accept": "text/event-stream"})
                assert response.status_code == 200
                assert response.text.endswith("data: [DONE]\n\n")
                assert "event: error" not in response.text
                items = [json.loads(line[6:]) for line in response.text.splitlines()
                         if line.startswith("data: ") and line != "data: [DONE]"]
                answer = "".join(item["token"] for item in items if "token" in item)
                result, = [item for item in items if item.get("complete") is True]
            else:
                response = await client.post("/api/chat", json=body)
                assert response.status_code == 200
                result = response.json()
                answer = result["answer"]
            assert result["generation_backend"] == expected_backend
            assert [source["id"] for source in result["sources"]] == ["fact"]
            faithfulness, relevance = synthetic_claim_scores(answer, result["sources"], expected)
            assert faithfulness >= 0.95 and relevance >= 0.88
            answers.append(answer)
        history = await client.get("/api/chat/history", params={"session_id": "quality-fixture"})
        assert history.status_code == 200
        assert [message["assistant_message"] for message in history.json()["messages"]] == answers[::-1]
    assert len(requests) == len(labeled)
    for (path, payload, prompt), (question, facts) in zip(requests, labeled.items()):
        assert path == "/api/v1/chat/completions"
        assert payload["messages"][0]["role"] == "system"
        assert "untrusted data" in payload["messages"][0]["content"]
        assert prompt["question"] == question
        assert all(fact in prompt["untrusted_evidence"] for fact in facts)
