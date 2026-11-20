from scripts.evaluate_golden import load_golden, score_case


def test_golden_fixture_keeps_distinct_annotated_positives_and_negatives():
    data = load_golden()
    assert len(data["cases"]) == 30
    assert sum(bool(case["facts"]) for case in data["cases"]) == 20


def test_golden_scores_annotated_facts_and_pages_separately_from_answer_coverage():
    case = {"facts": [
        {"page": 1, "text": "retain audit material for forty days"},
        {"page": 2, "text": "store the manifest in cold storage"},
    ]}
    common = {"document_id": "doc"}
    payload = {
        "answer": "Retain audit material for forty days.",
        "refusal_triggered": False,
        "sources": [
            dict(common, id="one", page_number=1, content="Retain audit\nmaterial for forty days."),
            dict(common, id="two", page_number=1, content="Store the manifest in cold storage."),
        ],
    }
    assert score_case(case, payload, "doc") == {
        "precision": 0.5, "facts_found_at_5": 1, "facts_expected": 2, "answer_facts_found": 1,
    }
