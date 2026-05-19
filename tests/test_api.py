from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_vague_query_clarifies_without_recommendations() -> None:
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["end_of_conversation"] is False
    assert "role" in body["reply"].lower() or "skill" in body["reply"].lower()


def test_java_developer_recommends_catalog_items() -> None:
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hiring a mid-level Java developer who works with stakeholders and backend services.",
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 1 <= len(body["recommendations"]) <= 10
    names = [item["name"].lower() for item in body["recommendations"]]
    assert any("java" in name or "coding" in name for name in names)


def test_refinement_adds_personality_emphasis() -> None:
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "We are hiring a Java developer."},
                {
                    "role": "assistant",
                    "content": "Here are assessments for the Java developer context.",
                },
                {"role": "user", "content": "Actually add personality tests for stakeholder collaboration."},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert any(item["test_type"] == "P" for item in body["recommendations"])


def test_compare_catalog_items() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "What is the difference between OPQ32r and General Ability Screen?"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 2
    assert "OPQ32r" in body["reply"]
    assert "General Ability Screen" in body["reply"]


def test_refuses_off_topic() -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Give me legal advice about firing someone."}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert "SHL assessment" in body["reply"]
