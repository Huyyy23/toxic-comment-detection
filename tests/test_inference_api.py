from fastapi.testclient import TestClient
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.main import create_app


def test_binary_joblib_api_contract(tmp_path, monkeypatch):
    pipeline = Pipeline(
        [
            ("tfidf", CountVectorizer()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(
        ["bài viết rất tốt", "cảm ơn bạn", "đồ ngu", "thật đáng ghét"],
        [0, 0, 1, 1],
    )
    artifact = tmp_path / "baseline.joblib"
    dump(pipeline, artifact)
    monkeypatch.setenv("N11_MODEL_PATH", str(artifact))

    client = TestClient(create_app())
    health = client.get("/health")
    info = client.get("/model-info")
    prediction = client.post("/predict", json={"text": "Bài viết rất tốt"})

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert info.status_code == 200
    assert info.json()["label_taxonomy"] == {"0": "CLEAN", "1": "TOXIC"}
    assert prediction.status_code == 200
    result = prediction.json()
    assert result["label_name"] in {"CLEAN", "TOXIC"}
    assert result["label_id"] in {0, 1}
    assert set(result["probabilities"]) == {"CLEAN", "TOXIC"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    assert result["is_calibrated"] is False
    assert client.post("/predict", json={"text": "   "}).status_code == 422
