import json
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:7860"
MODEL_IDS = ("n6_lr_d", "v3_tri_gated_ft", "v3_phobert_v2")
SAMPLE = "Bình luận mẫu dùng để kiểm tra inference và REST API."


def get_json(path: str):
    with urlopen(f"{BASE_URL}{path}", timeout=30) as response:
        assert response.status == 200, f"GET {path} returned {response.status}"
        return json.load(response)


def post_json(path: str, payload: dict):
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        assert response.status == 200, f"POST {path} returned {response.status}"
        return json.load(response)


def assert_prediction(result: dict, expected_model: str, info: dict) -> None:
    assert result["model_id"] == expected_model
    assert result["label_id"] in (0, 1)
    assert result["label_name"] == ("TOXIC" if result["label_id"] else "CLEAN")
    probabilities = result["probabilities"]
    assert set(probabilities) == {"CLEAN", "TOXIC"}
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 0.05
    assert result["confidence"] == probabilities[result["label_name"]]
    assert result["label_id"] == int(probabilities["TOXIC"] >= info["threshold"])
    assert result["threshold"] == info["threshold"] == 0.5
    assert result["latency_ms"] >= 0.0
    assert result["is_calibrated"] is info["is_calibrated"]
    assert result["calibration_method"] == info["calibration_method"]
    assert result["calibration_status"] == info["calibration_status"]

def main() -> None:
    model_info = {item["model_id"]: item for item in get_json("/models")}
    assert set(model_info) == set(MODEL_IDS)
    assert get_json("/model-info")["model_id"] == "n6_lr_d"
    with urlopen(f"{BASE_URL}/docs", timeout=30) as response:
        assert response.status == 200 and b"Swagger UI" in response.read()

    default = post_json("/predict", {"text": SAMPLE})
    assert_prediction(default, "n6_lr_d", model_info["n6_lr_d"])
    for model_id in MODEL_IDS:
        assert_prediction(post_json("/predict", {"text": SAMPLE, "model_id": model_id}), model_id, model_info[model_id])

    compared = post_json("/compare", {"text": SAMPLE})
    assert [item["model_id"] for item in compared] == list(MODEL_IDS)
    for item, model_id in zip(compared, MODEL_IDS):
        assert_prediction(item, model_id, model_info[model_id])
    print("REST, Swagger, model selection, comparison, labels, probabilities, latency, and calibration checks passed")


if __name__ == "__main__":
    main()
