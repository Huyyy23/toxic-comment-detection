"""Stable inference interface used by API routes across model backends."""
from typing import Protocol


class InferenceAdapter(Protocol):
    model_backend: str
    def predict(self, text: str) -> dict[str, object]: ...

    def info(self) -> dict[str, object]: ...
