"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI service."""
from __future__ import annotations

import httpx


class ApiError(Exception):
    pass


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _check(self, r: httpx.Response):
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ApiError(str(detail))
        return r

    def health(self) -> bool:
        try:
            return self.http.get("/health", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def upload(self, name: str, content: bytes) -> dict:
        return self._check(self.http.post("/upload", files={"file": (name, content)})).json()

    def detect(self, dataset_id: str, columns=None, exclude_zeros=False, top_n=10) -> dict:
        body = {"dataset_id": dataset_id, "columns": columns, "exclude_zeros": exclude_zeros, "top_n": top_n}
        return self._check(self.http.post("/detect", json=body)).json()

    def explain(self, dataset_id: str, rows=None, top_n=5, exclude_zeros=False) -> dict:
        params = {"top_n": top_n, "exclude_zeros": exclude_zeros}
        if rows:
            params["rows"] = ",".join(map(str, rows))
        return self._check(self.http.get(f"/explain/{dataset_id}", params=params)).json()

    def scored(self, dataset_id: str, exclude_zeros=False) -> dict:
        return self._check(self.http.get(f"/scored/{dataset_id}", params={"exclude_zeros": exclude_zeros})).json()

    def report(self, dataset_id: str, exclude_zeros=False) -> bytes:
        return self._check(self.http.get(f"/report/{dataset_id}", params={"exclude_zeros": exclude_zeros})).content

    def chat(self, dataset_id: str, question: str, session_id: str) -> dict:
        body = {"dataset_id": dataset_id, "question": question, "session_id": session_id}
        return self._check(self.http.post("/chat", json=body)).json()

    def drift(self, reference_id: str, current_id: str) -> dict:
        return self._check(self.http.post("/drift", json={"reference_id": reference_id, "current_id": current_id})).json()
