"""Integration smoke test for blob download + gzip parse without Azure."""

import gzip
import json

import transform.build_silver_games as build_silver_games


def test_download_and_parse_with_mocked_blob_client(monkeypatch):
    payload = {"resource": "test", "parameters": {}, "resultSets": []}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))

    class _FakeDownloader:
        def readall(self):
            return raw

    class _FakeBlobClient:
        def download_blob(self):
            return _FakeDownloader()

    class _FakeService:
        def get_blob_client(self, container, blob):
            return _FakeBlobClient()

    monkeypatch.setattr(build_silver_games, "get_blob_service_client", lambda: _FakeService())

    data = build_silver_games._download_blob_bytes("container", "blob")
    parsed = build_silver_games._parse_gz_json_bytes(data)

    assert parsed == payload
