"""Smoke test for gzipped JSON parsing used by transforms."""

import gzip
import json

import transform.build_silver_games as build_silver_games


def test_parse_gz_json_bytes_roundtrip():
    payload = {"resource": "test", "parameters": {}, "resultSets": []}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    parsed = build_silver_games._parse_gz_json_bytes(raw)
    assert parsed == payload
