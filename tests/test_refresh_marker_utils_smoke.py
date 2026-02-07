"""Smoke tests for refresh marker parsing and cache signatures."""

from __future__ import annotations

from datetime import datetime, timezone

from app.ui.status_utils import marker_signature, parse_refresh_marker_payload


def test_parse_refresh_marker_payload_smoke() -> None:
    raw = (
        b'{"max_game_date":"2026-02-05","rows":1536,'
        b'"season":2025,"timestamp_utc":"2026-02-06T23:39:33.907576+00:00"}'
    )
    payload = parse_refresh_marker_payload(raw)
    assert payload["season"] == 2025
    assert payload["max_game_date"] == "2026-02-05"


def test_marker_signature_prefers_last_modified() -> None:
    meta = {"last_modified_utc": datetime(2026, 2, 7, 12, 0, 0, tzinfo=timezone.utc)}
    payload = {"timestamp_utc": "2026-02-06T23:39:33.907576+00:00"}
    signature = marker_signature(meta, payload)
    assert signature is not None
    assert signature.startswith("last_modified:")


def test_marker_signature_falls_back_to_timestamp() -> None:
    payload = {"timestamp_utc": "2026-02-06T23:39:33.907576+00:00"}
    signature = marker_signature(None, payload)
    assert signature == "timestamp:2026-02-06T23:39:33.907576+00:00"
