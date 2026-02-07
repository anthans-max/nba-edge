"""Helpers for refresh marker parsing and cache signatures."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any


def parse_refresh_marker_payload(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def marker_signature(
    meta: dict[str, Any] | None, payload: dict[str, Any] | None
) -> str | None:
    if meta:
        last_modified = meta.get("last_modified_utc")
        if isinstance(last_modified, datetime):
            return f"last_modified:{last_modified.astimezone(timezone.utc).isoformat()}"
        if isinstance(last_modified, str) and last_modified:
            return f"last_modified:{last_modified}"
        etag = meta.get("etag")
        if etag:
            return f"etag:{etag}"

    if payload:
        timestamp = payload.get("timestamp_utc")
        if isinstance(timestamp, str) and timestamp:
            return f"timestamp:{timestamp}"
        try:
            normalized = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            digest = hashlib.sha256(normalized).hexdigest()[:12]
            return f"hash:{digest}"
        except Exception:
            return None
    return None
