"""Ingest betting spreads from The Odds API into the raw data lake."""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import datetime, timezone

import requests

from common.blob import upload_bytes
from common.config import Config


BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def main() -> None:
    """Entry point for ingesting betting spreads."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config()
    run_dt = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": config.ODDS_REGIONS,
        "markets": config.ODDS_MARKETS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    payload = _retry_odds_request(params)
    blob_base = f"{config.raw_odds_spreads_path()}/run_dt={run_dt}"

    upload_bytes(
        config.BLOB_CONTAINER,
        f"{blob_base}/payload.json.gz",
        _gzip_json(payload),
        content_type="application/json",
    )

    metadata = {
        "request": {k: v for k, v in params.items() if k != "apiKey"},
        "fetched_at": run_dt,
        "items": len(payload) if isinstance(payload, list) else None,
    }
    upload_bytes(
        config.BLOB_CONTAINER,
        f"{blob_base}/metadata.json",
        json.dumps(metadata).encode("utf-8"),
        content_type="application/json",
    )

    logging.info("Uploaded odds spreads payload. Items=%s Blob=%s", metadata["items"], blob_base)


def _retry_odds_request(params: dict, retries: int = 3, backoff: float = 1.5):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt == retries:
                raise
            sleep_s = backoff**attempt
            logging.warning("Retry %s for odds request: %s", attempt, exc)
            time.sleep(sleep_s)


def _gzip_json(payload: dict | list) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    return gzip.compress(data)


if __name__ == "__main__":
    main()
