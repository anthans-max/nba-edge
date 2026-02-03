"""Transform raw odds spreads into a cleaned silver odds table."""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
from datetime import datetime

import pandas as pd

from common.blob import download_bytes, list_blobs, upload_bytes
from common.config import Config


RUN_DT_RE = re.compile(r"run_dt=([^/]+)")


def main() -> None:
    """Entry point for building the silver odds spreads dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config()
    prefix = config.raw_odds_spreads_path()

    blobs = list(list_blobs(config.BLOB_CONTAINER, prefix))
    if not blobs:
        logging.info("No raw odds blobs found under %s", prefix)
        return

    rows_by_date: dict[str, list[pd.DataFrame]] = {}
    for blob in blobs:
        if not blob.name.endswith("payload.json.gz"):
            continue

        payload = _load_gz_json(config.BLOB_CONTAINER, blob.name)
        snapshot_dt = _snapshot_dt_from_path(blob.name)
        df = _payload_to_dataframe(payload, snapshot_dt)
        if df.empty:
            continue

        partition_date = df["snapshot_dt"].dt.date.astype(str).iloc[0]
        rows_by_date.setdefault(partition_date, []).append(df)

    for partition_date, frames in rows_by_date.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            continue

        combined = combined.drop_duplicates(
            subset=[
                "snapshot_dt",
                "commence_time",
                "home_team",
                "away_team",
                "bookmaker",
                "last_update",
            ],
            keep="last",
        )
        combined = combined.sort_values(["commence_time", "home_team", "bookmaker"]).reset_index(
            drop=True
        )

        blob_name = f"{config.BLOB_PREFIX}/silver/odds_spreads/date={partition_date}/odds.parquet"
        data = _to_parquet_bytes(combined)
        upload_bytes(config.BLOB_CONTAINER, blob_name, data, content_type="application/octet-stream")
        logging.info("Wrote date=%s rows=%s blob=%s", partition_date, len(combined), blob_name)


def _payload_to_dataframe(payload: list, snapshot_dt: datetime | None) -> pd.DataFrame:
    rows: list[dict] = []
    snapshot_value = snapshot_dt or datetime.utcnow()

    for game in payload or []:
        commence_time = _parse_dt(game.get("commence_time"))
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        for bookmaker in game.get("bookmakers", []) or []:
            bookmaker_key = bookmaker.get("key") or bookmaker.get("title")
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "spreads":
                    continue
                last_update = _parse_dt(market.get("last_update"))
                outcomes = market.get("outcomes", []) or []
                spreads = _extract_spreads(outcomes, home_team, away_team)
                rows.append(
                    {
                        "snapshot_dt": snapshot_value,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker_key,
                        "last_update": last_update,
                        "spread_home": spreads.get("home"),
                        "spread_away": spreads.get("away"),
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"], errors="coerce", utc=True)
    df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce", utc=True)
    return df


def _extract_spreads(outcomes: list, home_team: str | None, away_team: str | None) -> dict:
    spread_home = None
    spread_away = None
    for outcome in outcomes:
        name = outcome.get("name")
        point = outcome.get("point")
        if name == home_team:
            spread_home = point
        elif name == away_team:
            spread_away = point
    return {"home": spread_home, "away": spread_away}


def _snapshot_dt_from_path(path: str) -> datetime | None:
    match = RUN_DT_RE.search(path)
    if not match:
        return None
    value = match.group(1)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_gz_json(container: str, blob_name: str) -> list:
    raw = download_bytes(container, blob_name)
    data = gzip.decompress(raw)
    return json.loads(data)


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
