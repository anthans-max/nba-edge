"""Transform raw game logs into a cleaned silver layer games table."""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import re

import pandas as pd

from common.blob import get_blob_service_client, list_blobs, upload_bytes
from common.config import Config


SEASON_RE = re.compile(r"season=(\d{4})")
TEAM_ID_RE = re.compile(r"team_id=(\d+)")
REQUIRED_COLUMNS = {"GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "WL", "PTS"}


def main() -> None:
    """Entry point for building the silver games dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config()
    prefix = config.raw_nba_teamgamelogs_path()
    debug_blob = os.getenv("DEBUG_BLOB")

    if debug_blob:
        _run_debug_blob(config, debug_blob)
        return

    blobs = list(list_blobs(config.BLOB_CONTAINER, prefix))
    if not blobs:
        logging.info("No raw game log blobs found under %s", prefix)
        return

    frames: list[pd.DataFrame] = []
    total_scanned = 0
    parsed_ok = 0
    total_rows = 0
    counts = {
        "download_failed": 0,
        "truly_empty_bytes": 0,
        "decompress_failed": 0,
        "json_failed": 0,
        "df_empty": 0,
        "missing_required": 0,
        "skipped_non_payload": 0,
        "skipped_out_of_range": 0,
    }
    for blob in blobs:
        total_scanned += 1
        if not blob.name.endswith("payload.json.gz"):
            counts["skipped_non_payload"] += 1
            continue
        season = _season_from_path(blob.name)
        if season is None or season < config.START_SEASON or season > config.END_SEASON:
            counts["skipped_out_of_range"] += 1
            continue

        try:
            raw = _download_blob_bytes(config.BLOB_CONTAINER, blob.name)
        except Exception as exc:
            logging.warning("Failed to download blob=%s err=%r", blob.name, exc)
            counts["download_failed"] += 1
            continue

        if len(raw) == 0:
            logging.warning("Truly empty blob=%s", blob.name)
            counts["truly_empty_bytes"] += 1
            continue

        try:
            decompressed = gzip.decompress(raw)
        except Exception as exc:
            logging.warning("Failed to decompress blob=%s err=%r", blob.name, exc)
            counts["decompress_failed"] += 1
            continue

        try:
            payload = json.loads(decompressed)
        except Exception as exc:
            logging.warning("Failed to json parse blob=%s err=%r", blob.name, exc)
            counts["json_failed"] += 1
            continue

        try:
            df = _payload_to_dataframe(payload)
        except Exception as exc:
            logging.warning("Failed to build dataframe blob=%s err=%r", blob.name, exc)
            counts["json_failed"] += 1
            continue

        if df.empty:
            logging.warning("Empty payload blob=%s", blob.name)
            counts["df_empty"] += 1
            continue

        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            logging.warning(
                "Missing required columns blob=%s missing=%s present=%s",
                blob.name,
                sorted(missing),
                sorted(df.columns),
            )
            counts["missing_required"] += 1
            continue

        team_id = _team_id_from_path(blob.name)
        if team_id is not None:
            df["team_id"] = team_id

        parsed_ok += 1
        df["season"] = season
        df = _normalize_games(df)
        total_rows += len(df)
        frames.append(df)

    logging.info(
        "Scanned=%s Parsed=%s Total rows=%s",
        total_scanned,
        parsed_ok,
        total_rows,
    )
    logging.info(
        "Skipped counts: non_payload=%s out_of_range=%s download_failed=%s truly_empty=%s "
        "decompress_failed=%s json_failed=%s df_empty=%s missing_required=%s",
        counts["skipped_non_payload"],
        counts["skipped_out_of_range"],
        counts["download_failed"],
        counts["truly_empty_bytes"],
        counts["decompress_failed"],
        counts["json_failed"],
        counts["df_empty"],
        counts["missing_required"],
    )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        logging.info("No valid game rows parsed; skipping silver write.")
        return

    combined = combined.drop_duplicates(subset=["game_id", "team_id"], keep="last")
    combined = combined.sort_values(["game_date", "team_id"]).reset_index(drop=True)

    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    invalid_dates = combined["game_date"].isna().sum()
    if invalid_dates:
        logging.warning("Dropping rows with invalid game_date count=%s", invalid_dates)
        combined = combined[combined["game_date"].notna()].copy()

    if combined.empty:
        logging.info("All rows dropped due to invalid game_date; skipping silver write.")
        return

    combined["season"] = _canonical_season_start_year(combined["game_date"])
    combined["game_date"] = combined["game_date"].dt.date

    for season, season_df in combined.groupby("season", sort=True):
        if season_df.empty:
            continue
        blob_name = f"{config.BLOB_PREFIX}/silver/games/season={season}/games.parquet"
        _log_write(season, season_df, blob_name)
        data = _to_parquet_bytes(season_df)
        upload_bytes(config.BLOB_CONTAINER, blob_name, data, content_type="application/octet-stream")


def _payload_to_dataframe(payload: dict) -> pd.DataFrame:
    result_sets = payload.get("resultSets") or []
    if not result_sets:
        return pd.DataFrame()

    for result in result_sets:
        headers = result.get("headers") or []
        rows = result.get("rowSet") or []
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            df.columns = [str(c).strip().upper() for c in df.columns]
            return df

    return pd.DataFrame()


def _normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()

    output["game_id"] = df.get("GAME_ID").astype(str)
    output["season"] = df.get("season")

    game_date = df.get("GAME_DATE")
    if game_date is not None:
        output["game_date"] = pd.to_datetime(game_date, errors="coerce").dt.date
    else:
        output["game_date"] = pd.NaT

    output["team_id"] = pd.to_numeric(df.get("team_id"), errors="coerce").astype("Int64")
    output["team_abbr"] = df.get("TEAM_ABBREVIATION")
    output["matchup"] = df.get("MATCHUP")
    output["is_home"] = output["matchup"].fillna("").str.contains("vs.")

    output["pts"] = pd.to_numeric(df.get("PTS"), errors="coerce").astype("Int64")
    output["wl"] = df.get("WL")
    output["plus_minus"] = pd.to_numeric(df.get("PLUS_MINUS"), errors="coerce")

    output["opp_team_abbr"] = output["matchup"].apply(_opponent_from_matchup)

    return output


def _canonical_season_start_year(game_date: pd.Series) -> pd.Series:
    dates = pd.to_datetime(game_date, errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid game_date encountered while computing canonical season.")
    years = dates.dt.year
    adjust = (dates.dt.month < 10).astype(int)
    return (years - adjust).astype("Int64")


def _log_write(season: int, df: pd.DataFrame, blob_name: str) -> None:
    min_date = df["game_date"].min()
    max_date = df["game_date"].max()
    logging.info(
        "Writing season=%s rows=%s min_game_date=%s max_game_date=%s blob=%s",
        season,
        len(df),
        min_date,
        max_date,
        blob_name,
    )


def _opponent_from_matchup(matchup: str | None) -> str | None:
    if not matchup:
        return None
    parts = matchup.split()
    if not parts:
        return None
    return parts[-1]


def _season_from_path(path: str) -> int | None:
    match = SEASON_RE.search(path)
    if not match:
        return None
    return int(match.group(1))


def _team_id_from_path(path: str) -> int | None:
    match = TEAM_ID_RE.search(path)
    if not match:
        return None
    return int(match.group(1))


def _download_blob_bytes(container: str, blob_name: str) -> bytes:
    service = get_blob_service_client()
    blob_client = service.get_blob_client(container=container, blob=blob_name)
    return blob_client.download_blob().readall()


def _parse_gz_json_bytes(raw: bytes) -> dict:
    data = gzip.decompress(raw)
    return json.loads(data)


def _run_debug_blob(config: Config, blob_name: str) -> None:
    logging.info("DEBUG_BLOB mode enabled for %s", blob_name)
    raw = _download_blob_bytes(config.BLOB_CONTAINER, blob_name)
    logging.info("downloaded_bytes=%s", len(raw))
    if len(raw) == 0:
        logging.info("Truly empty blob=%s", blob_name)
        return

    data = gzip.decompress(raw)
    logging.info("decompressed_bytes=%s", len(data))
    payload = json.loads(data)

    logging.info("top_level_keys=%s", sorted(payload.keys()))
    result_sets = payload.get("resultSets") or []
    logging.info("resultSets_count=%s", len(result_sets))

    if result_sets:
        first = result_sets[0]
        headers = first.get("headers") or []
        rows = first.get("rowSet") or []
        logging.info(
            "first_result_set name=%s headers=%s rows=%s",
            first.get("name"),
            len(headers),
            len(rows),
        )

    df = _payload_to_dataframe(payload)
    logging.info("dataframe_shape=%s", df.shape)
    logging.info("dataframe_columns=%s", list(df.columns))
    logging.info("debug_has_game_id=%s", "GAME_ID" in df.columns)
    if not df.empty:
        logging.info("dataframe_head=%s", df.head(2).to_dict(orient="records"))


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
