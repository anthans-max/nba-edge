"""Build model features from silver data for training and inference."""

from __future__ import annotations

import io
import re
import sys
from typing import Iterable

import pandas as pd

from common.blob import download_bytes, list_blobs, upload_bytes
from common.config import Config


SEASON_RE = re.compile(r"/season=(\d{4})/")
BASE_COLUMNS = [
    "season",
    "game_id",
    "game_date",
    "team_id",
    "team_abbr",
    "opponent_id",
    "opponent_abbr",
    "is_home",
    "pts_for",
    "pts_against",
    "margin",
]
FEATURE_COLUMNS = [
    "games_played_to_date",
    "margin_avg_last_5",
    "margin_avg_last_10",
    "pts_for_avg_last_5",
    "pts_against_avg_last_5",
    "win_rate_last_5",
    "rest_days",
]


def main() -> None:
    """Entry point for feature computation."""
    print("[features] start")

    config = Config()
    prefix = f"{config.BLOB_PREFIX}/silver/games/"
    season_blobs = _discover_season_blobs(config, prefix)

    if not season_blobs:
        print(f"[features] error: no silver games found under {prefix}")
        sys.exit(1)

    seasons = sorted(season_blobs.keys())
    print(f"[features] seasons discovered: {', '.join(str(s) for s in seasons)}")

    for season in seasons:
        blob_name = season_blobs[season]
        games = _load_parquet(config, blob_name)
        print(f"[features] season={season} rows_loaded={len(games)} blob={blob_name}")
        if games.empty:
            continue

        team_games = _build_team_games(games)
        features = _compute_team_features(team_games)
        if features.empty:
            print(f"[features] season={season} produced_rows=0")
            continue

        output_blob = f"{config.BLOB_PREFIX}/silver/features/season={season}/features.parquet"
        upload_bytes(
            config.BLOB_CONTAINER,
            output_blob,
            _to_parquet_bytes(features),
            content_type="application/octet-stream",
        )
        print(f"[features] season={season} produced_rows={len(features)}")
        print(f"[features] wrote {output_blob}")

    print("[features] done")


def _discover_season_blobs(config: Config, prefix: str) -> dict[int, str]:
    blobs = list(list_blobs(config.BLOB_CONTAINER, prefix))
    seasons: dict[int, list[str]] = {}
    for blob in blobs:
        name = blob.name
        if not name.endswith("/games.parquet"):
            continue
        match = SEASON_RE.search(name)
        if not match:
            continue
        season = int(match.group(1))
        seasons.setdefault(season, []).append(name)

    resolved: dict[int, str] = {}
    for season, names in seasons.items():
        resolved[season] = sorted(names)[-1]
    return resolved


def _load_parquet(config: Config, blob_name: str) -> pd.DataFrame:
    raw = download_bytes(config.BLOB_CONTAINER, blob_name)
    return pd.read_parquet(io.BytesIO(raw))


def _build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df.columns = [str(c).lower() for c in df.columns]

    for col in ["game_id", "season", "game_date", "team_id", "team_abbr", "is_home", "pts"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in silver games data.")

    df = df.drop_duplicates(subset=["game_id", "team_id"]).copy()
    df["game_id"] = df["game_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["team_abbr"] = df["team_abbr"].astype("string")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["is_home"] = df["is_home"].astype("boolean")
    df["pts"] = pd.to_numeric(df["pts"], errors="coerce")

    left_cols = [
        "season",
        "game_id",
        "game_date",
        "team_id",
        "team_abbr",
        "is_home",
        "pts",
    ]
    if "plus_minus" in df.columns:
        left_cols.append("plus_minus")
    left = df[left_cols].copy()
    right = df[["season", "game_id", "team_id", "team_abbr", "pts"]].rename(
        columns={
            "team_id": "opponent_id",
            "team_abbr": "opponent_abbr",
            "pts": "opponent_pts",
        }
    )

    merged = left.merge(right, on=["season", "game_id"], how="left")
    merged = merged[merged["team_id"] != merged["opponent_id"]]

    merged["pts_for"] = merged["pts"]
    merged["pts_against"] = merged["opponent_pts"]
    merged["margin"] = merged["pts_for"] - merged["pts_against"]

    if "plus_minus" in merged.columns:
        plus_minus = pd.to_numeric(merged["plus_minus"], errors="coerce")
        merged["margin"] = merged["margin"].fillna(plus_minus)
        merged["pts_against"] = merged["pts_against"].fillna(merged["pts_for"] - plus_minus)

    return merged[BASE_COLUMNS]


def _compute_team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy()
    df = df.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)

    grouped = df.groupby("team_id", sort=False)

    df["games_played_to_date"] = grouped.cumcount()
    df["margin_avg_last_5"] = _rolling_mean(grouped, "margin", 5)
    df["margin_avg_last_10"] = _rolling_mean(grouped, "margin", 10)
    df["pts_for_avg_last_5"] = _rolling_mean(grouped, "pts_for", 5)
    df["pts_against_avg_last_5"] = _rolling_mean(grouped, "pts_against", 5)
    df["win_rate_last_5"] = _rolling_rate(grouped, "margin", 5)
    df["rest_days"] = grouped["game_date"].apply(lambda s: s.diff().dt.days).reset_index(
        level=0, drop=True
    )

    for feature_col in FEATURE_COLUMNS:
        df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")

    output = df[BASE_COLUMNS + FEATURE_COLUMNS].copy()
    missing = [col for col in FEATURE_COLUMNS if col not in output.columns]
    if missing:
        raise ValueError(f"Missing required feature columns before write: {missing}")
    return output


def _rolling_mean(grouped: pd.core.groupby.generic.SeriesGroupBy, column: str, window: int) -> pd.Series:
    return (
        grouped[column]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def _rolling_rate(grouped: pd.core.groupby.generic.SeriesGroupBy, column: str, window: int) -> pd.Series:
    return (
        grouped[column]
        .apply(lambda s: (s.shift(1) > 0).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
