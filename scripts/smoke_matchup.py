"""Smoke test for matchup explorer team normalization."""

from __future__ import annotations

import io
import os
import sys
from typing import Iterable

from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.team_normalization import TeamNormalizationResult, normalize_games_df  # noqa: E402


def _storage_account() -> str:
    return os.getenv("AZURE_STORAGE_ACCOUNT", "anthansunderrgaddf")


def _container_name() -> str:
    return os.getenv("AZURE_STORAGE_CONTAINER", "nba-edge")


def _lake_prefix() -> str:
    prefix = os.getenv("AZURE_LAKE_PREFIX", "lake/")
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _blob_service_client() -> BlobServiceClient:
    account_url = f"https://{_storage_account()}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


def _list_blobs(prefix: str) -> list[str]:
    client = _blob_service_client()
    container = client.get_container_client(_container_name())
    return [item.name for item in container.list_blobs(name_starts_with=prefix)]


def _read_parquet(blob_name: str) -> pd.DataFrame:
    client = _blob_service_client()
    blob = client.get_blob_client(container=_container_name(), blob=blob_name)
    data = blob.download_blob().readall()
    return pd.read_parquet(io.BytesIO(data))


def _parse_seasons(blob_names: Iterable[str]) -> list[int]:
    seasons: set[int] = set()
    for name in blob_names:
        if "/season=" in name:
            try:
                part = name.split("/season=")[1].split("/")[0]
                seasons.add(int(part))
            except Exception:
                continue
    return sorted(seasons)


def _print_summary(season: int, result: TeamNormalizationResult) -> None:
    df = result.df
    sample = []
    if "team_abbr" in df.columns:
        sample = sorted(df["team_abbr"].dropna().unique().tolist())[:10]
    print(
        f"season={season} rows={len(df)} team_source={result.source_column} "
        f"distinct_teams={result.distinct_teams} sample={sample}"
    )


def main() -> int:
    prefix = f"{_lake_prefix()}silver/games/"
    try:
        blob_names = _list_blobs(prefix)
    except Exception as exc:
        print("FAILED: Unable to list blobs for silver games.")
        print(exc)
        print("If Azure auth failed, install Azure CLI and run `az login`.")
        return 1

    seasons = _parse_seasons(blob_names)
    if not seasons:
        print("FAILED: No seasons found in silver games.")
        return 1

    failures = 0
    warnings = 0
    for season in seasons:
        blob_name = f"{prefix}season={season}/games.parquet"
        try:
            df = _read_parquet(blob_name)
        except Exception as exc:
            print(f"FAILED: Could not read {blob_name}")
            print(exc)
            failures += 1
            continue

        result = normalize_games_df(df)
        _print_summary(season, result)

        if result.distinct_teams < 10:
            print(f"FAILED: season {season} has < 10 distinct teams.")
            failures += 1
        elif result.distinct_teams < 20:
            print(f"WARNING: season {season} has < 20 distinct teams.")
            warnings += 1

    if failures:
        print(f"Smoke test failed with {failures} failures and {warnings} warnings.")
        return 1

    if warnings:
        print(f"Smoke test passed with {warnings} warnings.")
    else:
        print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
