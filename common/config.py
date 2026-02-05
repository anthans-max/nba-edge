"""Centralized configuration and blob path helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import io
import os

import pandas as pd
from dotenv import load_dotenv

from common.blob import download_bytes


load_dotenv()


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _join_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part)


@dataclass(frozen=True)
class Config:
    STORAGE_ACCOUNT_NAME: str | None = field(default_factory=lambda: _env_str("STORAGE_ACCOUNT_NAME"))
    BLOB_CONTAINER: str = field(default_factory=lambda: _env_str("BLOB_CONTAINER", "nba-edge") or "nba-edge")
    BLOB_PREFIX: str = field(default_factory=lambda: _env_str("BLOB_PREFIX", "lake") or "lake")
    ODDS_API_KEY: str | None = field(default_factory=lambda: _env_str("ODDS_API_KEY"))
    ODDS_REGIONS: str = field(default_factory=lambda: _env_str("ODDS_REGIONS", "us") or "us")
    ODDS_MARKETS: str = field(default_factory=lambda: _env_str("ODDS_MARKETS", "spreads") or "spreads")
    START_SEASON: int = field(default_factory=lambda: _env_int("START_SEASON", 2022))
    END_SEASON: int = field(default_factory=lambda: _env_int("END_SEASON", datetime.utcnow().year))
    WINDOW_SIZES: str = field(default_factory=lambda: _env_str("WINDOW_SIZES", "5,10,30") or "5,10,30")

    def raw_nba_teamgamelogs_path(self) -> str:
        return _join_path(self.BLOB_PREFIX, "raw", "nba", "teamgamelogs")

    def raw_odds_spreads_path(self) -> str:
        return _join_path(self.BLOB_PREFIX, "raw", "odds", "spreads")

    def silver_games_parquet_path(self) -> str:
        return _join_path(self.BLOB_PREFIX, "silver", "games.parquet")

    def silver_odds_parquet_path(self) -> str:
        return _join_path(self.BLOB_PREFIX, "silver", "odds.parquet")

    def features_parquet_path(self) -> str:
        return _join_path(self.BLOB_PREFIX, "features", "features.parquet")

    def backtests_path(self) -> str:
        return _join_path(self.BLOB_PREFIX, "reports", "backtests")

    def read_parquet(self, blob_name: str) -> pd.DataFrame:
        data = download_bytes(self.BLOB_CONTAINER, blob_name)
        return pd.read_parquet(io.BytesIO(data))
