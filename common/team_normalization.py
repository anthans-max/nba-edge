"""Normalize team identifiers across silver games datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

NULL_LITERALS = {"", "none", "nan", "null", "na", "n/a"}

TEAM_ROWS: list[tuple[int, str, str]] = [
    (1610612737, "ATL", "Atlanta Hawks"),
    (1610612738, "BOS", "Boston Celtics"),
    (1610612751, "BKN", "Brooklyn Nets"),
    (1610612766, "CHA", "Charlotte Hornets"),
    (1610612741, "CHI", "Chicago Bulls"),
    (1610612739, "CLE", "Cleveland Cavaliers"),
    (1610612742, "DAL", "Dallas Mavericks"),
    (1610612743, "DEN", "Denver Nuggets"),
    (1610612765, "DET", "Detroit Pistons"),
    (1610612744, "GSW", "Golden State Warriors"),
    (1610612745, "HOU", "Houston Rockets"),
    (1610612754, "IND", "Indiana Pacers"),
    (1610612746, "LAC", "Los Angeles Clippers"),
    (1610612747, "LAL", "Los Angeles Lakers"),
    (1610612763, "MEM", "Memphis Grizzlies"),
    (1610612748, "MIA", "Miami Heat"),
    (1610612749, "MIL", "Milwaukee Bucks"),
    (1610612750, "MIN", "Minnesota Timberwolves"),
    (1610612740, "NOP", "New Orleans Pelicans"),
    (1610612752, "NYK", "New York Knicks"),
    (1610612760, "OKC", "Oklahoma City Thunder"),
    (1610612753, "ORL", "Orlando Magic"),
    (1610612755, "PHI", "Philadelphia 76ers"),
    (1610612756, "PHX", "Phoenix Suns"),
    (1610612757, "POR", "Portland Trail Blazers"),
    (1610612758, "SAC", "Sacramento Kings"),
    (1610612759, "SAS", "San Antonio Spurs"),
    (1610612761, "TOR", "Toronto Raptors"),
    (1610612762, "UTA", "Utah Jazz"),
    (1610612764, "WAS", "Washington Wizards"),
]

TEAM_ID_TO_ABBR = {team_id: abbr for team_id, abbr, _ in TEAM_ROWS}
TEAM_ABBRS = {abbr for _, abbr, _ in TEAM_ROWS}
TEAM_NAME_BY_ABBR = {abbr: name for _, abbr, name in TEAM_ROWS}


def _normalize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


NAME_TO_ABBR = {_normalize_name(name): abbr for _, abbr, name in TEAM_ROWS}
NAME_TO_ABBR.update(
    {
        "la clippers": "LAC",
        "los angeles clippers": "LAC",
        "la lakers": "LAL",
        "los angeles lakers": "LAL",
        "new orleans hornets": "NOP",
        "charlotte bobcats": "CHA",
        "new jersey nets": "BKN",
        "seattle supersonics": "OKC",
    }
)


@dataclass(frozen=True)
class TeamNormalizationResult:
    df: pd.DataFrame
    source_column: str | None
    distinct_teams: int


def _clean_string_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    mask = s.str.lower().isin(NULL_LITERALS)
    return s.mask(mask, pd.NA)


def _normalize_to_abbr(value: str | int | float | None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return TEAM_ID_TO_ABBR.get(int(text))
    upper = text.upper()
    if upper in TEAM_ABBRS:
        return upper
    name_key = _normalize_name(text)
    return NAME_TO_ABBR.get(name_key)


def _series_to_abbr(series: pd.Series, treat_as_name: bool = False) -> pd.Series:
    cleaned = _clean_string_series(series)
    if treat_as_name:
        return cleaned.map(lambda v: _normalize_to_abbr(v))
    return cleaned.map(lambda v: _normalize_to_abbr(v))


def _first_non_empty(series_list: Iterable[tuple[str, pd.Series]]) -> tuple[str | None, pd.Series | None]:
    for name, series in series_list:
        if series is None:
            continue
        distinct = series.dropna().nunique()
        if distinct > 0:
            return name, series
    return None, None


def normalize_games_df(df: pd.DataFrame) -> TeamNormalizationResult:
    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]

    candidates: list[tuple[str, pd.Series | None]] = []

    if "team_abbr" in data.columns:
        candidates.append(("team_abbr", _series_to_abbr(data["team_abbr"])))
    if "team_abbreviation" in data.columns:
        candidates.append(("team_abbreviation", _series_to_abbr(data["team_abbreviation"])))
    if "team" in data.columns:
        candidates.append(("team", _series_to_abbr(data["team"])))
    if "team_name" in data.columns:
        candidates.append(("team_name", _series_to_abbr(data["team_name"], treat_as_name=True)))
    if "team_id" in data.columns:
        team_id_series = pd.to_numeric(data["team_id"], errors="coerce").map(TEAM_ID_TO_ABBR)
        candidates.append(("team_id", team_id_series))

    source_column, abbr_series = _first_non_empty(candidates)
    if abbr_series is not None:
        data["team_abbr"] = abbr_series

    distinct = int(data["team_abbr"].dropna().nunique()) if "team_abbr" in data.columns else 0
    return TeamNormalizationResult(df=data, source_column=source_column, distinct_teams=distinct)
