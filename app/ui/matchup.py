"""Matchup UI rendering and helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
import streamlit as st

from azure_blob import lake_prefix, list_blobs, read_parquet_from_blob, render_exception
from chat_context import build_matchup_context

IDENTIFIER_COLUMNS = {
    "season",
    "game_id",
    "game_date",
    "team_id",
    "team_abbr",
    "opponent_id",
    "opponent_abbr",
    "is_home",
}
OUTCOME_COLUMNS = {
    "pts_for",
    "pts_against",
    "margin",
    "wl",
    "plus_minus",
}
FEATURE_ALIASES = {
    "is_home": ["home", "home_flag", "home_indicator"],
    "games_played_to_date": ["games_played", "games_played_td", "games_played_count"],
    "margin_avg_last_5": ["margin_last_5_avg", "avg_margin_last_5", "margin_avg_5"],
    "margin_avg_last_10": ["margin_last_10_avg", "avg_margin_last_10", "margin_avg_10"],
    "pts_for_avg_last_5": ["pts_for_last_5_avg", "avg_pts_for_last_5"],
    "pts_against_avg_last_5": ["pts_against_last_5_avg", "avg_pts_against_last_5"],
    "win_rate_last_5": ["win_pct_last_5", "win_rate_last5", "win_pct_5"],
    "rest_days": ["days_rest", "rest_days_last", "rest_days_avg"],
}
INFERENCE_COMPUTED_FEATURES = {"is_home"}
SEASON_RE = re.compile(r"/season=(\d{4})/")


@st.cache_data(show_spinner=False)
def load_games(season: int, marker_signature: str | None = None) -> pd.DataFrame:
    prefix = lake_prefix()
    blob_name = f"{prefix}silver/games/season={season}/games.parquet"
    df = read_parquet_from_blob(blob_name, cache_buster=marker_signature)
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    assert "team_abbr" in df.columns, f"team_abbr missing; columns={df.columns.tolist()}"
    df["team_abbr"] = (
        df["team_abbr"]
        .astype("string")
        .str.strip()
        .replace({"None": None, "nan": None, "NaN": None, "NULL": None, "": None})
    )
    df["season"] = df["season"].astype(int)
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    df["pts"] = pd.to_numeric(df.get("pts"), errors="coerce")
    df["plus_minus"] = pd.to_numeric(df.get("plus_minus"), errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_features_table(season: int, expected_features: tuple[str, ...]) -> pd.DataFrame:
    prefix = lake_prefix()
    season_prefix = f"{prefix}silver/features/season={season}/"
    print(f"[matchup] feature prefix {season_prefix}", flush=True)
    blobs = list_blobs(season_prefix)
    parquet_blobs = [b for b in blobs if str(b.get("name", "")).endswith(".parquet")]
    if not parquet_blobs:
        raise FileNotFoundError(f"No parquet files found under {season_prefix}")

    def _candidate_key(blob: dict) -> tuple[int, int]:
        name = str(blob.get("name", ""))
        is_features = 0 if name.endswith("features.parquet") else 1
        size = int(blob.get("size_bytes") or 0)
        return (is_features, -size)

    candidates = sorted(parquet_blobs, key=_candidate_key)
    chosen_blob: dict | None = None
    chosen_df: pd.DataFrame | None = None
    best_overlap = -1
    expected_core = [
        f for f in expected_features if f not in INFERENCE_COMPUTED_FEATURES
    ]

    for blob in candidates:
        blob_name = str(blob.get("name", ""))
        df = read_parquet_from_blob(blob_name)
        df = _normalize_features_frame(df)
        df = _apply_feature_aliases_to_columns(df)
        overlap = len([f for f in expected_core if f in df.columns])
        print(
            f"[matchup] parquet_candidate={blob_name} overlap={overlap}",
            flush=True,
        )
        if overlap > best_overlap:
            chosen_blob = blob
            chosen_df = df
            best_overlap = overlap
            if expected_core and overlap == len(expected_core):
                break

    if chosen_blob is None or chosen_df is None:
        raise FileNotFoundError(f"No usable features parquet found under {season_prefix}")

    print(
        f"[matchup] chosen_parquet={chosen_blob.get('name')} overlap={best_overlap}",
        flush=True,
    )
    print(
        f"[matchup] features parquet chosen name={chosen_blob.get('name')} size={chosen_blob.get('size_bytes')}",
        flush=True,
    )
    key_cols = [
        "team_id",
        "team_abbr",
        "team",
        "team_abbrev",
        "team_abbreviation",
        "team_code",
        "game_date",
        "feature_date",
        "date",
        "asof_date",
    ]
    required_missing = _missing_expected_columns(chosen_df, expected_features)
    print(
        f"[matchup] loaded features season={season} rows={len(chosen_df)} "
        f"cols={list(chosen_df.columns)[:30]} key_cols_present={[c for c in key_cols if c in chosen_df.columns]} "
        f"required_missing={required_missing}",
        flush=True,
    )
    return chosen_df


@st.cache_data(show_spinner=False)
def list_available_feature_seasons() -> list[int]:
    prefix = f"{lake_prefix()}silver/features/"
    blobs = list_blobs(prefix)
    seasons: set[int] = set()
    for blob in blobs:
        name = str(blob.get("name", ""))
        match = SEASON_RE.search(name)
        if match:
            seasons.add(int(match.group(1)))
    return sorted(seasons)


@st.cache_data(show_spinner=False)
def _nba_team_metadata() -> dict[int, dict[str, str]]:
    try:
        from nba_api.stats.static import teams as nba_static_teams
    except Exception:
        return {}

    mapping: dict[int, dict[str, str]] = {}
    try:
        for team in nba_static_teams.get_teams():
            raw_id = team.get("id")
            if raw_id is None:
                continue
            try:
                team_id = int(raw_id)
            except Exception:
                continue
            mapping[team_id] = {
                "abbreviation": str(team.get("abbreviation") or "").strip(),
                "full_name": str(team.get("full_name") or "").strip(),
            }
    except Exception:
        return {}
    return mapping


@st.cache_data(show_spinner=False)
def list_feature_teams(season: int) -> tuple[list[int], dict[int, str]]:
    df = load_features_table(season, tuple())
    if "team_id" not in df.columns:
        raise KeyError(
            f"team_id missing in features parquet for season={season}; columns={df.columns.tolist()}"
        )

    numeric_ids = pd.to_numeric(df["team_id"], errors="coerce").dropna()
    team_ids = sorted({int(v) for v in numeric_ids.tolist()})
    if not team_ids:
        raise ValueError(f"No team_id values found in features parquet for season={season}.")

    metadata = _nba_team_metadata()
    labels: dict[int, str] = {}
    for team_id in team_ids:
        team_meta = metadata.get(team_id, {})
        abbr = str(team_meta.get("abbreviation") or "").strip()
        full_name = str(team_meta.get("full_name") or "").strip()
        if abbr and full_name:
            labels[team_id] = f"{abbr} - {full_name}"
        elif abbr:
            labels[team_id] = abbr
        elif full_name:
            labels[team_id] = full_name
        else:
            labels[team_id] = f"Team {team_id}"
    return team_ids, labels


def _missing_expected_columns(
    df: pd.DataFrame, expected_features: Iterable[str]
) -> list[str]:
    return [
        feature
        for feature in expected_features
        if feature not in INFERENCE_COMPUTED_FEATURES and feature not in df.columns
    ]


def _select_feature_table_for_inference(
    season: int, expected_features: tuple[str, ...]
) -> tuple[pd.DataFrame, int, str | None]:
    available = list_available_feature_seasons()
    print(
        f"[matchup] candidate feature seasons={available} requested_season={season}",
        flush=True,
    )
    if season not in available:
        raise FileNotFoundError(
            f"Features parquet is missing for season={season}. "
            "Run `make features` for that season."
        )

    candidate_df = load_features_table(season, expected_features)
    selected_missing = _missing_expected_columns(candidate_df, expected_features)
    print(
        f"[matchup] chosen feature season={season} missing={selected_missing}",
        flush=True,
    )
    return candidate_df, season, None


def _normalize_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    team_col = _resolve_team_column(df)
    if team_col and "team_abbr" not in df.columns:
        df["team_abbr"] = df[team_col]

    if "team_abbr" in df.columns:
        df["team_abbr"] = (
            df["team_abbr"]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({"None": None, "nan": None, "NaN": None, "NULL": None, "": None})
        )

    date_col = _resolve_date_column(df)
    if date_col:
        df["_feature_date"] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def _apply_feature_aliases_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    cols = set(normalized.columns)
    for canonical, aliases in FEATURE_ALIASES.items():
        if canonical in cols:
            continue
        for alias in aliases:
            if alias in cols:
                normalized = normalized.rename(columns={alias: canonical})
                cols = set(normalized.columns)
                break
    return normalized


def _resolve_team_column(df: pd.DataFrame) -> str | None:
    for col in [
        "team_abbr",
        "team_abbrev",
        "team_abbreviation",
        "team_code",
        "team",
    ]:
        if col in df.columns:
            return col
    return None


def _resolve_date_column(df: pd.DataFrame) -> str | None:
    for col in ["feature_date", "game_date", "date", "asof_date"]:
        if col in df.columns:
            return col
    return None


@st.cache_resource(show_spinner=False)
def load_ridge_model(model_path: Path):
    return joblib.load(model_path)


def _load_metrics_feature_list(metrics_path: Path) -> list[str]:
    try:
        payload = metrics_path.read_text(encoding="utf-8")
        data = json.loads(payload)
    except Exception:
        return []

    if isinstance(data, dict):
        cols = data.get("feature_columns")
        if isinstance(cols, list):
            return [str(c) for c in cols]
    return []


def _normalize_team(team: str) -> str:
    return str(team or "").strip().upper()


def _resolve_feature_order(
    model: Any, metrics_path: Path, df: pd.DataFrame
) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in getattr(model, "feature_names_in_")]

    metrics_cols = _load_metrics_feature_list(metrics_path)
    if metrics_cols:
        return metrics_cols

    if df is None or df.empty:
        return []

    candidate = df.drop(
        columns=[c for c in df.columns if c in IDENTIFIER_COLUMNS | OUTCOME_COLUMNS],
        errors="ignore",
    )
    numeric = candidate.select_dtypes(include=["number", "bool"]).copy()
    return [str(c) for c in numeric.columns]


def _latest_team_row(features_df: pd.DataFrame, team_id: int) -> pd.Series | None:
    if features_df is None or features_df.empty:
        return None
    if "team_id" not in features_df.columns:
        return None
    subset = features_df[pd.to_numeric(features_df["team_id"], errors="coerce") == team_id]
    if subset.empty:
        print(
            f"[matchup] team not found in features team_id={team_id}",
            flush=True,
        )
        return None
    if "_feature_date" in subset.columns:
        subset = subset.sort_values(["_feature_date", "game_id"], ascending=True)
    elif "game_date" in subset.columns:
        subset = subset.sort_values(["game_date", "game_id"], ascending=True)
    return subset.tail(1).iloc[0]


def _build_feature_vector(
    row: pd.Series | None, feature_order: Iterable[str]
) -> tuple[pd.Series, list[str], dict[str, str]]:
    missing: list[str] = []
    sources: dict[str, str] = {}
    if row is None:
        return pd.Series({f: 0.0 for f in feature_order}), list(feature_order), sources
    data: dict[str, float] = {}
    lower_index = {str(c).lower(): c for c in row.index}
    for feature in feature_order:
        column_name = feature if feature in row.index else None
        if column_name is None:
            lower_match = lower_index.get(str(feature).lower())
            if lower_match is not None:
                column_name = lower_match
        if column_name is None:
            for alt in FEATURE_ALIASES.get(feature, []):
                alt_match = alt if alt in row.index else lower_index.get(alt.lower())
                if alt_match is not None:
                    column_name = alt_match
                    break
        if column_name is None:
            data[feature] = 0.0
            missing.append(feature)
            continue
        value = pd.to_numeric(row[column_name], errors="coerce")
        data[feature] = float(value) if pd.notna(value) else 0.0
        sources[feature] = column_name
    return pd.Series(data), missing, sources

def compute_team_form(df: pd.DataFrame, team_id: int, window: int) -> tuple[dict, list[dict]]:
    if df is None or df.empty:
        summary = {
            "avg_pts_for": None,
            "avg_pts_against": None,
            "avg_margin": None,
            "win_pct": None,
            "games_count": 0,
        }
        return summary, []
    team_id_series = pd.to_numeric(df.get("team_id"), errors="coerce")
    team_df = df[team_id_series == team_id].sort_values("game_date").tail(window).copy()
    if team_df.empty:
        summary = {
            "avg_pts_for": None,
            "avg_pts_against": None,
            "avg_margin": None,
            "win_pct": None,
            "games_count": 0,
        }
        return summary, []

    pts_for = team_df["pts"].astype(float)
    plus_minus = team_df["plus_minus"].astype(float)
    pts_against = pts_for - plus_minus

    summary = {
        "avg_pts_for": round(float(pts_for.mean()), 2),
        "avg_pts_against": round(float(pts_against.mean()), 2),
        "avg_margin": round(float(plus_minus.mean()), 2),
        "win_pct": round(float((team_df["wl"] == "W").mean()), 3),
        "games_count": int(len(team_df)),
    }

    recent_cols = ["game_date", "opp_team_abbr", "is_home", "pts", "plus_minus", "wl"]
    recent = team_df.tail(10)[recent_cols].copy()
    recent["game_date"] = recent["game_date"].dt.strftime("%Y-%m-%d")
    recent_samples = recent.to_dict(orient="records")
    return summary, recent_samples


def build_model_prediction(
    *,
    model: Any,
    metrics_path: Path,
    season: int,
    team_home_id: int,
    team_away_id: int,
    team_home_label: str,
    team_away_label: str,
) -> tuple[dict, pd.DataFrame, list[str], int, str | None, pd.Series, pd.Series, pd.Series]:
    feature_order = _resolve_feature_order(model, metrics_path, pd.DataFrame())
    features_df, feature_season_used, feature_notice = _select_feature_table_for_inference(
        season, tuple(feature_order)
    )
    features_df = _apply_feature_aliases_to_columns(features_df)
    if not feature_order:
        feature_order = _resolve_feature_order(model, metrics_path, features_df)
    if not feature_order:
        raise ValueError("No feature columns resolved for ridge model input.")

    home_row = _latest_team_row(features_df, team_home_id)
    away_row = _latest_team_row(features_df, team_away_id)

    home_vec, missing_home, home_sources = _build_feature_vector(home_row, feature_order)
    away_vec, missing_away, away_sources = _build_feature_vector(away_row, feature_order)

    if "is_home" in feature_order:
        home_vec["is_home"] = 1.0
        away_vec["is_home"] = 0.0
        if "is_home" in missing_home:
            missing_home.remove("is_home")
        if "is_home" in missing_away:
            missing_away.remove("is_home")
        home_sources["is_home"] = "matchup_override"
        away_sources["is_home"] = "matchup_override"

    missing_features = sorted(set(missing_home + missing_away))

    print(
        f"[matchup] expected_features={feature_order} available_columns={list(features_df.columns)[:30]}",
        flush=True,
    )
    print(
        f"[matchup] team_lookup team_home_id={team_home_id} "
        f"team_away_id={team_away_id} "
        f"home_found={home_row is not None} away_found={away_row is not None}",
        flush=True,
    )
    if missing_features:
        print(f"[matchup] missing_features={missing_features}", flush=True)
        if home_row is None or away_row is None:
            print(
                f"[matchup] team_abbrs_available_sample={features_df.get('team_abbr', pd.Series()).dropna().unique()[:10]}",
                flush=True,
            )
        else:
            missing_sources = {
                name: {
                    "home": home_sources.get(name),
                    "away": away_sources.get(name),
                }
                for name in missing_features
            }
            print(f"[matchup] missing_sources={missing_sources}", flush=True)

    diff = home_vec - away_vec

    input_df = pd.DataFrame([diff], columns=feature_order)
    pred_margin = float(model.predict(input_df)[0])
    favored_team = team_home_label if pred_margin >= 0 else team_away_label

    ridge = model.named_steps.get("ridge", model)
    coefs = getattr(ridge, "coef_", None)
    if coefs is None or len(coefs) != len(feature_order):
        raise ValueError("Ridge coefficients unavailable or misaligned with features.")

    contributions = pd.Series(coefs, index=feature_order) * diff
    drivers = pd.DataFrame(
        {
            "feature": feature_order,
            "home": home_vec.values,
            "away": away_vec.values,
            "diff": diff.values,
            "contribution": contributions.values,
        }
    )
    drivers["direction"] = drivers["contribution"].apply(
        lambda v: "home" if v > 0 else ("away" if v < 0 else "neutral")
    )
    drivers = drivers.reindex(
        drivers["contribution"].abs().sort_values(ascending=False).index
    )

    prediction = {
        "favored_team": favored_team,
        "pred_margin": round(pred_margin, 2),
    }
    if feature_notice:
        print(f"[matchup] feature fallback notice {feature_notice}", flush=True)
    return (
        prediction,
        drivers,
        missing_features,
        feature_season_used,
        feature_notice,
        home_vec,
        away_vec,
        diff,
    )


def _render_prediction_summary(
    *,
    feature_season: int,
    favored_team_label: str,
    predicted_margin: float,
    drivers_df: pd.DataFrame,
) -> None:
    st.subheader("Prediction")
    st.caption(f"Feature season: {feature_season}")
    st.metric("Favored team", favored_team_label)
    st.metric("Predicted margin", f"{predicted_margin:+.2f}")

    st.subheader("Top Drivers")
    top_n = min(12, max(8, len(drivers_df)))
    st.dataframe(
        drivers_df.head(top_n),
        use_container_width=True,
    )


def render_matchup(
    *,
    seasons: list[int],
    season_error: Exception | None,
    load_error: Exception | None,
    selected_season: int | str,
    games_df: pd.DataFrame | None,
    df_season: pd.DataFrame | None,
    teams: list[int],
    team_labels: dict[int, str],
    team_options_error: Exception | None,
    team_a: int | None,
    team_b: int | None,
    team_a_is_home: bool,
    window: int,
    predict: bool,
    matchup_signature: tuple[int, int, int | None, int | None],
) -> dict | None:
    prediction: dict[str, Any] | None = None
    why_factors: list[dict] = []
    team_a_form: dict[str, Any] | None = None
    team_b_form: dict[str, Any] | None = None
    team_a_recent: list[dict] = []
    team_b_recent: list[dict] = []

    if season_error is not None:
        render_exception(season_error)
    if team_options_error is not None:
        st.error(
            f"Features parquet is unavailable for selected season {selected_season}. "
            "Predict is disabled until season features are present."
        )
        st.exception(team_options_error)
    if load_error is not None:
        st.error("Failed to load games parquet.")
        st.exception(load_error)
    if not seasons:
        st.warning("No seasons available yet. Load silver games to continue.")
    elif not teams:
        st.warning("No teams found for this season.")
    else:
        stored_result = st.session_state.get("last_matchup_result")
        stored_signature = (
            tuple(stored_result.get("signature"))
            if isinstance(stored_result, dict) and "signature" in stored_result
            else None
        )

        if predict and (team_a is None or team_b is None):
            st.warning("Select teams to compute a matchup.")
        elif predict and team_a == team_b:
            st.warning("Pick two different teams to compute a matchup.")
        elif predict:
            team_home_id = team_a if team_a_is_home else team_b
            team_away_id = team_b if team_a_is_home else team_a
            assert team_home_id is not None and team_away_id is not None
            team_a_label = team_labels.get(team_a, f"Team {team_a}")
            team_b_label = team_labels.get(team_b, f"Team {team_b}")
            team_home_label = team_a_label if team_a_is_home else team_b_label
            team_away_label = team_b_label if team_a_is_home else team_a_label

            team_a_form, team_a_recent = compute_team_form(df_season, team_a, window)
            team_b_form, team_b_recent = compute_team_form(df_season, team_b, window)

            model_path = Path("models") / "artifacts" / "ridge_margin.joblib"
            metrics_path = Path("models") / "artifacts" / "ridge_margin_metrics.json"

            if not model_path.exists():
                st.info(
                    "Ridge model not found. Train it with "
                    "`.venv/bin/python -m models.train_ridge --train_seasons 2022 2023 2024 "
                    "--test_seasons 2025`, then place the artifact at "
                    "`models/artifacts/ridge_margin.joblib`."
                )
            else:
                try:
                    model = load_ridge_model(model_path)
                    (
                        prediction,
                        drivers,
                        missing_features,
                        feature_season_used,
                        feature_notice,
                        home_vec,
                        away_vec,
                        diff_vec,
                    ) = build_model_prediction(
                        model=model,
                        metrics_path=metrics_path,
                        season=int(selected_season),
                        team_home_id=team_home_id,
                        team_away_id=team_away_id,
                        team_home_label=team_home_label,
                        team_away_label=team_away_label,
                    )
                    st.session_state["matchup_context"] = build_matchup_context(
                        season=int(selected_season),
                        window_n=int(window),
                        home_team_id=int(team_home_id),
                        away_team_id=int(team_away_id),
                        home_team_label=team_home_label,
                        away_team_label=team_away_label,
                        home_features=home_vec,
                        away_features=away_vec,
                        diff_features=diff_vec,
                        predicted_margin=float(prediction["pred_margin"]),
                        top_drivers=drivers,
                    )
                    st.session_state["last_matchup_result"] = {
                        "signature": list(matchup_signature),
                        "feature_season": int(feature_season_used),
                        "favored_team_label": str(prediction["favored_team"]),
                        "predicted_margin": float(prediction["pred_margin"]),
                        "top_drivers": drivers.to_dict(orient="records"),
                        "home_team_label": team_home_label,
                        "away_team_label": team_away_label,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if feature_notice:
                        st.info(feature_notice)

                    _render_prediction_summary(
                        feature_season=feature_season_used,
                        favored_team_label=str(prediction["favored_team"]),
                        predicted_margin=float(prediction["pred_margin"]),
                        drivers_df=drivers,
                    )

                    if missing_features:
                        st.warning(
                            "Missing features filled with 0: "
                            + ", ".join(missing_features[:8])
                            + ("…" if len(missing_features) > 8 else "")
                        )
                except Exception as err:
                    st.session_state.pop("matchup_context", None)
                    st.session_state.pop("last_matchup_result", None)
                    st.error("Failed to generate ridge prediction.")
                    st.exception(err)
        elif stored_signature == matchup_signature:
            try:
                stored_drivers_df = pd.DataFrame(stored_result.get("top_drivers") or [])
                _render_prediction_summary(
                    feature_season=int(stored_result.get("feature_season")),
                    favored_team_label=str(stored_result.get("favored_team_label") or ""),
                    predicted_margin=float(stored_result.get("predicted_margin") or 0.0),
                    drivers_df=stored_drivers_df,
                )
            except Exception:
                st.session_state.pop("last_matchup_result", None)


    if (
        prediction
        and team_a is not None
        and team_b is not None
        and team_a_form
        and team_b_form
    ):
        team_a_label = team_labels.get(team_a, f"Team {team_a}")
        team_b_label = team_labels.get(team_b, f"Team {team_b}")
        return {
            "inputs": {
                "season": int(selected_season),
                "window": window,
                "team_a": team_a_label,
                "team_b": team_b_label,
                "team_a_id": int(team_a),
                "team_b_id": int(team_b),
                "team_a_is_home": team_a_is_home,
            },
            "prediction": prediction,
            "why_factors": why_factors,
            "team_summaries": {
                team_a_label: team_a_form,
                team_b_label: team_b_form,
            },
            "recent_games_samples": {
                team_a_label: team_a_recent,
                team_b_label: team_b_recent,
            },
        }

    return None
