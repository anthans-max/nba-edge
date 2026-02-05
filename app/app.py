"""Streamlit UI for NBA Edge V1."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st

from ui.chat import init_chat_state, render_chat
from ui.matchup import (
    list_available_feature_seasons,
    list_feature_teams,
    load_games,
    render_matchup,
)
from ui.status import render_status


st.set_page_config(layout="wide", page_title="NBA Edge (V1)", page_icon="🏀")


def main() -> None:
    st.title("NBA Edge")
    st.caption("V1 matchup explorer with chat and data snapshot.")

    def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        if df is None or df.empty:
            return None
        cols = {str(col).lower(): str(col) for col in df.columns}
        for candidate in candidates:
            key = candidate.lower()
            if key in cols:
                return cols[key]
        return None

    def normalize_season_value(value: Any) -> int | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            match = re.match(r"^(\d{4})", raw)
            if match:
                return int(match.group(1))
            if raw.isdigit():
                return int(raw)
            try:
                return int(float(raw))
            except Exception:
                return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            try:
                return int(value)
            except Exception:
                return None
        return None

    feature_seasons: list[int] = []
    season_error: Exception | None = None
    try:
        feature_seasons = list_available_feature_seasons()
    except Exception as exc:
        season_error = exc

    selected_season = st.session_state.get("selected_season")

    window = int(st.session_state.get("window", 10))

    games_df: pd.DataFrame | None = None
    load_error: Exception | None = None
    if selected_season is not None:
        try:
            games_df = load_games(int(selected_season))
        except Exception as exc:
            load_error = exc

    season_col = detect_column(
        games_df, ["season", "season_year", "season_id", "seasonyear", "seasonid"]
    )
    season_values: list[int] = []
    if games_df is not None and season_col is not None:
        season_series = games_df[season_col].map(normalize_season_value)
        season_values = sorted(season_series.dropna().unique().tolist())
        games_df = games_df.copy()
        games_df["_season_norm"] = season_series

    seasons_for_dropdown = feature_seasons
    default_season = seasons_for_dropdown[-1] if seasons_for_dropdown else None
    if selected_season not in seasons_for_dropdown:
        selected_season = default_season

    with st.sidebar:
        st.subheader("Settings")
        if season_error is not None:
            st.warning("Failed to load seasons.")
        selected_season = st.selectbox(
            "Season",
            seasons_for_dropdown,
            index=seasons_for_dropdown.index(selected_season)
            if seasons_for_dropdown and selected_season in seasons_for_dropdown
            else 0,
            disabled=not seasons_for_dropdown,
            format_func=lambda season: f"{season}\u2013{season + 1}",
        )
        if selected_season is not None:
            st.session_state["selected_season"] = selected_season
        window = st.slider(
            "Recent form window (games)",
            min_value=3,
            max_value=20,
            value=window,
            disabled=not seasons_for_dropdown,
            key="window",
        )

    df_season: pd.DataFrame | None = None
    selected_season_norm = normalize_season_value(selected_season)
    if games_df is not None and season_col is not None and selected_season_norm is not None:
        df_season = games_df[games_df["_season_norm"] == selected_season_norm]
    else:
        df_season = games_df

    teams: list[int] = []
    team_labels: dict[int, str] = {}
    team_options_error: Exception | None = None
    if selected_season is not None:
        try:
            teams, team_labels = list_feature_teams(int(selected_season))
        except Exception as exc:
            team_options_error = exc

    def team_label(team_id: int | None) -> str:
        if team_id is None:
            return ""
        return team_labels.get(team_id, f"Team {team_id}")

    default_team_a = teams[0] if teams else None
    default_team_b = teams[1] if len(teams) > 1 else default_team_a

    with st.sidebar:
        st.subheader("Matchup")
        if team_options_error is not None and selected_season is not None:
            st.error(
                f"Features parquet missing or invalid for season {selected_season}. "
                "Run `make features` for this season."
            )
        if teams:
            team_a = st.selectbox(
                "Team A",
                teams,
                index=teams.index(default_team_a) if default_team_a in teams else 0,
                format_func=team_label,
            )
            team_b = st.selectbox(
                "Team B",
                teams,
                index=teams.index(default_team_b) if default_team_b in teams else 0,
                format_func=team_label,
            )
        else:
            st.selectbox("Team A", [""], index=0, disabled=True)
            st.selectbox("Team B", [""], index=0, disabled=True)
            team_a = None
            team_b = None
        team_a_is_home = st.toggle("Team A is home", value=True, disabled=not teams)
        predict = st.button(
            "Predict",
            type="primary",
            disabled=(not teams) or (team_options_error is not None),
        )
    st.session_state["team_a_id"] = team_a
    st.session_state["team_b_id"] = team_b

    tab_matchup, tab_chat, tab_status = st.tabs(["Matchup", "Chat", "Data Snapshot"])

    home_team_id = team_a if team_a_is_home else team_b
    away_team_id = team_b if team_a_is_home else team_a
    season_for_signature = (
        int(selected_season) if selected_season is not None else -1
    )
    current_signature = (season_for_signature, int(window), home_team_id, away_team_id)

    previous_signature = st.session_state.get("matchup_signature")
    if previous_signature != current_signature:
        st.session_state.pop("matchup_ctx", None)
        st.session_state.pop("prediction", None)
        st.session_state.pop("matchup_context", None)
        st.session_state.pop("last_matchup_result", None)
    st.session_state["matchup_signature"] = current_signature

    context_packet: dict[str, Any] | None = None
    with tab_matchup:
        context_packet = render_matchup(
            seasons=feature_seasons,
            season_error=season_error,
            load_error=load_error,
            selected_season=selected_season,
            games_df=games_df,
            df_season=df_season,
            teams=teams,
            team_labels=team_labels,
            team_options_error=team_options_error,
            team_a=team_a,
            team_b=team_b,
            team_a_is_home=team_a_is_home,
            window=window,
            predict=predict,
            matchup_signature=current_signature,
        )

    if context_packet:
        st.session_state["matchup_ctx"] = context_packet
        st.session_state["prediction"] = context_packet.get("prediction")
        if df_season is not None:
            st.session_state["matchup_ctx"]["games_df"] = df_season.copy()

    team_a_label = team_label(team_a) if team_a is not None else "Team A"
    team_b_label = team_label(team_b) if team_b is not None else "Team B"
    context_key = f"{selected_season}-{window}-{team_a}-{team_b}-{team_a_is_home}"
    init_chat_state(context_key)

    with tab_chat:
        render_chat(st.session_state.get("matchup_ctx"), team_a_label, team_b_label)

    with tab_status:
        render_status(
            games_df=games_df,
            df_season=df_season,
            selected_season=selected_season,
            team_a=team_a_label,
            team_b=team_b_label,
            team_a_id=team_a,
            team_b_id=team_b,
            window=window,
            team_a_is_home=team_a_is_home,
        )


if __name__ == "__main__":
    main()
