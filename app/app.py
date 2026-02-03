"""Streamlit UI for NBA Edge V1."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pandas as pd
import streamlit as st

from azure_blob import lake_prefix, list_blobs, parse_available_seasons
from ui.chat import init_chat_state, render_chat
from ui.matchup import load_games, render_matchup
from ui.status import render_status


st.set_page_config(layout="wide", page_title="NBA Edge (V1)", page_icon="🏀")


def main() -> None:
    st.title("NBA Edge")
    st.caption("V1 matchup explorer with chat and system status.")

    seasons: list[int] = []
    season_error: Exception | None = None
    try:
        blobs = list_blobs(f"{lake_prefix()}silver/games/")
        seasons = parse_available_seasons(blobs)
    except Exception as exc:
        season_error = exc

    with st.sidebar:
        st.subheader("Settings")
        if season_error is not None:
            st.warning("Failed to load seasons.")
        selected_season = st.selectbox(
            "Season",
            seasons,
            index=len(seasons) - 1 if seasons else 0,
            disabled=not seasons,
        )
        window = st.slider(
            "Recent form window (games)",
            min_value=3,
            max_value=20,
            value=10,
            disabled=not seasons,
        )

    games_df: pd.DataFrame | None = None
    load_error: Exception | None = None
    if seasons:
        try:
            games_df = load_games(int(selected_season))
        except Exception as exc:
            load_error = exc

    df_season: pd.DataFrame | None = None
    if games_df is not None and "season" in games_df.columns:
        df_season = games_df[games_df["season"] == int(selected_season)]
    else:
        df_season = games_df

    teams: list[str] = []
    if df_season is not None and "team_abbr" in df_season.columns:
        teams = sorted(df_season["team_abbr"].dropna().unique().tolist())

    with st.sidebar:
        st.subheader("Matchup")
        team_a = st.selectbox("Team A", teams, index=0 if teams else 0, disabled=not teams)
        team_b = st.selectbox(
            "Team B",
            teams,
            index=1 if len(teams) > 1 else 0,
            disabled=not teams,
        )
        team_a_is_home = st.toggle("Team A is home", value=True, disabled=not teams)
        with st.expander("Options"):
            show_factors = st.checkbox("Show factor table", value=True)
            show_summaries = st.checkbox("Show team summaries", value=True)
            show_recent = st.checkbox("Show recent games", value=True)
        predict = st.button("Predict", type="primary", disabled=not teams)

        with st.expander("Runtime"):
            try:
                repo_root = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )
            except Exception:
                repo_root = "unknown"
            st.code(
                f"repo_root = {repo_root}\n"
                f"__file__ = {__file__}\n"
                f"cwd = {os.getcwd()}\n",
                language="text",
            )

    tab_matchup, tab_chat, tab_status = st.tabs(["Matchup", "Chat", "Status"])

    context_packet: dict[str, Any] | None = None
    with tab_matchup:
        context_packet = render_matchup(
            seasons=seasons,
            season_error=season_error,
            load_error=load_error,
            selected_season=selected_season,
            games_df=games_df,
            df_season=df_season,
            teams=teams,
            team_a=team_a,
            team_b=team_b,
            team_a_is_home=team_a_is_home,
            window=window,
            show_factors=show_factors,
            show_summaries=show_summaries,
            show_recent=show_recent,
            predict=predict,
        )

    st.session_state["matchup_context"] = context_packet

    context_key = f"{selected_season}-{window}-{team_a}-{team_b}-{team_a_is_home}"
    init_chat_state(context_key)

    with tab_chat:
        render_chat(st.session_state.get("matchup_context"), team_a, team_b)

    with tab_status:
        render_status()


if __name__ == "__main__":
    main()
