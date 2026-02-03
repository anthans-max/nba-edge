"""Status UI rendering."""

from __future__ import annotations

import streamlit as st

from azure_blob import (
    lake_prefix,
    list_blobs,
    parse_available_odds_dates,
    parse_available_seasons,
    render_exception,
)


def render_status() -> None:
    st.subheader("Data status")
    prefix = lake_prefix()
    games_prefix = f"{prefix}silver/games/"
    odds_prefix = f"{prefix}silver/odds_spreads/"

    col_games, col_odds = st.columns(2)

    with col_games:
        st.markdown("**Games (silver)**")
        try:
            game_blobs = list_blobs(games_prefix)
            seasons = parse_available_seasons(game_blobs)
            st.metric("Blobs", len(game_blobs))
            st.write("Seasons:", seasons if seasons else "None")
        except Exception as exc:
            render_exception(exc)

    with col_odds:
        st.markdown("**Odds spreads (silver)**")
        try:
            odds_blobs = list_blobs(odds_prefix)
            dates = parse_available_odds_dates(odds_blobs)
            st.metric("Blobs", len(odds_blobs))
            st.write("Dates:", dates if dates else "None")
        except Exception as exc:
            render_exception(exc)
