"""Streamlit UI for browsing silver datasets in Azure Blob Storage."""

from __future__ import annotations

import streamlit as st

from azure_blob import (
    lake_prefix,
    list_blobs,
    parse_available_odds_dates,
    parse_available_seasons,
    read_parquet_from_blob,
    render_exception,
)


st.set_page_config(page_title="NBA Edge", layout="wide")

st.title("NBA Edge - Silver Data Browser")

prefix = lake_prefix()

games_prefix = f"{prefix}silver/games/"
odds_prefix = f"{prefix}silver/odds_spreads/"


tab_overview, tab_games, tab_odds, tab_model, tab_predictions = st.tabs(
    ["Overview", "Games", "Odds", "Model", "Predictions"]
)

with tab_overview:
    st.subheader("Available Silver Datasets")

    col_games, col_odds = st.columns(2)

    with col_games:
        st.markdown("**Games blobs**")
        try:
            game_blobs = list_blobs(games_prefix)
            seasons = parse_available_seasons(game_blobs)
            st.write("Seasons:", seasons if seasons else "None")
            st.dataframe(game_blobs, use_container_width=True)
        except Exception as exc:
            render_exception(exc)
            game_blobs = []

    with col_odds:
        st.markdown("**Odds spreads blobs**")
        try:
            odds_blobs = list_blobs(odds_prefix)
            dates = parse_available_odds_dates(odds_blobs)
            st.write("Dates:", dates if dates else "None")
            st.dataframe(odds_blobs, use_container_width=True)
        except Exception as exc:
            render_exception(exc)
            odds_blobs = []

with tab_games:
    st.subheader("Silver Games")
    try:
        game_blobs = list_blobs(games_prefix)
        seasons = parse_available_seasons(game_blobs)
        if not seasons:
            st.warning("No game seasons found.")
        else:
            selected_season = st.selectbox("Season", seasons, index=len(seasons) - 1)
            blob_name = f"{games_prefix}season={selected_season}/games.parquet"
            df = read_parquet_from_blob(blob_name)
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
            with st.expander("Columns"):
                st.write(list(df.columns))
            st.dataframe(df.head(200), use_container_width=True)
    except Exception as exc:
        render_exception(exc)

with tab_odds:
    st.subheader("Silver Odds Spreads")
    try:
        odds_blobs = list_blobs(odds_prefix)
        dates = parse_available_odds_dates(odds_blobs)
        if not dates:
            st.warning("No odds dates found.")
        else:
            selected_date = st.selectbox("Snapshot date", dates, index=len(dates) - 1)
            blob_name = f"{odds_prefix}date={selected_date}/odds.parquet"
            df = read_parquet_from_blob(blob_name)
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
            with st.expander("Columns"):
                st.write(list(df.columns))
            st.dataframe(df.head(200), use_container_width=True)
    except Exception as exc:
        render_exception(exc)

with tab_model:
    st.subheader("Model Artifacts")
    st.write("Expected prefix:")
    st.code("lake/models/margin_model/")

with tab_predictions:
    st.subheader("Predictions")
    st.write("Expected prefix:")
    st.code("lake/reports/predictions/")
