"""Streamlit page for exploring matchup edges and model predictions."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from azure_blob import (
    lake_prefix,
    list_blobs,
    parse_available_seasons,
    read_parquet_from_blob,
    render_exception,
)
from llm_chat import chat_with_context
from settings import get_setting


@st.cache_data(show_spinner=False)
def load_games(season: int) -> pd.DataFrame:
    prefix = lake_prefix()
    blob_name = f"{prefix}silver/games/season={season}/games.parquet"
    df = read_parquet_from_blob(blob_name)
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


def compute_team_form(df: pd.DataFrame, team: str, window: int) -> tuple[dict, list[dict]]:
    team_df = df[df["team_abbr"] == team].sort_values("game_date").tail(window).copy()
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


def build_prediction(
    team_a: str,
    team_b: str,
    team_a_form: dict,
    team_b_form: dict,
    team_a_is_home: bool,
) -> tuple[dict, list[dict]]:
    def coalesce_number(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if isinstance(value, float) and math.isnan(value):
                return default
        except Exception:
            return default
        try:
            return float(value)
        except Exception:
            return default

    a_margin = coalesce_number(team_a_form.get("avg_margin"))
    b_margin = coalesce_number(team_b_form.get("avg_margin"))
    base_margin = a_margin - b_margin
    home_adv = 2.5 if team_a_is_home else -2.5
    pred_margin = base_margin + home_adv

    a_pts_for = coalesce_number(team_a_form.get("avg_pts_for"))
    b_pts_for = coalesce_number(team_b_form.get("avg_pts_for"))
    a_pts_against = coalesce_number(team_a_form.get("avg_pts_against"))
    b_pts_against = coalesce_number(team_b_form.get("avg_pts_against"))
    win_gap = coalesce_number(team_a_form.get("win_pct")) - coalesce_number(
        team_b_form.get("win_pct")
    )

    favored_team = team_a if pred_margin >= 0 else team_b
    prob_team_a = 1.0 / (1.0 + math.exp(-pred_margin / 7.0))
    win_prob = prob_team_a if favored_team == team_a else 1.0 - prob_team_a

    why_factors = [
        {
            "name": "Recent average margin",
            "contribution_points": round(base_margin, 2),
            "direction": f"toward {team_a if base_margin >= 0 else team_b}",
            "short_explanation": (
                f"{team_a} avg margin {a_margin:+.2f} vs {team_b} {b_margin:+.2f}."
            ),
        },
        {
            "name": "Home court",
            "contribution_points": round(home_adv, 2),
            "direction": f"toward {team_a if team_a_is_home else team_b}",
            "short_explanation": (
                f"Home edge applied to {team_a if team_a_is_home else team_b}."
            ),
        },
        {
            "name": "Recent win rate gap",
            "contribution_points": round(win_gap * 10.0, 2),
            "direction": f"toward {team_a if win_gap >= 0 else team_b}",
            "short_explanation": (
                f"Win pct {team_a} {team_a_form.get('win_pct')} vs {team_b} "
                f"{team_b_form.get('win_pct')}."
            ),
        },
        {
            "name": "Scoring edge",
            "contribution_points": round((a_pts_for - b_pts_for) * 0.3, 2),
            "direction": f"toward {team_a if a_pts_for >= b_pts_for else team_b}",
            "short_explanation": (
                f"Avg pts for {team_a} {a_pts_for:.2f} vs {team_b} {b_pts_for:.2f}."
            ),
        },
        {
            "name": "Defense edge",
            "contribution_points": round((b_pts_against - a_pts_against) * 0.3, 2),
            "direction": f"toward {team_a if a_pts_against <= b_pts_against else team_b}",
            "short_explanation": (
                f"Avg pts against {team_a} {a_pts_against:.2f} vs {team_b} "
                f"{b_pts_against:.2f}."
            ),
        },
    ]

    prediction = {
        "favored_team": favored_team,
        "pred_margin": round(float(pred_margin), 2),
        "win_prob": round(float(win_prob), 3),
    }
    return prediction, why_factors


def init_chat_state(context_key: str) -> None:
    if st.session_state.get("chat_context_key") != context_key:
        st.session_state["chat_context_key"] = context_key
        st.session_state["chat_messages"] = []
        st.session_state.pop("pending_user_message", None)


def render_chat_panel(context_packet: dict | None, team_a: str, team_b: str) -> None:
    st.subheader("Ask the Model")
    st.caption("Answers are grounded in the current matchup's recent-form data and prediction.")
    api_key = str(get_setting("GEMINI_API_KEY", "")).strip()

    if not api_key:
        st.info("Chat unavailable (API key not configured).")
        return
    if context_packet is None:
        st.info("Select teams and click Predict to enable chat context.")
        return
    if context_packet is None:
        return

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    chip_cols = st.columns(2)
    if chip_cols[0].button(f"Why is {team_a} favored?"):
        st.session_state["pending_user_message"] = f"Why is {team_a} favored?"
    if chip_cols[1].button("What are the top 3 factors?"):
        st.session_state["pending_user_message"] = "What are the top 3 factors?"
    if chip_cols[0].button("How much did home court matter?"):
        st.session_state["pending_user_message"] = "How much did home court matter?"
    if chip_cols[1].button("Summarize both teams' last 10 games form."):
        st.session_state["pending_user_message"] = "Summarize both teams' last 10 games form."

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    pending = st.session_state.pop("pending_user_message", None)
    user_input = st.chat_input("Ask a question about this matchup")
    if pending and not user_input:
        user_input = pending

    if not user_input:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_context(st.session_state["chat_messages"], context_packet)
        st.write(response)

    st.session_state["chat_messages"].append({"role": "assistant", "content": response})


def main() -> None:
    """Entry point for matchup exploration."""
    st.title("Matchup Explorer")
    st.caption("Baseline, last-N form model with home/away adjustment.")

    try:
        blobs = list_blobs(f"{lake_prefix()}silver/games/")
        seasons = parse_available_seasons(blobs)
    except Exception as exc:
        render_exception(exc)
        seasons = []

    if not seasons:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.warning("No seasons available yet. Load silver games to continue.")
        with col_right:
            render_chat_panel(None, "", "")
        return

    selected_season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    selected_season = int(selected_season)
    window = st.slider("Recent form window (games)", min_value=3, max_value=20, value=10)

    col_left, col_right = st.columns([2, 1])

    prefix = lake_prefix()
    blob_name = f"{prefix}silver/games/season={selected_season}/games.parquet"
    blob_matches: list[dict] = []
    games_df: pd.DataFrame | None = None
    load_error: Exception | None = None

    try:
        blob_matches = list_blobs(blob_name)
    except Exception:
        blob_matches = []

    try:
        games_df = load_games(selected_season)
    except Exception as exc:
        load_error = exc
        with col_left:
            st.error("Failed to load games parquet.")
            st.exception(exc)
        games_df = None

    df_season: pd.DataFrame | None = None
    if games_df is not None and "season" in games_df.columns:
        df_season = games_df[games_df["season"] == selected_season]
    else:
        df_season = games_df

    with st.expander("Debug", expanded=False):
        st.write({"selected_season": selected_season, "type": type(selected_season)})
        st.write("df.shape:", games_df.shape if games_df is not None else None)
        st.write("df_season.shape:", df_season.shape if df_season is not None else None)
        if df_season is not None and "team_abbr" in df_season.columns:
            st.write(
                "Distinct team_abbr (filtered):",
                int(df_season["team_abbr"].dropna().nunique()),
            )
            st.write(
                "team_abbr sample:",
                sorted(df_season["team_abbr"].dropna().unique().tolist())[:10],
            )
        else:
            st.write("Distinct team_abbr (filtered):", None)
            st.write("team_abbr sample:", None)

    teams: list[str] = []
    if df_season is not None and "team_abbr" in df_season.columns:
        teams = sorted(df_season["team_abbr"].dropna().unique().tolist())

    team_a = teams[0] if teams else ""
    team_b = teams[1] if len(teams) > 1 else ""
    team_a_is_home = True
    prediction: dict[str, Any] | None = None
    why_factors: list[dict] = []
    team_a_form: dict[str, Any] | None = None
    team_b_form: dict[str, Any] | None = None
    team_a_recent: list[dict] = []
    team_b_recent: list[dict] = []

    with col_left:
        if load_error is not None:
            st.warning("Fix the load error above to view matchup details.")
        elif not teams:
            st.warning("No teams found for this season.")
        else:
            team_a = st.selectbox("Team A", teams, index=0)
            team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)
            team_a_is_home = st.toggle("Team A is home", value=True)
            predict = st.button("Predict")

            if predict and team_a == team_b:
                st.warning("Pick two different teams to compute a matchup.")
            elif predict:
                team_a_form, team_a_recent = compute_team_form(df_season, team_a, window)
                team_b_form, team_b_recent = compute_team_form(df_season, team_b, window)

                prediction, why_factors = build_prediction(
                    team_a, team_b, team_a_form, team_b_form, team_a_is_home
                )

                st.subheader("Prediction")
                st.metric("Favored team", prediction["favored_team"])
                st.metric("Predicted margin", f"{prediction['pred_margin']:+.2f}")
                st.metric("Win probability", f"{prediction['win_prob'] * 100:.1f}%")

                st.subheader("Why (factors)")
                st.dataframe(pd.DataFrame(why_factors), use_container_width=True)

                st.subheader("Recent form summary")
                summary_df = pd.DataFrame(
                    {
                        team_a: team_a_form,
                        team_b: team_b_form,
                    }
                ).T
                st.dataframe(summary_df, use_container_width=True)

                st.subheader("Recent games sample")
                sample_tabs = st.tabs([team_a, team_b])
                with sample_tabs[0]:
                    st.dataframe(pd.DataFrame(team_a_recent), use_container_width=True)
                with sample_tabs[1]:
                    st.dataframe(pd.DataFrame(team_b_recent), use_container_width=True)

    context_packet: dict[str, Any] | None = None
    if prediction and team_a and team_b and team_a_form and team_b_form:
        context_packet = {
            "inputs": {
                "season": selected_season,
                "window": window,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_is_home": team_a_is_home,
            },
            "prediction": prediction,
            "why_factors": why_factors,
            "team_summaries": {
                team_a: team_a_form,
                team_b: team_b_form,
            },
            "recent_games_samples": {
                team_a: team_a_recent,
                team_b: team_b_recent,
            },
        }

    context_key = f"{selected_season}-{window}-{team_a}-{team_b}-{team_a_is_home}"
    init_chat_state(context_key)

    with col_right:
        render_chat_panel(context_packet, team_a, team_b)


if __name__ == "__main__":
    main()
