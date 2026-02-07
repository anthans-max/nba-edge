"""Status UI rendering."""

from __future__ import annotations

from typing import Any
import json
from datetime import datetime, timezone
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from azure_blob import (
    download_blob_bytes,
    get_blob_metadata,
    lake_prefix,
    list_blobs,
    storage_account,
    storage_container,
)
from ui.status_utils import marker_signature, parse_refresh_marker_payload

def _normalize_team(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().upper()


def _app_debug_enabled() -> bool:
    value = os.getenv("APP_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes"}


def env_truthy(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_refresh_marker() -> tuple[dict[str, Any] | None, dict[str, Any] | None, Exception | None]:
    marker_blob = f"{lake_prefix()}_meta/refresh_last_success.json"
    meta: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    err: Exception | None = None
    try:
        meta = get_blob_metadata(marker_blob)
    except Exception as exc:
        err = exc
    if meta is not None:
        try:
            raw = download_blob_bytes(marker_blob)
            payload = parse_refresh_marker_payload(raw)
        except Exception as exc:
            err = err or exc
    return meta, payload, err


def _resolve_app_version() -> str:
    env_sha = (
        os.getenv("GIT_SHA")
        or os.getenv("GITHUB_SHA")
        or os.getenv("SOURCE_VERSION")
    )
    if env_sha:
        return env_sha[:12]

    head_path = Path(".git") / "HEAD"
    if head_path.exists():
        try:
            head = head_path.read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                ref_path = Path(".git") / ref
                if ref_path.exists():
                    ref_sha = ref_path.read_text(encoding="utf-8").strip()
                    if ref_sha:
                        return ref_sha[:12]
            elif head:
                return head[:12]
        except Exception:
            pass

    image_tag = (
        os.getenv("IMAGE_TAG")
        or os.getenv("CONTAINER_IMAGE")
        or os.getenv("DOCKER_IMAGE")
    )
    if image_tag:
        return image_tag
    return "unknown"


def _format_refresh_date(
    marker: dict[str, Any] | None, meta: dict[str, Any] | None
) -> str:
    timestamp = None
    if marker:
        raw = marker.get("timestamp_utc")
        if isinstance(raw, str) and raw:
            timestamp = raw
    if not timestamp and meta:
        raw_meta = meta.get("last_modified_utc")
        if isinstance(raw_meta, datetime):
            return raw_meta.astimezone(timezone.utc).date().isoformat()
        if isinstance(raw_meta, str) and raw_meta:
            timestamp = raw_meta
    if timestamp:
        try:
            normalized = timestamp.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed.date().isoformat()
        except Exception:
            pass
    return "—"


def _render_refresh_values(
    *,
    refresh_date: str,
    season: Any,
    rows: Any,
    max_game_date: Any,
) -> None:
    labels = ["Last refresh (UTC)", "Season", "Rows", "Max game date"]
    values = [
        refresh_date or "—",
        season if season is not None else "—",
        rows if rows is not None else "—",
        max_game_date if max_game_date is not None else "—",
    ]
    cols = st.columns(4)
    for col, label, value in zip(cols, labels, values, strict=True):
        with col:
            st.markdown(
                f"""
<div style="line-height:1.2;">
  <div style="font-size:0.9rem;color:#6b7280;">{label}</div>
  <div style="font-size:0.9rem;font-weight:500;">{value}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def _render_refresh_debug(
    *,
    marker_blob: str,
    meta: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    err: Exception | None,
) -> None:
    with st.expander("Debug: Refresh marker", expanded=False):
        version = _resolve_app_version()
        st.write(f"App version: `{version}`")
        st.write(f"Storage account (resolved): `{storage_account()}`")
        st.write(f"Storage container (resolved): `{storage_container()}`")
        st.write(f"Lake prefix (resolved): `{lake_prefix()}`")
        st.write(
            "Env STORAGE_ACCOUNT_NAME: "
            f"`{os.getenv('STORAGE_ACCOUNT_NAME') or '—'}`"
        )
        st.write(
            "Env STORAGE_CONTAINER_NAME: "
            f"`{os.getenv('STORAGE_CONTAINER_NAME') or '—'}`"
        )
        st.write(
            "Env AZURE_STORAGE_ACCOUNT: "
            f"`{os.getenv('AZURE_STORAGE_ACCOUNT') or '—'}`"
        )
        st.write(
            "Env AZURE_STORAGE_CONTAINER: "
            f"`{os.getenv('AZURE_STORAGE_CONTAINER') or '—'}`"
        )
        st.write(
            "Env AZURE_LAKE_PREFIX: "
            f"`{os.getenv('AZURE_LAKE_PREFIX') or '—'}`"
        )
        st.write(f"Env BLOB_PREFIX: `{os.getenv('BLOB_PREFIX') or '—'}`")
        st.write(f"Marker blob path: `{marker_blob}`")

        exists = meta is not None
        st.write(f"Marker blob exists: `{exists}`")
        if meta:
            st.write(
                "Marker last_modified_utc: "
                f"`{meta.get('last_modified_utc') or '—'}`"
            )
            st.write(f"Marker etag: `{meta.get('etag') or '—'}`")

        if payload:
            st.write("Marker payload:")
            st.code(json.dumps(payload, indent=2, sort_keys=True))
            signature = marker_signature(meta, payload)
            if signature:
                st.write(f"Marker cache signature: `{signature}`")
        elif err:
            st.caption(f"Marker read error: {err}")


def render_status(
    *,
    games_df: pd.DataFrame | None = None,
    df_season: pd.DataFrame | None = None,
    selected_season: Any | None = None,
    team_a: str | None = None,
    team_b: str | None = None,
    team_a_id: int | None = None,
    team_b_id: int | None = None,
    window: int | None = None,
    team_a_is_home: bool | None = None,
) -> None:
    debug_enabled = _app_debug_enabled()
    debug_mode = False
    if debug_enabled:
        debug_mode = st.sidebar.checkbox(
            "Debug mode", value=False, key="debug_mode_status"
        )
    st.subheader("Model Snapshot")
    selected_season = (
        selected_season
        if selected_season is not None
        else st.session_state.get("selected_season")
    )
    team_a = team_a or st.session_state.get("team_a") or ""
    team_b = team_b or st.session_state.get("team_b") or ""
    team_a_id = team_a_id if team_a_id is not None else st.session_state.get("team_a_id")
    team_b_id = team_b_id if team_b_id is not None else st.session_state.get("team_b_id")
    window = int(window or st.session_state.get("window") or 0)
    team_a_is_home = (
        team_a_is_home
        if team_a_is_home is not None
        else st.session_state.get("team_a_is_home")
    )

    season_rows = int(df_season.shape[0]) if df_season is not None else 0
    teams_union_count = 0
    if df_season is not None and {"team_abbr", "opp_team_abbr"} <= set(df_season.columns):
        teams_union = pd.concat(
            [df_season["team_abbr"], df_season["opp_team_abbr"]],
            ignore_index=True,
        ).dropna()
        teams_union_count = int(teams_union.nunique())
    date_range = None
    if df_season is not None and "game_date" in df_season.columns:
        dates = pd.to_datetime(df_season["game_date"], errors="coerce").dropna()
        if not dates.empty:
            date_range = (dates.min().date(), dates.max().date())

    snapshot_cols = st.columns(3)
    snapshot_cols[0].metric("Season", str(selected_season) if selected_season is not None else "—")
    snapshot_cols[1].metric("Team A / Team B", f"{team_a} vs {team_b}".strip())
    snapshot_cols[2].metric("Team A is home", bool(team_a_is_home))

    snapshot_cols_2 = st.columns(3)
    snapshot_cols_2[0].metric("Recent form window (N)", window if window else "—")
    snapshot_cols_2[1].metric("Games loaded (season)", season_rows)
    snapshot_cols_2[2].metric("Teams in season", teams_union_count)

    if date_range:
        st.write(f"Date range: {date_range[0]} to {date_range[1]}")
    else:
        st.write("Date range: —")

    st.subheader("Model Training Snapshot")
    _render_training_snapshot()

    st.subheader("Last Refresh")
    marker_blob = f"{lake_prefix()}_meta/refresh_last_success.json"
    try:
        marker_meta, marker, marker_err = _load_refresh_marker()
    except Exception as exc:
        if debug_enabled and debug_mode:
            st.info("No refresh marker found yet.")
            st.caption(f"Debug: {exc}")
        else:
            st.info("No refresh marker found yet.")
        marker_meta = None
        marker = None
        marker_err = exc

    if marker:
        refresh_date = _format_refresh_date(marker, marker_meta)
        _render_refresh_values(
            refresh_date=refresh_date,
            season=marker.get("season"),
            rows=marker.get("rows"),
            max_game_date=marker.get("max_game_date"),
        )
    else:
        _render_refresh_values(
            refresh_date=_format_refresh_date(marker, marker_meta),
            season=None,
            rows=None,
            max_game_date=None,
        )

    if env_truthy("SHOW_DEBUG_REFRESH_MARKER"):
        _render_refresh_debug(
            marker_blob=marker_blob,
            meta=marker_meta,
            payload=marker,
            err=marker_err,
        )

    st.subheader("Recent games used")
    ctx = st.session_state.get("matchup_ctx")
    if not ctx or "games_df" not in ctx:
        st.info("Run Predict to view recent games used.")
    else:
        recent_df = ctx["games_df"].copy()
        if "team_abbr" in recent_df.columns:
            recent_df["team_abbr"] = (
                recent_df["team_abbr"].astype("string").str.strip().str.upper()
            )
        if "opp_team_abbr" in recent_df.columns:
            recent_df["opp_team_abbr"] = (
                recent_df["opp_team_abbr"].astype("string").str.strip().str.upper()
            )
        if "matchup" in recent_df.columns:
            matchup_series = recent_df["matchup"].astype("string").str.strip().str.upper()
            derived_team = matchup_series.str.split().str[0]
            derived_opp = matchup_series.str.split().str[-1]
            if "team_abbr" in recent_df.columns:
                recent_df["team_abbr"] = recent_df["team_abbr"].fillna(derived_team)
                empty_team = recent_df["team_abbr"].str.len() == 0
                recent_df.loc[empty_team, "team_abbr"] = derived_team
            if "opp_team_abbr" in recent_df.columns:
                recent_df["opp_team_abbr"] = recent_df["opp_team_abbr"].fillna(derived_opp)
                empty_opp = recent_df["opp_team_abbr"].str.len() == 0
                recent_df.loc[empty_opp, "opp_team_abbr"] = derived_opp
        recent_df["game_date_dt"] = pd.to_datetime(
            recent_df.get("game_date"), errors="coerce"
        )
        recent_team_id = pd.to_numeric(recent_df.get("team_id"), errors="coerce")
        if team_a_id is not None:
            recent_a = (
                recent_df[recent_team_id == int(team_a_id)]
                .sort_values("game_date_dt", ascending=False)
                .head(window or 0)
            )
        else:
            team_a_norm = _normalize_team(team_a or "")
            recent_a = (
                recent_df[recent_df["team_abbr"] == team_a_norm]
                .sort_values("game_date_dt", ascending=False)
                .head(window or 0)
            )
        if team_b_id is not None:
            recent_b = (
                recent_df[recent_team_id == int(team_b_id)]
                .sort_values("game_date_dt", ascending=False)
                .head(window or 0)
            )
        else:
            team_b_norm = _normalize_team(team_b or "")
            recent_b = (
                recent_df[recent_df["team_abbr"] == team_b_norm]
                .sort_values("game_date_dt", ascending=False)
                .head(window or 0)
            )

        if debug_mode:
            with st.expander("Debug: Recent games used", expanded=True):
                selected_season_value = selected_season
                storage_account = os.getenv("AZURE_STORAGE_ACCOUNT", "anthansunderrgaddf")
                storage_container = os.getenv("AZURE_STORAGE_CONTAINER", "nba-edge")
                lake = os.getenv("AZURE_LAKE_PREFIX", "lake/")
                if not lake.endswith("/"):
                    lake = f"{lake}/"
                st.write(
                    f"Storage account: `{storage_account}` | "
                    f"Container: `{storage_container}` | "
                    f"Lake prefix: `{lake}`"
                )

                blob_name = None
                features_prefix = None
                if selected_season_value is not None:
                    try:
                        prefix = lake_prefix()
                        blob_name = (
                            f"{prefix}silver/games/season={int(selected_season_value)}/games.parquet"
                        )
                        features_prefix = (
                            f"{prefix}silver/features/season={int(selected_season_value)}/"
                        )
                    except Exception as exc:
                        st.write(f"Failed to build blob path: {exc}")

                if blob_name:
                    st.write("Games parquet path:")
                    st.code(blob_name)
                    try:
                        blobs = list_blobs(
                            f"{lake_prefix()}silver/games/season={int(selected_season_value)}/"
                        )
                        game_blob = next(
                            (b for b in blobs if str(b.get("name", "")).endswith("games.parquet")),
                            None,
                        )
                        if game_blob:
                            st.write(
                                "Games parquet last_modified: "
                                f"{game_blob.get('last_modified_utc')} | size_bytes="
                                f"{game_blob.get('size_bytes')}"
                            )
                    except Exception as exc:
                        st.write(f"Failed to read blob metadata: {exc}")

                if features_prefix:
                    st.write("Features parquet prefix:")
                    st.code(features_prefix)

                raw_df = ctx.get("games_df")
                raw_dates = (
                    pd.to_datetime(raw_df.get("game_date"), errors="coerce")
                    if raw_df is not None
                    else pd.Series([], dtype="datetime64[ns]")
                )
                raw_max = raw_dates.max() if not raw_dates.empty else None
                raw_post = (raw_dates > pd.Timestamp("2026-02-01")).sum()

                post_dates = recent_df.get("game_date_dt")
                post_max = (
                    post_dates.max() if post_dates is not None and not post_dates.empty else None
                )
                post_post = (
                    (post_dates > pd.Timestamp("2026-02-01")).sum()
                    if post_dates is not None
                    else 0
                )

                st.write(
                    f"Raw rows: {len(raw_df) if raw_df is not None else 0} | "
                    f"raw max_date: {raw_max} | "
                    f"raw post-2026-02-01 count: {raw_post}"
                )
                st.write(
                    f"Post-normalization rows: {len(recent_df)} | "
                    f"post max_date: {post_max} | "
                    f"post-2026-02-01 count: {post_post}"
                )
                if "game_date" in recent_df.columns:
                    st.write(f"game_date dtype: {recent_df['game_date'].dtype}")
                st.write(f"game_date_dt dtype: {recent_df['game_date_dt'].dtype}")

                top_recent = (
                    recent_df.sort_values("game_date_dt", ascending=False)
                    .head(10)
                    .loc[:, [c for c in ["game_date_dt", "game_date", "team_abbr", "opp_team_abbr", "pts", "plus_minus", "wl"] if c in recent_df.columns]]
                )
                st.write("Top 10 most recent rows (post-normalization):")
                st.dataframe(top_recent, use_container_width=True)

                st.write(
                    f"Team A filtered rows: {len(recent_a)} | "
                    f"max_date: {recent_a['game_date_dt'].max() if not recent_a.empty else None}"
                )
                st.write(
                    f"Team B filtered rows: {len(recent_b)} | "
                    f"max_date: {recent_b['game_date_dt'].max() if not recent_b.empty else None}"
                )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{team_a or 'Team A'}**")
            if not recent_a.empty:
                recent_a_display = recent_a.assign(
                    date=recent_a["game_date_dt"]
                    .dt.strftime("%Y-%m-%d")
                    .fillna(recent_a.get("game_date").astype(str)),
                    opponent=recent_a.get("opp_team_abbr"),
                    home_away=recent_a.get("is_home").map(
                        lambda v: "Home" if bool(v) else "Away"
                    ),
                )
                recent_a_display = recent_a_display[
                    ["date", "opponent", "home_away", "pts", "plus_minus", "wl"]
                ].copy()
                st.dataframe(recent_a_display, use_container_width=True)
            else:
                st.dataframe(recent_a, use_container_width=True)
        with col_b:
            st.markdown(f"**{team_b or 'Team B'}**")
            if not recent_b.empty:
                recent_b_display = recent_b.assign(
                    date=recent_b["game_date_dt"]
                    .dt.strftime("%Y-%m-%d")
                    .fillna(recent_b.get("game_date").astype(str)),
                    opponent=recent_b.get("opp_team_abbr"),
                    home_away=recent_b.get("is_home").map(
                        lambda v: "Home" if bool(v) else "Away"
                    ),
                )
                recent_b_display = recent_b_display[
                    ["date", "opponent", "home_away", "pts", "plus_minus", "wl"]
                ].copy()
                st.dataframe(recent_b_display, use_container_width=True)
            else:
                st.dataframe(recent_b, use_container_width=True)


def _render_training_snapshot() -> None:
    metrics_path = Path("models") / "artifacts" / "ridge_margin_metrics.json"
    if not metrics_path.exists():
        st.info(
            "No training snapshot found. Run `.venv/bin/python -m models.train_ridge --train_seasons 2022 2023 2024 --test_seasons 2025`"
        )
        return

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        st.info(
            "No training snapshot found. Run `.venv/bin/python -m models.train_ridge --train_seasons 2022 2023 2024 --test_seasons 2025`"
        )
        return

    created_at_raw = payload.get("created_at")
    created_at_label = "—"
    if isinstance(created_at_raw, str) and created_at_raw:
        try:
            normalized = created_at_raw.replace("Z", "+00:00")
            created_at = datetime.fromisoformat(normalized).astimezone(timezone.utc)
            created_at_label = created_at.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            created_at_label = created_at_raw

    train_seasons = payload.get("train_seasons") or []
    test_seasons = payload.get("test_seasons") or []
    feature_count = payload.get("feature_count")
    mae = payload.get("mae")
    rmse = payload.get("rmse")

    summary_cols = st.columns(3)
    summary_cols[0].metric("Created at", created_at_label)
    summary_cols[1].metric(
        "Train seasons", ", ".join(str(s) for s in train_seasons) or "—"
    )
    summary_cols[2].metric(
        "Test seasons", ", ".join(str(s) for s in test_seasons) or "—"
    )

    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Feature count", feature_count if feature_count else "—")
    metrics_cols[1].metric("MAE", f"{mae:.2f}" if isinstance(mae, (int, float)) else "—")
    metrics_cols[2].metric(
        "RMSE", f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "—"
    )

    pos_df = pd.DataFrame(payload.get("top_positive_coefficients") or [])
    neg_df = pd.DataFrame(payload.get("top_negative_coefficients") or [])
    if not pos_df.empty:
        pos_df = pos_df[["feature", "coefficient"]]
    if not neg_df.empty:
        neg_df = neg_df[["feature", "coefficient"]]

    table_cols = st.columns(2)
    with table_cols[0]:
        st.markdown("**Top positive coefficients**")
        st.dataframe(pos_df, use_container_width=True)
    with table_cols[1]:
        st.markdown("**Top negative coefficients**")
        st.dataframe(neg_df, use_container_width=True)
