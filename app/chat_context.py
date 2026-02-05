"""Build grounded matchup context for chat responses."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _to_float_map(series: pd.Series | dict[str, Any]) -> dict[str, float]:
    values = dict(series) if isinstance(series, pd.Series) else dict(series)
    output: dict[str, float] = {}
    for key, value in values.items():
        try:
            numeric = pd.to_numeric(value, errors="coerce")
        except Exception:
            numeric = float("nan")
        if pd.notna(numeric):
            output[str(key)] = float(numeric)
    return output


def build_matchup_context(
    *,
    season: int,
    window_n: int,
    home_team_id: int,
    away_team_id: int,
    home_team_label: str,
    away_team_label: str,
    home_features: pd.Series | dict[str, Any],
    away_features: pd.Series | dict[str, Any],
    diff_features: pd.Series | dict[str, Any],
    predicted_margin: float,
    top_drivers: pd.DataFrame,
) -> dict[str, Any]:
    """Return a JSON-serializable context packet for matchup chat."""
    top_driver_rows: list[dict[str, Any]] = []
    for _, row in top_drivers.iterrows():
        top_driver_rows.append(
            {
                "feature": str(row.get("feature")),
                "home": float(pd.to_numeric(row.get("home"), errors="coerce") or 0.0),
                "away": float(pd.to_numeric(row.get("away"), errors="coerce") or 0.0),
                "diff": float(pd.to_numeric(row.get("diff"), errors="coerce") or 0.0),
                "contribution": float(
                    pd.to_numeric(row.get("contribution"), errors="coerce") or 0.0
                ),
                "direction": str(row.get("direction") or "neutral"),
            }
        )

    return {
        "season": int(season),
        "window_n": int(window_n),
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "home_team_label": str(home_team_label),
        "away_team_label": str(away_team_label),
        "home_features": _to_float_map(home_features),
        "away_features": _to_float_map(away_features),
        "diff_features": _to_float_map(diff_features),
        "predicted_margin": float(predicted_margin),
        "top_drivers": top_driver_rows,
    }
