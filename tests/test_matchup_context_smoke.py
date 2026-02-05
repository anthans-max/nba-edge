"""Smoke assertions for matchup chat context construction."""

import pandas as pd

from app.chat_context import build_matchup_context


def test_build_matchup_context_has_expected_keys():
    home = pd.Series({"is_home": 1.0, "margin_avg_last_10": 3.2})
    away = pd.Series({"is_home": 0.0, "margin_avg_last_10": -1.1})
    diff = home - away
    drivers = pd.DataFrame(
        [
            {
                "feature": "margin_avg_last_10",
                "home": 3.2,
                "away": -1.1,
                "diff": 4.3,
                "contribution": 1.72,
                "direction": "home",
            },
            {
                "feature": "is_home",
                "home": 1.0,
                "away": 0.0,
                "diff": 1.0,
                "contribution": 0.25,
                "direction": "home",
            },
        ]
    )

    ctx = build_matchup_context(
        season=2022,
        window_n=10,
        home_team_id=1610612737,
        away_team_id=1610612738,
        home_team_label="ATL - Atlanta Hawks",
        away_team_label="BOS - Boston Celtics",
        home_features=home,
        away_features=away,
        diff_features=diff,
        predicted_margin=4.8,
        top_drivers=drivers,
    )

    expected = {
        "season",
        "window_n",
        "home_team_id",
        "away_team_id",
        "home_team_label",
        "away_team_label",
        "home_features",
        "away_features",
        "diff_features",
        "predicted_margin",
        "top_drivers",
    }
    assert expected <= set(ctx.keys())
    assert ctx["home_team_id"] == 1610612737
    assert isinstance(ctx["top_drivers"], list)
    assert ctx["top_drivers"][0]["feature"] == "margin_avg_last_10"
