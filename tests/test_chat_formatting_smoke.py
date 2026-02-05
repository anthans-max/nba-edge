"""Smoke checks for deterministic chat formatting helpers."""

from app.llm_chat import (
    _deterministic_answer,
    format_contribution,
    format_pct,
    format_stat,
)


def test_format_helpers():
    assert format_contribution(1.234) == "+1.23"
    assert format_contribution(-1.234) == "-1.23"
    assert format_pct(0.4) == "40%"
    assert format_stat(113.84) == "113.8"


def test_top3_phrase_and_negligible_note():
    context = {
        "home_team_label": "ATL - Atlanta Hawks",
        "away_team_label": "BOS - Boston Celtics",
        "predicted_margin": 2.345,
        "top_drivers": [
            {
                "feature": "margin_avg_last_10",
                "home": 3.2,
                "away": -1.1,
                "diff": 4.3,
                "contribution": 1.72,
                "direction": "home",
            },
            {
                "feature": "win_rate_last_5",
                "home": 0.4,
                "away": 0.6,
                "diff": -0.2,
                "contribution": -1.25,
                "direction": "away",
            },
            {
                "feature": "rest_days",
                "home": 2.0,
                "away": 2.0,
                "diff": 0.0,
                "contribution": 0.0,
                "direction": "neutral",
            },
        ],
    }
    messages = [{"role": "user", "content": "What are the top 3 factors?"}]
    answer = _deterministic_answer(messages, context)
    assert "Top factors by absolute contribution" in answer
    assert "Other factors were negligible in this prediction." in answer
    assert "This offsets the prediction by -1.25 points, favoring BOS - Boston Celtics." in answer
    assert "40%" in answer or "60%" in answer
