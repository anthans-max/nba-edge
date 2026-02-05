"""Gemini chat helper grounded in matchup context."""

from __future__ import annotations

import difflib
import json
import re
from typing import Iterable

import streamlit as st

try:
    from settings import get_setting
except ModuleNotFoundError:  # pragma: no cover - package import in tests
    from app.settings import get_setting
from dotenv import load_dotenv

load_dotenv()
try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled by dependency install
    genai = None


SYSTEM_PROMPT = """You are an NBA matchup assistant for NBA Edge.
You MUST only use facts from the provided context_packet JSON. Do not invent facts.
Do not query external systems or run code. If asked for info not in context_packet,
state that it is not available in this V1 and suggest changing season, window, or teams,
or using the Backtest page. Use plain English, concise, stakeholder-friendly tone."""


@st.cache_resource
def build_gemini_client(api_key: str, model_name: str):
    """Build and cache a Gemini client."""
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )


def _format_messages(messages: Iterable[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def resolve_gemini_key() -> str:
    return (
        str(get_setting("GEMINI_API_KEY", "")).strip()
        or str(get_setting("GOOGLE_API_KEY", "")).strip()
        or str(get_setting("GEMINI_KEY", "")).strip()
    )


def _latest_user_message(messages: Iterable[dict]) -> str:
    for msg in reversed(list(messages)):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def format_contribution(value: float) -> str:
    return f"{float(value):+.2f}"


def format_pct(value: float, signed: bool = False) -> str:
    scaled = float(value) * 100.0
    if signed:
        return f"{scaled:+.0f}%"
    return f"{scaled:.0f}%"


def format_stat(value: float, signed: bool = False) -> str:
    if signed:
        return f"{float(value):+.1f}"
    return f"{float(value):.1f}"


def _is_rate_feature(feature_name: str) -> bool:
    key = feature_name.lower()
    return "win_rate" in key or "win_pct" in key


def _format_feature_value(feature_name: str, value: float, signed: bool = False) -> str:
    if _is_rate_feature(feature_name):
        return format_pct(value, signed=signed)
    return format_stat(value, signed=signed)


def _direction_team_label(direction: str, home_label: str, away_label: str) -> str:
    if direction == "home":
        return home_label
    if direction == "away":
        return away_label
    return "neither team"


def _driver_impact_phrase(
    contribution: float, direction: str, home_label: str, away_label: str
) -> str:
    favored_team = _direction_team_label(direction, home_label, away_label)
    if contribution < 0:
        return (
            f"This offsets the prediction by {format_contribution(contribution)} points, "
            f"favoring {favored_team}."
        )
    return (
        f"This adds {format_contribution(contribution)} points, "
        f"favoring {favored_team}."
    )


def _top_drivers(context: dict, top_n: int = 3) -> list[dict]:
    drivers = list(context.get("top_drivers") or [])
    meaningful = [
        row for row in drivers if abs(float(row.get("contribution", 0.0))) > 1e-6
    ]
    return sorted(
        meaningful,
        key=lambda row: abs(float(row.get("contribution", 0.0))),
        reverse=True,
    )[:top_n]


def _find_feature_match(query: str, features: list[str]) -> str | None:
    if not query or not features:
        return None
    q = query.lower().strip()
    lookup = {f.lower(): f for f in features}
    if q in lookup:
        return lookup[q]
    for key, value in lookup.items():
        if key and key in q:
            return value
    tokens = [t for t in re.split(r"[^a-zA-Z0-9_]+", q) if t]
    for token in tokens:
        close = difflib.get_close_matches(token, list(lookup.keys()), n=1, cutoff=0.85)
        if close:
            return lookup[close[0]]
    close_full = difflib.get_close_matches(q, list(lookup.keys()), n=1, cutoff=0.65)
    if close_full:
        return lookup[close_full[0]]
    return None


def _feature_response(context: dict, feature_name: str) -> str:
    home_label = str(context.get("home_team_label") or "Home")
    away_label = str(context.get("away_team_label") or "Away")
    home = float((context.get("home_features") or {}).get(feature_name, 0.0))
    away = float((context.get("away_features") or {}).get(feature_name, 0.0))
    diff = float((context.get("diff_features") or {}).get(feature_name, home - away))
    contribution = None
    direction = "neutral"
    for row in list(context.get("top_drivers") or []):
        if str(row.get("feature")) == feature_name:
            contribution = float(row.get("contribution", 0.0))
            direction = str(row.get("direction") or "neutral")
            break
    if contribution is None:
        return (
            f"`{feature_name}`: {home_label}={_format_feature_value(feature_name, home)}, "
            f"{away_label}={_format_feature_value(feature_name, away)}, "
            f"diff={_format_feature_value(feature_name, diff, signed=True)}."
        )
    impact = _driver_impact_phrase(contribution, direction, home_label, away_label)
    return (
        f"`{feature_name}`: {home_label}={_format_feature_value(feature_name, home)}, "
        f"{away_label}={_format_feature_value(feature_name, away)}, "
        f"diff={_format_feature_value(feature_name, diff, signed=True)}, "
        f"contribution={format_contribution(contribution)} ({direction}). {impact}"
    )


def _deterministic_answer(messages: Iterable[dict], context: dict) -> str:
    query = _latest_user_message(messages)
    query_lower = query.lower()

    home_label = str(context.get("home_team_label") or "Home")
    away_label = str(context.get("away_team_label") or "Away")
    margin = float(context.get("predicted_margin") or 0.0)
    favored = home_label if margin >= 0 else away_label

    top3 = _top_drivers(context, top_n=3)
    top3_lines: list[str] = []
    for idx, row in enumerate(top3, start=1):
        feature = str(row.get("feature"))
        contribution = float(row.get("contribution", 0.0))
        direction = str(row.get("direction") or "neutral")
        impact = _driver_impact_phrase(contribution, direction, home_label, away_label)
        top3_lines.append(
            f"{idx}. {feature}: contribution={format_contribution(contribution)} "
            f"({home_label}={_format_feature_value(feature, float(row.get('home', 0.0)))}, "
            f"{away_label}={_format_feature_value(feature, float(row.get('away', 0.0)))}, "
            f"diff={_format_feature_value(feature, float(row.get('diff', 0.0)), signed=True)}). "
            f"{impact}"
        )

    if "top 3" in query_lower and ("factor" in query_lower or "driver" in query_lower):
        if not top3_lines:
            return "No top-driver rows are available for this prediction."
        if len(top3_lines) < 3:
            return (
                "Top factors by absolute contribution:\n"
                + "\n".join(top3_lines)
                + "\nOther factors were negligible in this prediction."
            )
        return "Top 3 factors by absolute contribution:\n" + "\n".join(top3_lines)

    if ("why is" in query_lower and "favor" in query_lower) or "why favored" in query_lower:
        header = (
            f"{favored} is favored by {format_contribution(margin)} points based on the current feature diff vector."
        )
        if not top3_lines:
            return header
        return header + "\nTop drivers:\n" + "\n".join(top3_lines)

    if "home court" in query_lower or "is_home" in query_lower:
        drivers = list(context.get("top_drivers") or [])
        is_home_row = next(
            (row for row in drivers if str(row.get("feature")) == "is_home"),
            None,
        )
        if is_home_row is None:
            return "`is_home` is not present in this matchup's driver table."
        contribution = float(is_home_row.get("contribution", 0.0))
        direction = str(is_home_row.get("direction") or "neutral")
        impact = _driver_impact_phrase(contribution, direction, home_label, away_label)
        return (
            "Home court effect (`is_home`): "
            f"{home_label}={_format_feature_value('is_home', float(is_home_row.get('home', 0.0)))}, "
            f"{away_label}={_format_feature_value('is_home', float(is_home_row.get('away', 0.0)))}, "
            f"diff={_format_feature_value('is_home', float(is_home_row.get('diff', 0.0)), signed=True)}, "
            f"contribution={format_contribution(contribution)} ({direction}). "
            f"{impact}"
        )

    if (
        "last 10" in query_lower
        and ("form" in query_lower or "summarize" in query_lower)
    ) or "recent form" in query_lower:
        home_f = context.get("home_features") or {}
        away_f = context.get("away_features") or {}
        candidates = [
            "margin_avg_last_10",
            "margin_avg_last_5",
            "pts_for_avg_last_5",
            "pts_against_avg_last_5",
            "win_rate_last_5",
            "rest_days",
            "games_played_to_date",
        ]
        lines = [
            f"{feat}: {home_label}={_format_feature_value(feat, float(home_f.get(feat, 0.0)))}, "
            f"{away_label}={_format_feature_value(feat, float(away_f.get(feat, 0.0)))}"
            for feat in candidates
            if feat in home_f or feat in away_f
        ]
        if not lines:
            return "Recent-form rolling features are not present in this matchup context."
        return "Recent form summary from rolling features:\n" + "\n".join(lines)

    feature_names = list((context.get("diff_features") or {}).keys())
    matched = _find_feature_match(query, feature_names)
    if matched:
        return _feature_response(context, matched)

    if top3_lines:
        return (
            f"Predicted margin: {format_contribution(margin)} for {home_label} vs {away_label}. "
            "Ask for top 3 factors, home court impact, recent form, or a specific feature name."
        )
    return "Ask about top factors, home court impact, recent form, or a specific feature."


def _rewrite_with_llm(messages: Iterable[dict], context: dict, deterministic: str) -> str:
    api_key = resolve_gemini_key()
    if not api_key or genai is None:
        return deterministic

    model_name = str(get_setting("GEMINI_MODEL", "gemini-2.5-flash-lite")).strip()
    model = build_gemini_client(api_key, model_name)
    context_json = json.dumps(context, indent=2, default=str)
    conversation = _format_messages(messages)
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", deterministic)

    prompt = (
        "Rewrite the answer for tone only.\n"
        "Do not change any numbers, signs, team labels, feature names, or factual claims.\n"
        "If you cannot preserve all values exactly, return the original answer unchanged.\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"context_packet (JSON):\n{context_json}\n\n"
        f"Deterministic answer:\n{deterministic}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        "Assistant:"
    )

    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pragma: no cover - network/runtime issues
        return deterministic

    text = getattr(response, "text", None) or str(response)
    rewritten = text.strip()
    if not rewritten:
        return deterministic
    if any(num not in rewritten for num in numbers):
        return deterministic
    return rewritten


def chat_with_context(messages: Iterable[dict], context_packet: dict | None = None) -> str:
    """Return grounded deterministic chat output with optional LLM tone rewrite."""
    context = st.session_state.get("matchup_context")
    if not context and context_packet:
        context = context_packet
    if not context:
        return "Run a prediction on the Matchup tab first."
    deterministic = _deterministic_answer(messages, context)
    return _rewrite_with_llm(messages, context, deterministic)
