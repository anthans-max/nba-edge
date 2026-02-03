"""Gemini chat helper grounded in matchup context."""

from __future__ import annotations

import json
import os
from typing import Iterable

import streamlit as st

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
def build_gemini_client(api_key: str):
    """Build and cache a Gemini client."""
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
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


def chat_with_context(messages: Iterable[dict], context_packet: dict) -> str:
    """Send a grounded prompt to Gemini and return the assistant response."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Chat is disabled. Set GEMINI_API_KEY to enable it."
    if genai is None:
        return "Chat is disabled. Install google-generativeai to enable it."

    model = build_gemini_client(api_key)
    context_json = json.dumps(context_packet, indent=2, default=str)
    conversation = _format_messages(messages)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"context_packet (JSON):\n{context_json}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        "Assistant:"
    )

    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pragma: no cover - network/runtime issues
        return f"Sorry, I couldn't reach Gemini just now. {exc}"

    text = getattr(response, "text", None) or str(response)
    return text.strip()
