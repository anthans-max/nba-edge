"""Chat UI rendering."""

from __future__ import annotations

import os

import streamlit as st

from llm_chat import chat_with_context

DEBUG = os.getenv("NBA_EDGE_DEBUG", "").lower() in ("1", "true", "yes", "y", "on")


def init_chat_state(context_key: str) -> None:
    if st.session_state.get("chat_context_key") != context_key:
        st.session_state["chat_context_key"] = context_key
        st.session_state["chat_messages"] = []
        st.session_state.pop("pending_user_message", None)


def render_chat(context_packet: dict | None, team_a: str, team_b: str) -> None:
    st.subheader("Ask the Model")
    st.caption("Answers are grounded in the current matchup's recent-form data and prediction.")
    if DEBUG:
        st.write("has_matchup_ctx:", bool(st.session_state.get("matchup_ctx")))
        st.write("has_full_matchup_context:", bool(st.session_state.get("matchup_context")))
        st.write("has_prediction:", bool(st.session_state.get("prediction")))
        st.write("matchup_signature:", st.session_state.get("matchup_signature"))

    context_packet = st.session_state.get("matchup_ctx") or context_packet
    if st.session_state.get("matchup_context") is None:
        st.info("Run a prediction on the Matchup tab first.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    chip_cols = st.columns(2)
    if chip_cols[0].button(f"Why is {team_a} favored?"):
        st.session_state["chat_draft"] = f"Why is {team_a} favored?"
    if chip_cols[1].button("What are the top 3 factors?"):
        st.session_state["chat_draft"] = "What are the top 3 factors?"
    if chip_cols[0].button("How much did home court matter?"):
        st.session_state["chat_draft"] = "How much did home court matter?"
    if chip_cols[1].button("Summarize both teams' last 10 games form."):
        st.session_state["chat_draft"] = "Summarize both teams' last 10 games form."

    pending = st.session_state.pop("chat_draft", None)
    user_input = st.chat_input("Ask a question about this matchup")
    if pending and not user_input:
        user_input = pending

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

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
