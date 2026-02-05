"""Shared configuration helpers for the Streamlit app."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st


def get_setting(name: str, default: Any | None = None) -> Any | None:
    """Resolve settings from Streamlit secrets, then environment, then default."""
    try:
        if name in st.secrets:
            return st.secrets.get(name)
    except Exception:
        # st.secrets can raise when secrets are not configured in some environments.
        pass
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    return default
