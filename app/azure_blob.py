"""Azure Blob helpers for Streamlit UI."""

from __future__ import annotations

import io
import os
import traceback
from typing import Iterable

import pandas as pd
import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


def _storage_account() -> str:
    return os.getenv("AZURE_STORAGE_ACCOUNT", "anthansunderrgaddf")


def _container_name() -> str:
    return os.getenv("AZURE_STORAGE_CONTAINER", "nba-edge")


def _lake_prefix() -> str:
    prefix = os.getenv("AZURE_LAKE_PREFIX", "lake/")
    return prefix if prefix.endswith("/") else f"{prefix}/"


@st.cache_resource
def get_blob_service_client() -> BlobServiceClient:
    account_url = f"https://{_storage_account()}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


def list_blobs(prefix: str) -> list[dict]:
    client = get_blob_service_client()
    container = client.get_container_client(_container_name())
    items = container.list_blobs(name_starts_with=prefix)
    return [
        {
            "name": b.name,
            "size_bytes": b.size,
            "last_modified_utc": b.last_modified,
        }
        for b in items
    ]


def download_blob_bytes(blob_name: str) -> bytes:
    client = get_blob_service_client()
    blob = client.get_blob_client(container=_container_name(), blob=blob_name)
    return blob.download_blob().readall()


@st.cache_data(show_spinner=False)
def read_parquet_from_blob(blob_name: str) -> pd.DataFrame:
    data = download_blob_bytes(blob_name)
    return pd.read_parquet(io.BytesIO(data))


def parse_available_seasons(blobs: Iterable[dict]) -> list[int]:
    seasons = set()
    for blob in blobs:
        name = blob.get("name", "")
        if "/season=" in name:
            try:
                part = name.split("/season=")[1].split("/")[0]
                seasons.add(int(part))
            except Exception:
                continue
    return sorted(seasons)


def parse_available_odds_dates(blobs: Iterable[dict]) -> list[str]:
    dates = set()
    for blob in blobs:
        name = blob.get("name", "")
        if "/date=" in name:
            try:
                part = name.split("/date=")[1].split("/")[0]
                dates.add(part)
            except Exception:
                continue
    return sorted(dates)


def render_exception(err: Exception) -> None:
    st.error(str(err))
    st.code(traceback.format_exc())


def lake_prefix() -> str:
    return _lake_prefix()
