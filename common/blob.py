"""Azure Blob Storage helpers."""

from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings


def get_blob_service_client() -> BlobServiceClient:
    """Create a BlobServiceClient using connection string or default credentials."""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME or AZURE_STORAGE_CONNECTION_STRING must be set.")

    account_url = f"https://{account_name}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


def upload_bytes(container: str, blob_name: str, data: bytes, content_type: str) -> None:
    """Upload bytes to Azure Blob Storage with a content type."""
    service = get_blob_service_client()
    container_client = service.get_container_client(container)
    try:
        container_client.create_container()
    except Exception:
        pass

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )


def list_blobs(container: str, prefix: str):
    """List blobs under a prefix."""
    service = get_blob_service_client()
    container_client = service.get_container_client(container)
    return container_client.list_blobs(name_starts_with=prefix)


def download_bytes(container: str, blob_name: str) -> bytes:
    """Download a blob's content as bytes."""
    service = get_blob_service_client()
    blob_client = service.get_blob_client(container=container, blob=blob_name)
    return blob_client.download_blob().readall()
