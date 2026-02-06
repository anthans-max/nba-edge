import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


def _run_module(module: str) -> None:
    print(f"[refresh_job] Running: {sys.executable} -m {module}", flush=True)
    subprocess.check_call([sys.executable, "-m", module])


def _canonical_season_for_utc(dt: datetime) -> int:
    return dt.year if dt.month >= 10 else dt.year - 1


def _get_env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _write_refresh_marker() -> None:
    account = _get_env_required("STORAGE_ACCOUNT_NAME")
    container = _get_env_required("STORAGE_CONTAINER_NAME")
    now = datetime.now(timezone.utc)
    season = _canonical_season_for_utc(now)
    prefix = os.getenv("BLOB_PREFIX", "lake").strip("/")
    games_blob = f"{prefix}/silver/games/season={season}/games.parquet"
    marker_blob = f"{prefix}/_meta/refresh_last_success.json"

    account_url = f"https://{account}.blob.core.windows.net"
    client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())

    tmp_path = Path("/tmp") / f"games-{season}.parquet"
    blob_client = client.get_blob_client(container=container, blob=games_blob)
    tmp_path.write_bytes(blob_client.download_blob().readall())

    df = pd.read_parquet(tmp_path)
    rows = int(len(df))
    max_game_date = None
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"], errors="coerce").dropna()
        if not dates.empty:
            max_game_date = dates.max().date().isoformat()

    image_tag = (
        os.getenv("IMAGE_TAG")
        or os.getenv("CONTAINER_IMAGE")
        or os.getenv("DOCKER_IMAGE")
    )
    payload = {
        "timestamp_utc": now.isoformat(),
        "season": season,
        "rows": rows,
        "max_game_date": max_game_date,
    }
    if image_tag:
        payload["image_tag"] = image_tag

    marker_client = client.get_blob_client(container=container, blob=marker_blob)
    marker_client.upload_blob(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        overwrite=True,
        content_type="application/json",
    )


def main() -> int:
    try:
        _run_module("transform.build_silver_games")
        _run_module("features.build_features")
        _write_refresh_marker()
    except subprocess.CalledProcessError as exc:
        print(
            f"[refresh_job] Failed with exit code {exc.returncode}: {exc.cmd}",
            flush=True,
        )
        return exc.returncode
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[refresh_job] Unexpected error: {exc}", flush=True)
        return 1

    print("[refresh_job] Completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
