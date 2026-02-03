"""Ingest NBA team game logs from nba_api into the raw data lake."""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import datetime, timezone

from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams

from common.blob import upload_bytes
from common.config import Config


def main() -> None:
    """Entry point for ingesting NBA game logs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = Config()
    run_dt = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    all_teams = teams.get_teams()
    logging.info("Fetched %s teams.", len(all_teams))

    total_game_rows = 0
    for team in all_teams:
        team_id = team.get("id")
        for season in range(config.START_SEASON, config.END_SEASON + 1):
            season_str = f"{season}-{str(season + 1)[-2:]}"
            response = _retry_team_gamelog(team_id=team_id, season=season_str)
            payload = response.get_dict()

            game_rows = _count_rows(payload)
            total_game_rows += game_rows

            blob_name = (
                f"{config.raw_nba_teamgamelogs_path()}/"
                f"season={season}/team_id={team_id}/run_dt={run_dt}/payload.json.gz"
            )
            upload_bytes(
                config.BLOB_CONTAINER,
                blob_name,
                _gzip_json(payload),
                content_type="application/json",
            )
            logging.info(
                "Uploaded team_id=%s season=%s rows=%s blob=%s",
                team_id,
                season,
                game_rows,
                blob_name,
            )

            time.sleep(0.6)

    logging.info("Completed ingest. Teams=%s Total rows=%s", len(all_teams), total_game_rows)


def _retry_team_gamelog(team_id: int, season: str, retries: int = 3, backoff: float = 1.5):
    for attempt in range(1, retries + 1):
        try:
            return teamgamelog.TeamGameLog(team_id=team_id, season=season)
        except Exception as exc:
            if attempt == retries:
                raise
            sleep_s = backoff**attempt
            logging.warning("Retry %s for team_id=%s season=%s: %s", attempt, team_id, season, exc)
            time.sleep(sleep_s)


def _gzip_json(payload: dict) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    return gzip.compress(data)


def _count_rows(payload: dict) -> int:
    try:
        result_sets = payload.get("resultSets") or []
        if not result_sets:
            return 0
        return len(result_sets[0].get("rowSet") or [])
    except Exception:
        return 0


if __name__ == "__main__":
    main()
