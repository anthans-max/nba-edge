"""Train a Ridge regression baseline for margin prediction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.config import Config


IDENTIFIER_COLUMNS = {
    "season",
    "game_id",
    "game_date",
    "team_id",
    "team_abbr",
    "opponent_id",
    "opponent_abbr",
}
OUTCOME_COLUMNS = {
    "pts_for",
    "pts_against",
    "margin",
    "wl",
    "plus_minus",
}


def main() -> None:
    args = _parse_args()
    config = Config()

    train_df = _load_seasons(config, args.train_seasons, label="train")
    test_df = _load_seasons(config, args.test_seasons, label="test")

    if train_df.empty:
        raise ValueError("No training data loaded.")
    if test_df.empty:
        raise ValueError("No test data loaded.")

    x_train, y_train = _build_xy(train_df)
    x_test, y_test = _build_xy(test_df)

    artifacts_dir = Path("models") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    top_positive, top_negative = _print_top_coefficients(
        model, x_train.columns, top_n=10
    )
    _write_artifacts(
        model,
        mae,
        rmse,
        train_seasons=args.train_seasons,
        test_seasons=args.test_seasons,
        feature_columns=list(x_train.columns),
        top_positive=top_positive,
        top_negative=top_negative,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ridge regression baseline.")
    parser.add_argument(
        "--train_seasons",
        nargs="+",
        type=int,
        required=True,
        help="Training seasons (e.g. 2022 2023 2024).",
    )
    parser.add_argument(
        "--test_seasons",
        nargs="+",
        type=int,
        required=True,
        help="Test seasons (e.g. 2025).",
    )
    return parser.parse_args()


def _load_seasons(config: Config, seasons: Iterable[int], label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        blob_name = f"{config.BLOB_PREFIX}/silver/features/season={season}/features.parquet"
        df = config.read_parquet(blob_name)
        if df.empty:
            print(f"[train_ridge] {label} season={season} is empty.")
        else:
            print(f"[train_ridge] {label} season={season} rows={len(df)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data.columns = [str(c) for c in data.columns]
    if "margin" not in data.columns:
        raise ValueError("margin column missing from feature set.")

    y = pd.to_numeric(data["margin"], errors="coerce").fillna(0.0)
    drop_cols = IDENTIFIER_COLUMNS | OUTCOME_COLUMNS
    candidate = data.drop(columns=[c for c in data.columns if c in drop_cols], errors="ignore")

    numeric = candidate.select_dtypes(include=["number", "bool"]).copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    x = numeric.fillna(0.0)

    if x.empty:
        raise ValueError("No numeric feature columns available after filtering.")

    print(f"Feature columns count: {len(x.columns)}")
    print("Feature columns sample:", ", ".join(list(x.columns)[:20]))

    return x, y


def _print_top_coefficients(
    model: Pipeline, feature_names: Iterable[str], top_n: int
) -> tuple[list[dict], list[dict]]:
    ridge: Ridge = model.named_steps["ridge"]
    coefs = pd.Series(ridge.coef_, index=list(feature_names))
    sorted_coefs = coefs.sort_values()

    print("Top negative coefficients:")
    top_negative: list[dict] = []
    for name, value in sorted_coefs.head(top_n).items():
        print(f"  {name}: {value:.4f}")
        top_negative.append({"feature": name, "coefficient": float(value)})

    print("Top positive coefficients:")
    top_positive: list[dict] = []
    for name, value in sorted_coefs.tail(top_n).sort_values(ascending=False).items():
        print(f"  {name}: {value:.4f}")
        top_positive.append({"feature": name, "coefficient": float(value)})

    return top_positive, top_negative


def _write_artifacts(
    model: Pipeline,
    mae: float,
    rmse: float,
    train_seasons: Iterable[int],
    test_seasons: Iterable[int],
    feature_columns: list[str],
    top_positive: list[dict],
    top_negative: list[dict],
) -> None:
    artifacts_dir = Path("models") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "ridge_margin.joblib"
    metrics_path = artifacts_dir / "ridge_margin_metrics.json"

    joblib.dump(model, model_path)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_seasons": list(train_seasons),
        "test_seasons": list(test_seasons),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "mae": float(mae),
        "rmse": float(rmse),
        "top_positive_coefficients": top_positive,
        "top_negative_coefficients": top_negative,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    model_exists = model_path.exists()
    metrics_exists = metrics_path.exists()
    print(f"Model artifact: {model_path.resolve()} size={model_path.stat().st_size if model_exists else 'missing'}")
    print(
        f"Metrics artifact: {metrics_path.resolve()} size={metrics_path.stat().st_size if metrics_exists else 'missing'}"
    )
    if not model_exists or not metrics_exists:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
