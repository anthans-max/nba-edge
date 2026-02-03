# nba-edge

## Project overview
Production-grade, reusable template for building NBA betting edge models. This repo separates ingestion, transformation, feature engineering, training/backtesting, and UI delivery for clear ownership and scalability.

## Architecture and data flow (text diagram)

NBA API + The Odds API
        |
        v
[Ingest: raw/bronze] -> Azure Blob Storage
        |
        v
[Transform: silver] -> Clean, standardized tables
        |
        v
[Features] -> Feature tables for training/inference
        |
        v
[Train + Backtest] -> Models + reports in Azure ML
        |
        v
[Streamlit UI] -> Matchup explorer + backtest reports
        |
        v
[Azure Container Apps] -> Deployment

## Layered pipeline summary
- Raw/Bronze: unmodified API snapshots for reproducibility.
- Silver: standardized schemas, canonical team IDs, and quality checks.
- Features: deterministic, versioned feature tables for modeling.
- Model: training and evaluation in Azure ML with registered artifacts.
- UI: read-only dashboard for predictions and backtests.

## Repository layout
- `ingest/`: API pulls and raw landing.
- `transform/`: silver layer builders.
- `features/`: feature engineering.
- `train/`: training and backtests.
- `app/`: Streamlit UI.
- `infra/`: deployment assets.
- `data_contracts/`: shared mappings and schema references.
- `scripts/`: local orchestration helpers.

## Dev setup
```
make venv
make install
pre-commit install
make pipeline
```

Note: set `GEMINI_API_KEY` to enable the Ask Lucky chat assistant in the Streamlit app.

## Azure ML (v2) quickstart
```
az extension add -n ml
az ml workspace show --name WORKSPACE_NAME_PLACEHOLDER --resource-group RESOURCE_GROUP_PLACEHOLDER
az ml job create --file aml/jobs/feature_job.yml
```
