# Ingest

Purpose: pull raw NBA game logs and betting odds into the bronze layer in Azure Blob Storage.

Notes:
- Keep ingestion idempotent and append-only.
- Log API requests, rate limits, and schema changes.
