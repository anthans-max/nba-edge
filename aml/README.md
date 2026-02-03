# Azure ML (v2)

This folder provides minimal Azure ML v2 job specs for feature building, training, and backtesting.

## Prereqs
- Azure ML CLI extension installed.
- Logged in to Azure CLI.
- An existing Azure ML workspace.

## Run a job
1) Update placeholders in the YAML files:
   - Workspace name
   - Resource group
   - Datastore paths
2) Submit a job:
```
az ml job create --file aml/jobs/feature_job.yml
```

## Notes
- Jobs are defined as command jobs and use a simple conda environment.
- The pipeline chains feature -> train -> backtest in a single submission.
