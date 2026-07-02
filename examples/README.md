# Examples

This folder provides ready-to-run assets for phase-5 onboarding.

## Included

- `sample_data/documents`: sample corpus files.
- `sample_data/evaluation_dataset.yaml`: labeled evaluation dataset.
- `sample_data/retrieval_queries.yaml`: example retrieval prompts.
- `sample_data/benchmark_config.yaml`: example benchmark asset mapping.
- `workflows/workflow_1_retrieval_pipeline.md`: ingestion-to-retrieval walkthrough.
- `workflows/workflow_2_benchmark_reporting.md`: benchmark-to-recommendation walkthrough.
- `workflows/workflow_3_sweep_comparison_tradeoffs.md`: sweep/compare/trade-off walkthrough.

## Quick Start

```bash
retrieval chunk examples/sample_data/documents --config configs/examples/small_collection.yaml
retrieval retrieval benchmark --corpus-path data/processed/processed_corpus.json --dataset-path examples/sample_data/evaluation_dataset.yaml --config configs/examples/benchmark_experiment.yaml
retrieval retrieval recommend --config configs/examples/benchmark_experiment.yaml
```
