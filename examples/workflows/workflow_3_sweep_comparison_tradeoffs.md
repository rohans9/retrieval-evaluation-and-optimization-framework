# Workflow 3: Sweep, Compare, Analyze Trade-Offs

## Goal

Explore configuration trade-offs with sweep and comparison workflows.

## Steps

1. Chunk sample corpus.
2. Run parameter sweep.
3. List experiments and choose IDs.
4. Compare selected experiments.
5. Inspect enriched history for trade-offs and artifacts.

## Commands

```bash
retrieval chunk examples/sample_data/documents --config configs/examples/benchmark_experiment.yaml
retrieval retrieval sweep --corpus-path data/processed/processed_corpus.json --dataset-path examples/sample_data/evaluation_dataset.yaml --parameters-path configs/examples/parameter_sweep.yaml --config configs/examples/benchmark_experiment.yaml
retrieval retrieval experiments --config configs/examples/benchmark_experiment.yaml
retrieval retrieval compare --experiment-ids exp-123,exp-456 --config configs/examples/benchmark_experiment.yaml
retrieval retrieval history --config configs/examples/benchmark_experiment.yaml
```
