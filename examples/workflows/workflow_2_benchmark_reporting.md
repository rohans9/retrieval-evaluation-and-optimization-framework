# Workflow 2: Benchmark To Recommendation

## Goal

Run a benchmark and generate actionable phase-4 outputs.

## Steps

1. Chunk sample corpus.
2. Run benchmark evaluation.
3. Generate report artifacts.
4. Generate visualizations.
5. Inspect leaderboard and recommendation.

## Commands

```bash
retrieval chunk examples/sample_data/documents --config configs/examples/benchmark_experiment.yaml
retrieval retrieval benchmark --corpus-path data/processed/processed_corpus.json --dataset-path examples/sample_data/evaluation_dataset.yaml --config configs/examples/benchmark_experiment.yaml
retrieval retrieval report --config configs/examples/benchmark_experiment.yaml
retrieval retrieval visualize --config configs/examples/benchmark_experiment.yaml
retrieval retrieval leaderboard --config configs/examples/benchmark_experiment.yaml
retrieval retrieval recommend --config configs/examples/benchmark_experiment.yaml
```
