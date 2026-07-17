#!/usr/bin/env bash
set -euo pipefail

# Final framework runner:
# - Baseline pipeline (ingest -> preprocess -> chunk -> embed/index -> benchmark -> reports)
# - Optuna (30 trials) for bm25/dense/hybrid
#
# Usage:
#   scripts/run_final_framework.sh
#   scripts/run_final_framework.sh --domain insurance
#   scripts/run_final_framework.sh --dry-run
#   scripts/run_final_framework.sh --domain legal --dry-run

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RETRIEVAL_BIN="${ROOT_DIR}/venv/bin/retrieval"
if [[ -x "$RETRIEVAL_BIN" ]]; then
  CLI="$RETRIEVAL_BIN"
elif command -v retrieval >/dev/null 2>&1; then
  CLI="retrieval"
else
  echo "ERROR: retrieval CLI not found. Install project env first (venv/bin/retrieval)." >&2
  exit 1
fi

DRY_RUN=0
DOMAIN_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --domain)
      DOMAIN_FILTER="${2:-}"
      if [[ -z "$DOMAIN_FILTER" ]]; then
        echo "ERROR: --domain requires one value: insurance|legal|news" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_cmd() {
  echo
  echo "$*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$*"
  fi
}

dataset_for_domain() {
  case "$1" in
    insurance) echo "insurance_eval_dataset.json" ;;
    legal) echo "legal_evaluation_dataset.json" ;;
    news) echo "news_evaluation_dataset.json" ;;
    *)
      echo "ERROR: Unknown domain '$1'" >&2
      exit 1
      ;;
  esac
}

config_for_retriever() {
  case "$1" in
    bm25) echo "configs/examples/bm25_only.yaml" ;;
    dense) echo "configs/examples/dense_retrieval.yaml" ;;
    hybrid) echo "configs/examples/hybrid_retrieval.yaml" ;;
    *)
      echo "ERROR: Unknown retriever '$1'" >&2
      exit 1
      ;;
  esac
}

search_space_for_retriever() {
  case "$1" in
    bm25) echo "configs/examples/optuna_search_space_bm25.yaml" ;;
    dense) echo "configs/examples/optuna_search_space_dense.yaml" ;;
    hybrid) echo "configs/examples/optuna_search_space_hybrid.yaml" ;;
    *)
      echo "ERROR: Unknown retriever '$1'" >&2
      exit 1
      ;;
  esac
}

DOMAINS=(insurance legal news)
if [[ -n "$DOMAIN_FILTER" ]]; then
  case "$DOMAIN_FILTER" in
    insurance|legal|news)
      DOMAINS=("$DOMAIN_FILTER")
      ;;
    *)
      echo "ERROR: Invalid domain: $DOMAIN_FILTER (use insurance|legal|news)" >&2
      exit 1
      ;;
  esac
fi

for DOMAIN in "${DOMAINS[@]}"; do
  DATASET_PATH="data/evaluation/$(dataset_for_domain "$DOMAIN")"
  CORPUS_PATH="data/processed/${DOMAIN}/processed_corpus.json"

  echo
  echo "============================================================"
  echo "Running BASELINE for domain: $DOMAIN"
  echo "============================================================"

  run_cmd "$CLI ingest data/raw/$DOMAIN --config configs/default.yaml"
  run_cmd "$CLI preprocess data/raw/$DOMAIN --config configs/default.yaml"
  run_cmd "$CLI chunk data/raw/$DOMAIN --config configs/default.yaml"

  run_cmd "$CLI retrieval embed --corpus-path $CORPUS_PATH --config configs/examples/dense_retrieval.yaml --output-path data/embeddings/$DOMAIN"

  run_cmd "$CLI retrieval index --corpus-path $CORPUS_PATH --config $(config_for_retriever bm25) --index-path data/index/$DOMAIN/bm25"
  run_cmd "$CLI retrieval index --corpus-path $CORPUS_PATH --config $(config_for_retriever dense) --index-path data/index/$DOMAIN/dense"
  run_cmd "$CLI retrieval index --corpus-path $CORPUS_PATH --config $(config_for_retriever hybrid) --index-path data/index/$DOMAIN/hybrid"

  run_cmd "$CLI retrieval benchmark --corpus-path $CORPUS_PATH --dataset-path $DATASET_PATH --config $(config_for_retriever bm25) --experiment-directory reports/experiments/final_baseline/$DOMAIN --notes baseline:$DOMAIN:bm25"
  run_cmd "$CLI retrieval benchmark --corpus-path $CORPUS_PATH --dataset-path $DATASET_PATH --config $(config_for_retriever dense) --experiment-directory reports/experiments/final_baseline/$DOMAIN --notes baseline:$DOMAIN:dense"
  run_cmd "$CLI retrieval benchmark --corpus-path $CORPUS_PATH --dataset-path $DATASET_PATH --config $(config_for_retriever hybrid) --experiment-directory reports/experiments/final_baseline/$DOMAIN --notes baseline:$DOMAIN:hybrid"

  run_cmd "$CLI retrieval experiments --config configs/default.yaml --experiment-directory reports/experiments/final_baseline/$DOMAIN"
  run_cmd "$CLI retrieval leaderboard --config configs/default.yaml --experiment-directory reports/experiments/final_baseline/$DOMAIN"
  run_cmd "$CLI retrieval report --config configs/default.yaml --experiment-directory reports/experiments/final_baseline/$DOMAIN"
  run_cmd "$CLI retrieval visualize --config configs/default.yaml --experiment-directory reports/experiments/final_baseline/$DOMAIN"

  echo
  echo "============================================================"
  echo "Running OPTUNA (30 trials) for domain: $DOMAIN"
  echo "============================================================"

  for RETRIEVER in bm25 dense hybrid; do
    run_cmd "$CLI retrieval optuna-search --corpus-path $CORPUS_PATH --dataset-path $DATASET_PATH --search-space-path $(search_space_for_retriever "$RETRIEVER") --config $(config_for_retriever "$RETRIEVER") --trials 30 --objective ndcg --seed 42 --experiment-directory reports/experiments/optuna_final/$DOMAIN/$RETRIEVER --notes optuna:$DOMAIN:$RETRIEVER:final"
  done
done

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete. No commands were executed."
else
  echo "Final framework run complete."
fi
