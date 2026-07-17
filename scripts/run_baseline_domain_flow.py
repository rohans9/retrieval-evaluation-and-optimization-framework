#!/usr/bin/env python3
"""Run the full non-Optuna baseline flow for one domain.

Usage:
    python scripts/run_baseline_domain_flow.py --domain insurance
    python scripts/run_baseline_domain_flow.py --domain legal
    python scripts/run_baseline_domain_flow.py --domain news
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DATASET_BY_DOMAIN = {
    "insurance": "insurance_eval_dataset.json",
    "legal": "legal_evaluation_dataset.json",
    "news": "news_evaluation_dataset.json",
}

CHUNK_STRATEGIES = ("fixed", "recursive", "semantic")

EVAL_COMBOS = (
    ("bm25_none", "bm25", []),
    ("dense_none", "dense", []),
    ("hybrid_none", "hybrid", []),
)


def run_command(args: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd), check=True)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_chunk_configs(base_cfg: dict[str, Any], cfg_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for strategy in CHUNK_STRATEGIES:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["chunking"]["strategy"] = strategy
        cfg["output"]["output_directory"] = f"data/processed/flow_{strategy}"
        cfg_path = cfg_dir / f"chunk_{strategy}.yaml"
        save_yaml(cfg_path, cfg)
        out[strategy] = cfg_path
    return out


def build_index_configs(base_cfg: dict[str, Any], cfg_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for retriever in ("bm25", "dense"):
        cfg = json.loads(json.dumps(base_cfg))
        cfg["retrieval"]["retriever"] = retriever
        cfg_path = cfg_dir / f"index_{retriever}.yaml"
        save_yaml(cfg_path, cfg)
        out[retriever] = cfg_path
    return out


def build_eval_configs(base_cfg: dict[str, Any], cfg_dir: Path, domain: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, retriever, _ in EVAL_COMBOS:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["retrieval"]["retriever"] = retriever

        cfg["benchmark"]["dataset_path"] = f"data/evaluation/{DATASET_BY_DOMAIN[domain]}"
        cfg["benchmark"]["experiment_directory"] = f"reports/experiments/flow_once_{domain}"
        cfg["benchmark"]["results_directory"] = f"reports/benchmarks/flow_once_{domain}"
        cfg["reranking"]["enabled"] = False
        cfg["retrieval"]["top_k"] = 10

        cfg_path = cfg_dir / f"eval_{name}.yaml"
        save_yaml(cfg_path, cfg)
        out[name] = cfg_path

    return out


def summarize_results(repo_root: Path, domain: str, run_id: str) -> None:
    exp_dir = repo_root / "reports" / "experiments" / f"flow_once_{domain}"
    note_prefix = f"flow_once:{domain}:{run_id}:"
    rows: list[tuple[float, str, float, float, str]] = []

    for path in sorted(exp_dir.glob("exp-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        notes = payload.get("notes") or ""
        if not notes.startswith(note_prefix):
            continue
        metrics = payload.get("retrieval_quality_metrics") or {}
        ndcg = float(metrics.get("ndcg_at_k") or 0.0)
        mrr = float(metrics.get("mean_reciprocal_rank") or 0.0)
        rows.append(
            (
                ndcg,
                notes.replace(note_prefix, ""),
                mrr,
                ndcg,
                str(path.relative_to(repo_root)),
            )
        )

    if not rows:
        print("\nNo matching experiment results were found for this run.")
        return

    rows.sort(reverse=True)
    print("\nSummary (sorted by nDCG@k):")
    for _, combo, mrr, ndcg, rel_path in rows:
        print(f"- {combo:14s}  nDCG={ndcg:.4f}  MRR={mrr:.4f}  ({rel_path})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full non-Optuna baseline flow for one domain."
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=sorted(DATASET_BY_DOMAIN.keys()),
        help="Domain to run (insurance, legal, news).",
    )
    parser.add_argument(
        "--retrieval-cmd",
        default="retrieval",
        help="CLI command name for the framework (default: retrieval).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    domain = args.domain
    retrieval_cmd = args.retrieval_cmd
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_domain_dir = repo_root / "data" / "raw" / domain
    dataset_file = repo_root / "data" / "evaluation" / DATASET_BY_DOMAIN[domain]

    if not raw_domain_dir.exists():
        print(f"Missing raw domain directory: {raw_domain_dir}", file=sys.stderr)
        return 1
    if not dataset_file.exists():
        print(f"Missing evaluation dataset file: {dataset_file}", file=sys.stderr)
        return 1

    temp_cfg_dir = Path("/tmp") / "reoflow" / domain / "configs"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    (repo_root / "reports" / "experiments" / f"flow_once_{domain}").mkdir(
        parents=True, exist_ok=True
    )
    (repo_root / "reports" / "benchmarks" / f"flow_once_{domain}").mkdir(
        parents=True, exist_ok=True
    )

    base_cfg = load_yaml(repo_root / "configs" / "default.yaml")
    chunk_cfgs = build_chunk_configs(base_cfg, temp_cfg_dir)
    index_cfgs = build_index_configs(base_cfg, temp_cfg_dir)
    eval_cfgs = build_eval_configs(base_cfg, temp_cfg_dir, domain)

    print(f"Running full baseline flow for domain: {domain}")
    print(f"Run ID: {run_id}")

    # Stage 1: Ingestion
    run_command(
        [retrieval_cmd, "ingest", f"data/raw/{domain}", "--config", "configs/default.yaml"],
        repo_root,
    )

    # Stage 2: Preprocessing
    run_command(
        [
            retrieval_cmd,
            "preprocess",
            f"data/raw/{domain}",
            "--config",
            "configs/default.yaml",
        ],
        repo_root,
    )

    # Stage 3: Chunking (fixed, recursive, semantic)
    for strategy in CHUNK_STRATEGIES:
        run_command(
            [
                retrieval_cmd,
                "chunk",
                f"data/raw/{domain}",
                "--config",
                str(chunk_cfgs[strategy]),
            ],
            repo_root,
        )

    # Stage 4: Index creation from recursive corpus
    recursive_corpus = f"data/processed/flow_recursive/{domain}/processed_corpus.json"
    run_command(
        [
            retrieval_cmd,
            "retrieval",
            "index",
            "--corpus-path",
            recursive_corpus,
            "--config",
            str(index_cfgs["bm25"]),
            "--index-path",
            f"data/index/flow_recursive/{domain}_bm25",
        ],
        repo_root,
    )
    run_command(
        [
            retrieval_cmd,
            "retrieval",
            "index",
            "--corpus-path",
            recursive_corpus,
            "--config",
            str(index_cfgs["dense"]),
            "--index-path",
            f"data/index/flow_recursive/{domain}_dense",
        ],
        repo_root,
    )

    # Stages 5-10: Evaluate all retriever/enhancement combinations
    for combo, _, _ in EVAL_COMBOS:
        run_command(
            [
                retrieval_cmd,
                "retrieval",
                "evaluate",
                "--corpus-path",
                recursive_corpus,
                "--dataset-path",
                f"data/evaluation/{DATASET_BY_DOMAIN[domain]}",
                "--config",
                str(eval_cfgs[combo]),
                "--experiment-directory",
                f"reports/experiments/flow_once_{domain}",
                "--notes",
                f"flow_once:{domain}:{run_id}:{combo}",
            ],
            repo_root,
        )

    # Stage 11: Report generation
    run_command(
        [
            retrieval_cmd,
            "retrieval",
            "report",
            "--config",
            str(eval_cfgs["hybrid_none"]),
            "--experiment-directory",
            f"reports/experiments/flow_once_{domain}",
        ],
        repo_root,
    )

    summarize_results(repo_root, domain, run_id)

    print("\nFlow complete.")
    print(f"- Experiments: reports/experiments/flow_once_{domain}")
    print(f"- Benchmarks:  reports/benchmarks/flow_once_{domain}")
    print("- Generated reports: reports/generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
