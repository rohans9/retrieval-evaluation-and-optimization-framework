"""Generate HTML and PNG visualization artifacts for experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from retrieval_evaluation_framework.benchmarking.models import (
  ExperimentRecord,
  VisualizationArtifacts,
)

plt.switch_backend("Agg")


class VisualizationGenerator:
    """Export portable dashboard artifacts for experiment history."""

    def __init__(self, output_directory: Path) -> None:
        """Initialize the visualization generator."""
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def generate(self, experiments: list[ExperimentRecord]) -> VisualizationArtifacts:
        """Create HTML and PNG artifacts summarizing experiment history."""
        rankable = [
            experiment
            for experiment in experiments
            if experiment.retrieval_quality_metrics is not None
            and experiment.performance_metrics is not None
        ]
        html_paths = {
            "leaderboard": str(self._write_html("leaderboard", rankable)),
            "quality_vs_latency": str(self._write_html("quality_vs_latency", rankable)),
        }
        png_paths = {
            "leaderboard": str(self._write_leaderboard_png(rankable)),
            "quality_vs_latency": str(self._write_quality_latency_png(rankable)),
        }
        return VisualizationArtifacts(html_paths=html_paths, png_paths=png_paths)

    def _write_html(self, name: str, experiments: list[ExperimentRecord]) -> Path:
        path = self.output_directory / f"{name}.html"
        path.write_text(self._html_document(name, experiments), encoding="utf-8")
        return path

    def _write_leaderboard_png(self, experiments: list[ExperimentRecord]) -> Path:
        path = self.output_directory / "leaderboard.png"
        labels = [experiment.experiment_id for experiment in experiments]
        scores = [experiment.overall_score or 0.0 for experiment in experiments]
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(labels, scores, color="#c8553d")
        axis.set_title("Experiment Leaderboard Scores")
        axis.set_ylabel("Overall score")
        axis.set_xlabel("Experiment")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    def _write_quality_latency_png(self, experiments: list[ExperimentRecord]) -> Path:
        path = self.output_directory / "quality_vs_latency.png"
        figure, axis = plt.subplots(figsize=(8, 5))
        for experiment in experiments:
            quality = experiment.retrieval_quality_metrics
            performance = experiment.performance_metrics
            if quality is None or performance is None:
                continue
            axis.scatter(
                performance.average_retrieval_latency_ms,
                quality.mean_reciprocal_rank,
                label=experiment.experiment_id,
                s=80,
                alpha=0.8,
            )
            axis.annotate(
                experiment.experiment_id,
                (performance.average_retrieval_latency_ms, quality.mean_reciprocal_rank),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        axis.set_title("Quality vs Latency")
        axis.set_xlabel("Average latency (ms)")
        axis.set_ylabel("Mean reciprocal rank")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    @staticmethod
    def _html_document(name: str, experiments: list[ExperimentRecord]) -> str:
        rows: list[str] = []
        for experiment in experiments:
            quality = experiment.retrieval_quality_metrics
            performance = experiment.performance_metrics
            if quality is None or performance is None:
                continue
            rows.append(
              
                f"<tr>"
                f"<td>{experiment.experiment_id}</td>"
                f"<td>{experiment.retriever}</td>"
                f"<td>{experiment.embedding_model}</td>"
                f"<td>{quality.mean_reciprocal_rank:.4f}</td>"
                f"<td>{quality.ndcg_at_k:.4f}</td>"
                f"<td>{performance.average_retrieval_latency_ms:.2f}</td>"
                f"<td>{(experiment.overall_score or 0.0):.4f}</td>"
                f"</tr>"
              
            )
        table_rows = "\n".join(rows) or "<tr><td colspan='7'>No experiments available</td></tr>"
        title = name.replace("_", " ").title()
        return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{
        --bg: #f5efe6;
        --ink: #1f2933;
        --accent: #c8553d;
        --panel: #fffaf2;
        --line: #e8dac8;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background: radial-gradient(circle at top, #fff8ef 0%, var(--bg) 55%, #eadfce 100%);
        color: var(--ink);
      }}
      main {{
        max-width: 1000px;
        margin: 0 auto;
        padding: 48px 20px 80px;
      }}
      h1 {{
        font-size: clamp(2rem, 5vw, 4rem);
        margin-bottom: 12px;
      }}
      p {{
        max-width: 700px;
        line-height: 1.6;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 28px;
        background: var(--panel);
        border: 1px solid var(--line);
        box-shadow: 0 18px 40px rgba(80, 55, 30, 0.08);
      }}
      th, td {{
        padding: 14px 12px;
        text-align: left;
        border-bottom: 1px solid var(--line);
      }}
      th {{
        background: rgba(200, 85, 61, 0.08);
        color: var(--accent);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.78rem;
      }}
      tr:nth-child(even) td {{
        background: rgba(255, 255, 255, 0.45);
      }}
      @media (max-width: 720px) {{
        table, thead, tbody, th, td, tr {{
          display: block;
        }}
        thead {{
          display: none;
        }}
        tr {{
          margin-bottom: 14px;
          border: 1px solid var(--line);
          background: var(--panel);
        }}
        td {{
          border: none;
          padding: 10px 12px;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>{title}</h1>
      <p>
        Phase-4 visualization artifact combining retrieval quality, latency, and
        recommendation score into a portable dashboard.
      </p>
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Retriever</th>
            <th>Embedding</th>
            <th>MRR</th>
            <th>NDCG@K</th>
            <th>Avg Latency</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </main>
  </body>
</html>
"""
