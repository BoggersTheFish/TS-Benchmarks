"""Generate scaling CSV, markdown, and optional PNG plots from run JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_runs(in_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(in_dir.glob("*.json")):
        if path.name.endswith(".receipt.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "graph" in payload and "metrics" in payload:
            payload["_path"] = str(path)
            runs.append(payload)
    return runs


def write_csv(runs: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "graph_type",
        "nodes",
        "edges",
        "runtime_s",
        "peak_rss_mb",
        "iterations",
        "initial_global_tension",
        "final_global_tension",
        "contradiction_localization_f1",
        "edges_relaxed_per_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "run_id": run["run_id"],
                "graph_type": run["graph"]["type"],
                "nodes": run["graph"]["nodes"],
                "edges": run["graph"]["edges"],
            }
            row.update({field: run["metrics"].get(field) for field in fields if field not in row})
            writer.writerow(row)


def write_markdown(runs: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TS-Core Scaling Report",
        "",
        "This report is generated from local benchmark receipts. It is not a capability claim.",
        "",
        "| Run | Graph | Nodes | Edges | Runtime s | Peak MB | Final tension | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        metrics = run["metrics"]
        lines.append(
            "| {run_id} | {graph} | {nodes} | {edges} | {runtime:.4f} | {mem:.2f} | {tension:.6f} | {f1:.3f} |".format(
                run_id=run["run_id"],
                graph=run["graph"]["type"],
                nodes=run["graph"]["nodes"],
                edges=run["graph"]["edges"],
                runtime=float(metrics["runtime_s"]),
                mem=float(metrics["peak_rss_mb"]),
                tension=float(metrics["final_global_tension"]),
                f1=float(metrics["contradiction_localization_f1"]),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_optional_plots(runs: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return []

    out_paths: list[Path] = []
    by_graph: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_graph.setdefault(run["graph"]["type"], []).append(run)

    for metric, filename, ylabel in [
        ("runtime_s", "runtime_vs_nodes.png", "Runtime (s)"),
        ("peak_rss_mb", "memory_vs_nodes.png", "Peak traced memory (MB)"),
        ("contradiction_localization_f1", "localization_f1_vs_nodes.png", "Localization F1"),
    ]:
        plt.figure()
        for graph_type, graph_runs in sorted(by_graph.items()):
            graph_runs = sorted(graph_runs, key=lambda run: run["graph"]["nodes"])
            plt.plot(
                [run["graph"]["nodes"] for run in graph_runs],
                [run["metrics"][metric] for run in graph_runs],
                marker="o",
                label=graph_type,
            )
        plt.xscale("log")
        plt.xlabel("Nodes")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / filename
        plt.savefig(out_path)
        plt.close()
        out_paths.append(out_path)
    return out_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(in_dir)
    write_csv(runs, out_dir / "scaling_summary.csv")
    write_markdown(runs, out_dir / "SCALING_REPORT.md")
    plot_paths = write_optional_plots(runs, out_dir)
    print(f"wrote {len(runs)} runs to {out_dir}; plots={len(plot_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
