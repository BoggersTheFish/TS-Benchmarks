"""Run one synthetic TS-Core scaling benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ts_benchmarks.baselines.graph_baselines import degree_baseline, pagerank_like_baseline
from ts_benchmarks.receipts import build_receipt, stable_hash, write_json
from ts_benchmarks.tasks.scaling import RelaxationConfig, generate_graph, run_relaxation


def run_one(args: argparse.Namespace, out_path: Path) -> dict[str, object]:
    graph = generate_graph(
        graph_type=args.graph,
        nodes=args.nodes,
        seed=args.seed,
        avg_degree=args.avg_degree,
        contradiction_rate=args.contradiction_rate,
    )
    config = RelaxationConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        damping=args.damping,
        tolerance=args.tolerance,
        frontier=not args.no_frontier,
        provenance_weighting=not args.no_provenance_weighting,
    )
    result = run_relaxation(graph, config)
    baselines = {
        "degree": degree_baseline(graph),
        "pagerank_like": pagerank_like_baseline(graph),
    }
    graph_payload = {
        "type": graph.spec.graph_type,
        "nodes": graph.spec.nodes,
        "edges": graph.spec.edges,
        "seed": graph.spec.seed,
        "contradiction_pairs": graph.spec.contradiction_pairs,
    }
    config_payload = {
        "seed": args.seed,
        "avg_degree": args.avg_degree,
        "contradiction_rate": args.contradiction_rate,
        "steps": config.steps,
        "learning_rate": config.learning_rate,
        "damping": config.damping,
        "tolerance": config.tolerance,
        "frontier": config.frontier,
        "provenance_weighting": config.provenance_weighting,
    }
    payload: dict[str, object] = {
        "run_id": f"scale-{args.graph}-{args.nodes}-seed{args.seed}",
        "graph": graph_payload,
        "config": config_payload,
        "metrics": result.metrics,
        "baselines": baselines,
        "tension_history": result.tension_history,
        "top_tension_nodes": result.top_tension_nodes,
    }
    write_json(out_path, payload)

    repo_root = Path(__file__).resolve().parents[2]
    dataset = {
        "name": f"synthetic-{args.graph}",
        "version": "v0.1.0",
        "hash": stable_hash(graph_payload),
    }
    receipt = build_receipt(
        run_id=str(payload["run_id"]),
        repo_root=repo_root,
        command=" ".join(sys.argv),
        dataset=dataset,
        system={"name": "ts-core-reference-relaxation", "variant": "sparse-frontier"},
        config=config_payload,
        metrics=result.metrics,
        artifacts=[out_path],
    )
    receipt_path = Path(args.receipt) if args.receipt else out_path.with_suffix(".receipt.json")
    write_json(receipt_path, receipt)
    payload["receipt_path"] = str(receipt_path)
    write_json(out_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument(
        "--graph",
        required=True,
        choices=["random", "scale_free", "small_world", "knowledge", "provenance", "temporal", "multi_context"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--avg-degree", type=int, default=6)
    parser.add_argument("--contradiction-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--no-frontier", action="store_true")
    parser.add_argument("--no-provenance-weighting", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_one(args, Path(args.out))
    metrics = payload["metrics"]
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "final_global_tension": metrics["final_global_tension"],
                "f1": metrics["contradiction_localization_f1"],
                "runtime_s": metrics["runtime_s"],
                "receipt_path": payload["receipt_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
