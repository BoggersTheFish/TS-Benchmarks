# TS-Benchmarks

`TS-Benchmarks` is the audit-first benchmark harness for the Thinking System ecosystem.

The first implemented slice is Workstream A: scalable graph/tension tests for TS-Core-style relaxation. It creates deterministic synthetic graphs, injects contradictions, runs a sparse active-frontier relaxation loop, compares against simple baselines, and writes auditable receipts.

## Current Status

Implemented:

- deterministic graph generators: random, scale-free, small-world, knowledge-like, provenance, temporal, multi-context
- TS-style tension relaxation reference implementation
- degree and PageRank-like localization baselines
- scaling CLIs with JSON receipts
- receipt JSON schema
- markdown/CSV report generation
- smoke tests

Not implemented yet:

- hard reasoning benchmark runners
- TensionLM matched softmax campaign
- GPU/Triton hardware microbenchmarks
- hybrid Graph+LLM contradiction demo

## Quickstart

```bash
python3 -m unittest discover -s tests -p 'test*.py'
python3 -m ts_benchmarks.runners.scale_graph --nodes 1000 --graph scale_free --seed 42 --out artifacts/scaling/scale_free_1000.json
python3 -m ts_benchmarks.runners.scale_sweep --sizes 100,1000,10000 --graphs random,scale_free,small_world --seed 42 --out-dir artifacts/scaling
python3 -m ts_benchmarks.reports.plot_scaling --in-dir artifacts/scaling --out-dir artifacts/scaling/report
```

## Claim Boundary

This repo does not prove TS is a transformer replacement. It measures specific graph/tension behavior under specified configs and emits receipts that can be audited.
