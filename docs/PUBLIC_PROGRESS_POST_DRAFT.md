# Public Progress Post Draft

I started the TS benchmark harness as a separate audit-first repo: `TS-Benchmarks`.

What exists now:

- deterministic synthetic graph scaling runs
- TS-Core-style tension relaxation reference
- contradiction-localization metrics
- simple graph baselines
- benchmark receipts with commit, command, seed, config, metrics, and artifact checksums
- first 100/1k/10k-node sweep across random, scale-free, and small-world graphs

Important limitation:

This is not a transformer-killer result. It is a scaling harness. The first sweep already exposed a failure mode: the initial reference relaxation handled random and small-world graphs much better than scale-free graphs, where final tension stayed high and contradiction localization failed under the first config.

That is exactly the point of the next phase: turn TS claims into measurements, including failures.
