# benchmarks

```bash
.venv/Scripts/python benchmarks/bench.py [--new-tokens N] [--reps K] [--int8] [--plot]
.venv/Scripts/python benchmarks/phases_plot.py
```

`bench.py` runs a fixed prompt to a fixed length with greedy decoding, `K` times
per config, and reports median + p10-p90 for prefill, decode tok/s, per-position
decode latency, and peak working set. Configs: `no-cache`, `kv-cache`, and with
`--int8` also `kv-cache+int8`. Output goes to `results/` as JSON plus (with
`--plot`) `results/latest.png`.

`phases_plot.py` writes `results/phases.png` — the first three bars are
historical, the last two read from `results/latest.json`.
