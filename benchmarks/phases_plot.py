from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

MILESTONES = [
    ("naive\nbaseline", 0.2, "recompute every token, float64"),
    ("OPENBLAS\n_NUM_THREADS=4", 0.9, "stop 24-thread oversubscription"),
    ("float32\neverywhere", 5.2, "kill silent float64 promotion"),
]


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("needs matplotlib: .venv/Scripts/python -m pip install matplotlib")

    labels = [m[0] for m in MILESTONES]
    vals = [m[1] for m in MILESTONES]
    notes = [m[2] for m in MILESTONES]

    latest = RESULTS / "latest.json"
    if latest.exists():
        d = json.loads(latest.read_text())["results"]
        if "kv-cache" in d:
            labels.append("+ KV cache")
            vals.append(d["kv-cache"]["decode_tok_s"]["median"])
            notes.append("reuse past keys/values")
        if "kv-cache+int8" in d:
            labels.append("+ int8\n(memory)")
            vals.append(d["kv-cache+int8"]["decode_tok_s"]["median"])
            notes.append("4x less RAM, slower in NumPy")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#bbb"] * 3 + ["#4c78a8", "#e45756"][: len(vals) - 3]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("decode throughput (tokens / sec)")
    ax.set_title("picogpt - GPT-2 124M on Intel i7-10510U (CPU, NumPy)")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(axis="x", labelsize=9)

    caption = "   ".join(f"{lbl.strip().splitlines()[-1]}: {n}" for lbl, n in zip(labels, notes))
    fig.text(0.5, 0.03, caption, ha="center", fontsize=7.5, color="#555", wrap=True)
    fig.subplots_adjust(bottom=0.20)

    out = RESULTS / "phases.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
