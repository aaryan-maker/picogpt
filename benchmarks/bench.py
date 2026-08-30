from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pico import Tokenizer
from pico import model as model_mod
from pico.model import KVCache
from pico.sample import greedy

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PROMPT = (
    "The history of computing is a long and winding road. It begins with the "
    "abacus and the mechanical calculators of the seventeenth century, and it "
    "runs through Babbage, Lovelace, Turing, and von Neumann before arriving at"
)


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if fn(ctypes.c_void_p(-1), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return float("nan")


def run_once(model, ids: list[int], new_tokens: int, use_cache: bool) -> dict:
    gc.collect()
    cache = KVCache(model.cfg, max_len=len(ids) + new_tokens) if use_cache else None
    seq = list(ids)

    t0 = time.perf_counter()
    logits = model.forward(seq, cache, last_only=True)[-1]
    prefill_s = time.perf_counter() - t0

    step_times: list[float] = []
    for _ in range(new_tokens):
        nxt = greedy(logits)
        seq.append(nxt)
        t0 = time.perf_counter()
        feed = [nxt] if use_cache else seq
        logits = model.forward(feed, cache, last_only=True)[-1]
        step_times.append(time.perf_counter() - t0)

    return {
        "prefill_s": prefill_s,
        "step_times": step_times,
        "decode_tok_s": len(step_times) / sum(step_times),
    }


def summarize(runs: list[dict]) -> dict:
    prefill = sorted(r["prefill_s"] for r in runs)
    decode = sorted(r["decode_tok_s"] for r in runs)
    per_pos = np.mean([r["step_times"] for r in runs], axis=0).tolist()

    def band(xs):
        return {
            "median": statistics.median(xs),
            "p10": xs[max(0, int(0.1 * (len(xs) - 1)))],
            "p90": xs[min(len(xs) - 1, int(0.9 * (len(xs) - 1)))],
        }

    return {
        "prefill_ms": {k: v * 1000 for k, v in band(prefill).items()},
        "decode_tok_s": band(decode),
        "decode_latency_ms_by_pos": [t * 1000 for t in per_pos],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    print(f"loading model...  (threads={os.environ['OPENBLAS_NUM_THREADS']})")
    model = model_mod.load()
    tok = Tokenizer()
    ids = tok.encode(PROMPT)
    print(f"prompt: {len(ids)} tokens | generating {args.new_tokens} | {args.reps} reps\n")

    configs = [
        ("no-cache", model, False),
        ("kv-cache", model, True),
    ]
    if args.int8:
        print("quantizing to int8...")
        configs.append(("kv-cache+int8", model.quantized(), True))
    results = {}

    for name, mdl, use_cache in configs:
        for _ in range(args.warmup):
            run_once(mdl, ids, min(8, args.new_tokens), use_cache)
        runs = [run_once(mdl, ids, args.new_tokens, use_cache) for _ in range(args.reps)]
        results[name] = summarize(runs)
        results[name]["weight_mb"] = mdl.weight_bytes() / (1024 * 1024)
        s = results[name]
        print(
            f"{name:10}  prefill {s['prefill_ms']['median']:7.1f} ms "
            f"(p10-p90 {s['prefill_ms']['p10']:.0f}-{s['prefill_ms']['p90']:.0f})   "
            f"decode {s['decode_tok_s']['median']:6.2f} tok/s "
            f"(p10-p90 {s['decode_tok_s']['p10']:.2f}-{s['decode_tok_s']['p90']:.2f})"
        )

    print("\ndecode latency, first -> last generated token:")
    for name in results:
        lat = results[name]["decode_latency_ms_by_pos"]
        print(f"  {name:10}  {lat[0]:6.1f} ms  ->  {lat[-1]:6.1f} ms")

    speedup = results["kv-cache"]["decode_tok_s"]["median"] / results["no-cache"]["decode_tok_s"]["median"]
    print(f"\nKV-cache decode speedup: {speedup:.2f}x")

    payload = {
        "when": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "openblas_threads": os.environ["OPENBLAS_NUM_THREADS"],
        },
        "config": {"prompt_tokens": len(ids), "new_tokens": args.new_tokens, "reps": args.reps},
        "peak_working_set_mb": _rss_mb(),
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"bench-{stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=2))
    print(f"\npeak working set: {payload['peak_working_set_mb']:.0f} MB")
    print(f"saved {out.relative_to(RESULTS_DIR.parent.parent)}")

    if args.plot:
        _plot(payload, RESULTS_DIR / "latest.png")


def _plot(payload: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return

    res = payload["results"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for name in res:
        ax1.plot(res[name]["decode_latency_ms_by_pos"], label=name)
    ax1.set_xlabel("generated token #")
    ax1.set_ylabel("decode latency (ms)")
    ax1.set_title("per-token decode latency")
    ax1.legend()
    ax1.grid(alpha=0.3)

    names = list(res)
    tok_s = [res[n]["decode_tok_s"]["median"] for n in names]
    palette = ["#bbb", "#4c78a8", "#e45756", "#54a24b"]
    ax2.bar(names, tok_s, color=palette[: len(names)])
    ax2.set_ylabel("tokens / sec")
    ax2.set_title("decode throughput (median)")
    for i, v in enumerate(tok_s):
        ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    fig.suptitle(f"picogpt / GPT-2 124M / {payload['machine']['processor'][:40]}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"saved {path.name}")


if __name__ == "__main__":
    main()
