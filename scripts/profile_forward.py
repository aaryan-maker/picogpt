import argparse
import cProfile
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pico import Tokenizer
from pico import model as model_mod
from pico.model import KVCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()

    m = model_mod.load(quantize=args.quantize)
    tok = Tokenizer()
    ids = tok.encode(" ".join(["hello"] * args.tokens))[: args.tokens]

    cache = None
    if args.cache:
        cache = KVCache(m.cfg, max_len=args.tokens + 2)
        m.forward(ids, cache, last_only=True)
        ids = [ids[-1]]

    for i in range(3):
        t = time.perf_counter()
        m.forward(ids, cache, last_only=True)
        print(f"forward #{i}: {(time.perf_counter() - t) * 1000:.1f} ms")

    out = os.path.join(tempfile.gettempdir(), "picogpt_prof.out")
    cProfile.runctx("[m.forward(ids, cache, last_only=True) for _ in range(10)]",
                    globals(), locals(), out)
    pstats.Stats(out).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":
    main()
