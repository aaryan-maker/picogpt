import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pico import Tokenizer, generate
from pico import model as model_mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=50)
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()

    prompt = " ".join(args.prompt)

    t0 = time.perf_counter()
    m = model_mod.load(quantize=args.quantize)
    print(f"[loaded weights in {time.perf_counter() - t0:.1f}s]\n", file=sys.stderr)

    tok = Tokenizer()

    print(prompt, end="", flush=True)
    res = generate(
        m, tok, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_cache=not args.no_cache,
    )

    print(
        f"\n--- {res.new_tokens} tokens | "
        f"prefill {res.prefill_s * 1000:.0f}ms ({res.prompt_tokens} tok) | "
        f"decode {res.tokens_per_s:.1f} tok/s ---",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
