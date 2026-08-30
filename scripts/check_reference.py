import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pico import Tokenizer
from pico import model as model_mod
from pico.model import KVCache

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a hole in the ground there lived a hobbit."
)


def kv_cache_logits(model, ids):
    cache = KVCache(model.cfg, max_len=len(ids) + 1)
    rows = [model.forward(ids[:1], cache)]
    for t in ids[1:]:
        rows.append(model.forward([t], cache))
    return np.vstack(rows)


def report(label, ours, ref, tok, tol, exact=True):
    diff = np.abs(ours - ref)
    agree = float((ours.argmax(-1) == ref.argmax(-1)).mean())
    last_ok = int(ours[-1].argmax()) == int(ref[-1].argmax())
    nxt = int(ours[-1].argmax())
    print(f"  {label:16}  max |dlogit| {diff.max():8.5f}   mean {diff.mean():.6f}   "
          f"argmax agree {agree:5.1%}   next -> {tok.decode([nxt])!r}")
    if exact:
        return diff.max() < tol and agree == 1.0
    return diff.max() < tol and last_ok and agree >= 0.9


def main() -> None:
    tok = Tokenizer()
    ids = tok.encode(PROMPT)

    base = model_mod.load()

    import torch
    from transformers import GPT2LMHeadModel

    ref_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    with torch.no_grad():
        ref = ref_model(torch.tensor([ids])).logits[0].numpy()

    print(f"prompt: {len(ids)} tokens\n")
    ok = True
    ok &= report("stateless", base.forward(ids), ref, tok, tol=1e-3)
    ok &= report("kv-cache", kv_cache_logits(base, ids), ref, tok, tol=2e-3)
    ok &= report("int8 + kv-cache", kv_cache_logits(base.quantized(), ids), ref, tok, tol=4.0, exact=False)

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
