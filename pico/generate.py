from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from .model import GPT2, KVCache
from .sample import greedy, sample
from .tokenizer import Tokenizer


@dataclass
class Result:
    text: str
    prompt_tokens: int
    new_tokens: int
    prefill_s: float
    decode_s: float
    tokens_per_s: float
    used_cache: bool


def generate(
    model: GPT2,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    stream: bool = True,
    use_cache: bool = True,
) -> Result:
    ids = tok.encode(prompt)
    prompt_len = len(ids)
    rng = np.random.default_rng(seed)
    new_ids: list[int] = []

    cache: KVCache | None = None
    if use_cache:
        cache = KVCache(model.cfg, max_len=prompt_len + max_new_tokens)

    def step(feed: list[int]) -> np.ndarray:
        return model.forward(feed if use_cache else ids, cache, last_only=True)[-1]

    t0 = time.perf_counter()
    logits = model.forward(ids, cache, last_only=True)[-1]
    prefill_s = time.perf_counter() - t0

    decode_s = 0.0
    decode_steps = 0
    for _ in range(max_new_tokens):
        nxt = greedy(logits) if temperature <= 0 else sample(
            logits, temperature, top_k, top_p, rng
        )
        if nxt == tok.eot:
            break
        ids.append(nxt)
        new_ids.append(nxt)
        if stream:
            sys.stdout.write(tok.decode([nxt]))
            sys.stdout.flush()
        if len(ids) >= model.cfg.n_ctx:
            break

        t0 = time.perf_counter()
        logits = step([nxt])
        decode_s += time.perf_counter() - t0
        decode_steps += 1

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return Result(
        text=tok.decode(new_ids),
        prompt_tokens=prompt_len,
        new_tokens=len(new_ids),
        prefill_s=prefill_s,
        decode_s=decode_s,
        tokens_per_s=(decode_steps / decode_s) if decode_s > 0 else 0.0,
        used_cache=use_cache,
    )
