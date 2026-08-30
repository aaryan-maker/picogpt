from __future__ import annotations

import numpy as np

from .model import softmax


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    if temperature <= 0:
        return greedy(logits)

    rng = rng or np.random.default_rng()
    logits = logits.astype(np.float64) / temperature

    if top_k:
        top_k = min(top_k, logits.shape[-1])
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked

    probs = softmax(logits)

    if top_p is not None and top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cut = cum > top_p
        cut[0] = False
        probs[order[cut]] = 0.0
        probs /= probs.sum()

    return int(rng.choice(probs.shape[-1], p=probs))
