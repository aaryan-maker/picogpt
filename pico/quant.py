from __future__ import annotations

from typing import NamedTuple

import numpy as np

_QUANTIZE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


class Q8(NamedTuple):
    q: np.ndarray
    scale: np.ndarray

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes

    @property
    def shape(self):
        return self.q.shape


def quantize_matrix(w: np.ndarray) -> Q8:
    scale = np.abs(w).max(axis=0) / 127.0
    scale[scale == 0.0] = 1.0
    q = np.round(w / scale).clip(-127, 127).astype(np.int8)
    return Q8(q, scale.astype(np.float32))


def dequantize(qw: Q8) -> np.ndarray:
    return qw.q.astype(np.float32) * qw.scale


def quantize_weights(weights: dict) -> dict:
    out = {}
    for k, v in weights.items():
        if k.endswith(_QUANTIZE_SUFFIXES):
            out[k] = quantize_matrix(v)
        else:
            out[k] = v
    return out


def max_abs_error(weights: dict) -> float:
    worst = 0.0
    for k, v in weights.items():
        if k.endswith(_QUANTIZE_SUFFIXES):
            worst = max(worst, float(np.abs(dequantize(quantize_matrix(v)) - v).max()))
    return worst
