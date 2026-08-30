from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_ctx: int = 1024
    vocab_size: int = 50257
    eps: float = 1e-5

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


_F32 = np.float32
_GELU_C = _F32(np.sqrt(2.0 / np.pi))
_GELU_A = _F32(0.044715)


def gelu(x: np.ndarray) -> np.ndarray:
    return _F32(0.5) * x * (_F32(1.0) + np.tanh(_GELU_C * (x + _GELU_A * x**3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    var = x.var(axis=-1, keepdims=True, dtype=np.float32)
    return (x - mu) / np.sqrt(var + _F32(eps)) * w + b


def linear(x: np.ndarray, w, b: np.ndarray) -> np.ndarray:
    if type(w) is not np.ndarray:
        return (x @ w.q.astype(np.float32)) * w.scale + b
    return x @ w + b


class KVCache:
    def __init__(self, cfg: Config, max_len: int | None = None):
        self.max_len = min(max_len or cfg.n_ctx, cfg.n_ctx)
        shape = (cfg.n_layer, cfg.n_head, self.max_len, cfg.head_size)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class GPT2:
    def __init__(self, weights: dict[str, np.ndarray], cfg: Config = Config()):
        self.w = weights
        self.cfg = cfg

    def _attention(
        self, x: np.ndarray, i: int, cache: KVCache | None, start: int, mask: np.ndarray | None
    ) -> np.ndarray:
        w, cfg = self.w, self.cfg
        p = f"h.{i}."
        t_new, C = x.shape
        nh, hs = cfg.n_head, cfg.head_size

        qkv = linear(x, w[p + "attn.c_attn.weight"], w[p + "attn.c_attn.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)

        to_heads = lambda t: t.reshape(t_new, nh, hs).transpose(1, 0, 2)
        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        if cache is not None:
            cache.k[i, :, start:start + t_new] = k
            cache.v[i, :, start:start + t_new] = v
            k = cache.k[i, :, : start + t_new]
            v = cache.v[i, :, : start + t_new]

        att = (q @ k.transpose(0, 2, 1)) * _F32(1.0 / np.sqrt(hs))
        if mask is not None:
            att = att + mask

        y = softmax(att, axis=-1) @ v
        y = y.transpose(1, 0, 2).reshape(t_new, C)
        return linear(y, w[p + "attn.c_proj.weight"], w[p + "attn.c_proj.bias"])

    @staticmethod
    def _causal_mask(t_new: int, total: int, start: int) -> np.ndarray | None:
        if t_new == 1:
            return None
        rows = np.arange(t_new)[:, None] + start
        cols = np.arange(total)[None, :]
        return np.where(cols <= rows, _F32(0.0), _F32(-np.inf))

    def _mlp(self, x: np.ndarray, i: int) -> np.ndarray:
        w, p = self.w, f"h.{i}."
        h = gelu(linear(x, w[p + "mlp.c_fc.weight"], w[p + "mlp.c_fc.bias"]))
        return linear(h, w[p + "mlp.c_proj.weight"], w[p + "mlp.c_proj.bias"])

    def forward(
        self,
        ids: list[int] | np.ndarray,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> np.ndarray:
        w, cfg = self.w, self.cfg
        ids = np.asarray(ids, dtype=np.int64)
        t_new = ids.shape[0]
        start = cache.length if cache is not None else 0
        if start + t_new > cfg.n_ctx:
            raise ValueError(f"position {start + t_new} exceeds context window {cfg.n_ctx}")

        x = w["wte.weight"][ids] + w["wpe.weight"][start:start + t_new]
        mask = self._causal_mask(t_new, start + t_new, start)

        for i in range(cfg.n_layer):
            p = f"h.{i}."
            x = x + self._attention(
                layernorm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"], cfg.eps), i, cache, start, mask
            )
            x = x + self._mlp(
                layernorm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"], cfg.eps), i
            )

        if cache is not None:
            cache.length += t_new

        if last_only:
            x = x[-1:]
        x = layernorm(x, w["ln_f.weight"], w["ln_f.bias"], cfg.eps)
        return x @ w["wte.weight"].T

    def quantized(self) -> "GPT2":
        from .quant import quantize_weights
        return GPT2(quantize_weights(self.w), self.cfg)

    def weight_bytes(self) -> int:
        total = 0
        for v in self.w.values():
            total += v.nbytes if hasattr(v, "nbytes") else v.q.nbytes + v.scale.nbytes
        return total


def load(path: str | None = None, cfg: Config = Config(), quantize: bool = False) -> GPT2:
    from .weights import load as load_weights
    model = GPT2(load_weights(path), cfg)
    return model.quantized() if quantize else model
