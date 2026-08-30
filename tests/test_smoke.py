import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pico import Tokenizer
from pico import model as model_mod
from pico.model import KVCache, gelu, layernorm, softmax

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = model_mod.load()
    return _MODEL


def test_softmax_sums_to_one():
    x = np.random.randn(4, 7)
    assert np.allclose(softmax(x).sum(axis=-1), 1.0)


def test_layernorm_zero_mean_unit_var():
    x = np.random.randn(3, 16) * 5 + 2
    y = layernorm(x, np.ones(16), np.zeros(16), 1e-5)
    assert np.allclose(y.mean(-1), 0, atol=1e-5)
    assert np.allclose(y.var(-1), 1, atol=1e-3)


def test_gelu_shape_and_sign():
    x = np.array([-3.0, 0.0, 3.0])
    y = gelu(x)
    assert y.shape == x.shape
    assert y[1] == 0.0 and y[2] > 2.9 and -0.1 < y[0] < 0.0


def test_forward_shapes_and_dtype():
    tok = Tokenizer()
    ids = tok.encode("The capital of France is")
    logits = get_model().forward(ids)
    assert logits.shape == (len(ids), get_model().cfg.vocab_size)
    assert logits.dtype == np.float32


def test_forward_follows_in_context_pattern():
    tok = Tokenizer()
    prompt = "The cat sat on the mat. The cat sat on the mat. The cat sat on the"
    ids = tok.encode(prompt)
    nxt = int(get_model().forward(ids)[-1].argmax())
    assert tok.decode([nxt]).strip() == "mat"


def test_kv_cache_matches_stateless():
    tok = Tokenizer()
    m = get_model()
    ids = tok.encode("In a hole in the ground there lived a hobbit.")

    full = m.forward(ids)

    cache = KVCache(m.cfg, max_len=len(ids) + 4)
    split = 5
    a = m.forward(ids[:split], cache)
    b = m.forward(ids[split:], cache)
    incremental = np.vstack([a, b])

    assert np.allclose(full, incremental, atol=2e-3)
    assert cache.length == len(ids)


def test_kv_cache_token_by_token():
    tok = Tokenizer()
    m = get_model()
    ids = tok.encode("The quick brown fox jumps over the lazy")

    ref_next = int(m.forward(ids)[-1].argmax())

    cache = KVCache(m.cfg, max_len=len(ids) + 1)
    logits = None
    for t in ids:
        logits = m.forward([t], cache)
    assert int(logits[-1].argmax()) == ref_next
