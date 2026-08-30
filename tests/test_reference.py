import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pico import Tokenizer
from pico import model as model_mod
from pico.model import KVCache

PROMPT = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole."


@pytest.fixture(scope="module")
def pieces():
    tok = Tokenizer()
    ids = tok.encode(PROMPT)
    ours = model_mod.load()
    ref_model = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    with torch.no_grad():
        ref = ref_model(torch.tensor([ids])).logits[0].numpy()
    return tok, ids, ours, ref


def test_stateless_matches_reference(pieces):
    _, ids, ours, ref = pieces
    logits = ours.forward(ids)
    assert np.abs(logits - ref).max() < 1e-3
    assert (logits.argmax(-1) == ref.argmax(-1)).all()


def test_kv_cache_matches_reference(pieces):
    _, ids, ours, ref = pieces
    cache = KVCache(ours.cfg, max_len=len(ids))
    rows = [ours.forward([t], cache) for t in ids]
    logits = np.vstack(rows)
    assert np.abs(logits - ref).max() < 2e-3
    assert (logits.argmax(-1) == ref.argmax(-1)).all()


def test_int8_picks_the_same_next_token(pieces):
    _, ids, ours, ref = pieces
    q = ours.quantized()
    assert int(q.forward(ids)[-1].argmax()) == int(ref[-1].argmax())
