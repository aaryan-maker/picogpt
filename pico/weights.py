from __future__ import annotations

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "openai-community/gpt2"
FILENAME = "model.safetensors"


def download(repo: str = REPO, filename: str = FILENAME) -> str:
    return hf_hub_download(repo_id=repo, filename=filename)


def load(path: str | None = None) -> dict[str, np.ndarray]:
    if path is None:
        path = download()
    st = load_file(path)
    return {
        k: np.array(v, dtype=np.float32, copy=True, order="C")
        for k, v in st.items()
        if not k.endswith(".attn.bias")
    }


def summarize(st: dict[str, np.ndarray]) -> str:
    lines, total = [], 0
    for k, v in st.items():
        total += v.size
        lines.append(f"  {k:<28} {str(list(v.shape)):<16} {v.dtype}")
    lines.append(f"  {'TOTAL PARAMS':<28} {total:,}")
    return "\n".join(lines)
