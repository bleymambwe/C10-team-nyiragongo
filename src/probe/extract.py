"""Gemma-2-2b layer-14 masked-mean extraction under the organizer's contract.

Contract, quoted from the CodaBench 17670 Task Description page:

  * `google/gemma-2-2b`, loaded with `output_hidden_states=True`
  * hidden-state **layer 14** of 27 (indices 0-26)
  * the **mean across all tokens**, not the last token
  * tokenized with **truncation at 64 tokens**

`hidden_states[14]` is the input to decoder block 14, so a forward pre-hook on
`model.layers[14]` sees exactly that tensor.  Raising out of the hook skips
blocks 14-25 and roughly halves the compute.  `verify_contract` checks that
shortcut against the plain `output_hidden_states=True` path before it is used,
so the saving can never quietly change the numbers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

MODEL_ID = "google/gemma-2-2b"
HIDDEN_STATE_INDEX = 14
MAX_LENGTH = 64
HIDDEN_SIZE = 2304


def load_hf_token(env_path: Path | str = "BMToken.env") -> str | None:
    """Read the token without printing or copying it anywhere."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    path = Path(env_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*['\"]?([^'\"\s#]+)", line)
        if match:
            return match.group(1)
    return None


class _Captured(RuntimeError):
    """Control-flow signal, not an error: block 14's input has been read."""


def load_model(token: str | None = None, device: str | None = None,
               model_id: str = MODEL_ID):
    """Load a decoder and its tokenizer for hidden-state extraction.

    `model_id` exists so the same code path can be rehearsed end to end with a
    small cached model on CPU before it is trusted with the real contract; it
    defaults to the model the competition specifies.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        # Ampere and later have bfloat16, whose range matches float32, so it is
        # safe. Before Ampere the only half format is float16, and Gemma-2
        # carries residual-stream magnitudes and soft-capping that overflow its
        # 65504 ceiling -- the failure is silent and would poison every
        # downstream number. On those cards float32 is the correct trade: about
        # 4x slower, but this runs unattended and correctness is the whole
        # point of the exercise. Kaggle's P100 and T4 both land here.
        dtype = torch.bfloat16 if major >= 8 else torch.float32
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModel.from_pretrained(model_id, token=token, dtype=dtype,
                                          low_cpu_mem_usage=True)
    except TypeError:  # transformers 4.x spelled it torch_dtype
        model = AutoModel.from_pretrained(model_id, token=token, torch_dtype=dtype,
                                          low_cpu_mem_usage=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print({"model": model_id, "device": str(device), "dtype": str(dtype)})
    return model, tokenizer, device


def _blocks(model):
    """The decoder block list, whatever the architecture calls it."""
    for attribute in ("layers", "h", "blocks"):
        found = getattr(model, attribute, None)
        if found is not None:
            return found
    raise AttributeError(f"no decoder block list on {type(model).__name__}")


def _set_blocks(model, value):
    for attribute in ("layers", "h", "blocks"):
        if getattr(model, attribute, None) is not None:
            setattr(model, attribute, value)
            return
    raise AttributeError("no decoder block list to replace")


def _masked_mean(hidden, mask):
    mask = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def verify_contract(model, tokenizer, device, sample_texts: list[str],
                    tolerance: float = 1e-4, layer: int = HIDDEN_STATE_INDEX,
                    max_length: int = MAX_LENGTH) -> float:
    """Assert the early-exit hook reproduces `hidden_states[layer]` exactly."""
    import torch

    batch = tokenizer(sample_texts, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
    with torch.inference_mode():
        full = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    expected = _masked_mean(full.hidden_states[layer],
                            batch["attention_mask"]).float().cpu()
    del full

    capture: dict = {}

    def hook(_module, args):
        capture["value"] = _masked_mean(args[0], batch["attention_mask"]).float().cpu()
        raise _Captured

    handle = _blocks(model)[layer].register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            try:
                model(**batch, use_cache=False, return_dict=True)
            except _Captured:
                pass
    finally:
        handle.remove()

    error = float((capture["value"] - expected).abs().max())
    if error > tolerance:
        raise RuntimeError(f"early-exit parity check failed: max error {error}")
    print(f"layer-{layer} early-exit parity: max error {error:.3g}")
    return error


def extract(texts: list[str], output_path: Path | str, *, model=None, tokenizer=None,
            device=None, token: str | None = None, batch_size: int = 32,
            sort_by_length: bool = True, log_every: int = 20,
            model_id: str = MODEL_ID, layer: int = HIDDEN_STATE_INDEX,
            max_length: int = MAX_LENGTH) -> np.ndarray:
    """Extract embeddings to a resumable `.npy` memmap.

    Interrupting costs at most one batch: progress is journalled next to the
    output and a rerun picks up where it stopped.
    """
    import torch

    output_path = Path(output_path)
    if model is None:
        model, tokenizer, device = load_model(token=token, model_id=model_id)
    verify_contract(model, tokenizer, device, texts[: min(4, len(texts))],
                    layer=layer, max_length=max_length)

    width = int(model.config.hidden_size)

    # Blocks after the capture point can never execute; dropping them frees
    # roughly 40% of the weights and, on a small GPU, the headroom matters.
    _set_blocks(model, torch.nn.ModuleList(list(_blocks(model)[: layer + 1])))

    n = len(texts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.json")

    if output_path.exists():
        embeddings = np.lib.format.open_memmap(output_path, mode="r+")
        if embeddings.shape != (n, width):
            raise ValueError(f"cache has shape {embeddings.shape}, expected {(n, width)}")
        done = int(json.loads(progress_path.read_text())["done"]) if progress_path.exists() else 0
    else:
        embeddings = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
                                               shape=(n, width))
        done = 0

    # Batching similar lengths together cuts padding waste substantially; the
    # original row order is restored by writing through `order`.
    order = (np.argsort([len(t) for t in texts], kind="mergesort") if sort_by_length
             else np.arange(n))

    capture: dict = {}
    active_mask: dict = {}

    def hook(_module, args):
        capture["value"] = _masked_mean(args[0], active_mask["value"]).float().cpu()
        raise _Captured

    handle = _blocks(model)[layer].register_forward_pre_hook(hook)
    resume_from = done
    started = __import__("time").perf_counter()
    try:
        for start in range(done, n, batch_size):
            chunk = order[start:start + batch_size]
            batch = tokenizer([texts[i] for i in chunk], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
            active_mask["value"] = batch["attention_mask"]
            with torch.inference_mode():
                try:
                    model(**batch, use_cache=False, return_dict=True)
                except _Captured:
                    pass
            embeddings[chunk] = capture["value"].numpy()
            done = start + len(chunk)
            progress_path.write_text(json.dumps({"done": done}))
            if (start // batch_size) % log_every == 0:
                elapsed = __import__("time").perf_counter() - started
                rate = (done - resume_from) / max(elapsed, 1e-6)
                remaining = (n - done) / rate if rate > 0 else float("nan")
                print(f"  {done}/{n}  {done / n:6.1%}  {rate:7.1f} rows/s  "
                      f"eta {remaining / 60:5.1f} min", flush=True)
    finally:
        handle.remove()

    embeddings.flush()
    if not np.isfinite(embeddings).all():
        raise ValueError("extraction produced non-finite values")
    print(f"extracted {n} rows -> {output_path}")
    return embeddings


def save_bundle(path: Path | str, X: np.ndarray, y, sources, families=None,
                texts=None, dtype=np.float16) -> Path:
    """Write a compact bundle for the selection harness.

    float16 halves the download from a remote runtime and costs nothing: the
    embeddings carry roughly three significant digits of signal and every probe
    here standardises before fitting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": np.asarray(X, dtype=dtype),
        "y": np.asarray(y, dtype=np.int8),
        "sources": np.asarray(sources, dtype=object),
    }
    if families is not None:
        payload["families"] = np.asarray(families, dtype=object)
    if texts is not None:
        payload["texts"] = np.asarray(texts, dtype=object)
    np.savez_compressed(path, **payload)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path
