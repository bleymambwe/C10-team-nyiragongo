"""Run the probe pipeline on Kaggle GPU kernels, headless.

Two kernels, because extraction is the only expensive step:

  `extract`  builds the corpus, runs Gemma-2-2b to layer 14 on a T4 and leaves
             the embedding bundle in its own output. ~30-40 min. Run it once.
  `select`   attaches the extract kernel's output as an input, so the
             embeddings are mounted rather than recomputed, then runs
             selection, the cross-source diagnostic, ensembling and packaging.
             ~5 min, and it is the one to re-run while iterating on strategy.

A Kaggle kernel takes exactly one code file, so `src/probe/*.py` is inlined
into the generated script the same way `build_colab_notebook.py` does it --
edit the modules, never the generated script.

The Hugging Face token is read inside the kernel from Kaggle Secrets
(`UserSecretsClient`), so it is never written into anything pushed from here.

    python run_kaggle.py extract --cap-scale 1.0
    python run_kaggle.py status extract
    python run_kaggle.py select --strategy default
    python run_kaggle.py pull select --out artifacts/kaggle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent
SRC = REPO / "src" / "probe"
MODULES = ["corpus", "select", "submission", "finalize", "extract"]

USER = "bleymambwe"
EXTRACT_SLUG = "latent-probe-extract"
SELECT_SLUG = "latent-probe-select"
WORK = REPO / "results" / "kaggle"


def _inline_modules() -> str:
    """Emit code that recreates the probe package inside the kernel.

    The modules are base64-encoded rather than pasted into a string literal:
    they contain their own triple-quoted docstrings, backslashes and quotes, and
    every literal form eventually collides with one of them. Base64 cannot.
    """
    import base64

    lines = [
        "import base64, pathlib, sys",
        # Kaggle runs the script from a directory that is not on sys.path, so a
        # package written to the CWD is invisible to `import probe`. Write it
        # somewhere explicit and put that on the path.
        "_PKG_ROOT = pathlib.Path('/tmp/probe_pkg')",
        "(_PKG_ROOT / 'probe').mkdir(parents=True, exist_ok=True)",
        "(_PKG_ROOT / 'probe' / '__init__.py').write_text('')",
        "sys.path.insert(0, str(_PKG_ROOT))",
        "_MODULES = {",
    ]
    for name in MODULES:
        blob = base64.b64encode((SRC / f"{name}.py").read_bytes()).decode("ascii")
        lines.append(f"    {name!r}: {blob!r},")
    lines += [
        "}",
        "for _name, _blob in _MODULES.items():",
        "    (_PKG_ROOT / 'probe' / f'{_name}.py').write_bytes(base64.b64decode(_blob))",
        "print('probe package written to', _PKG_ROOT, sorted(_MODULES))",
    ]
    return "\n".join(lines)


EXTRACT_BODY = '''
import json, os, subprocess, sys, time

# MUST run before torch is imported anywhere.
#
# Kaggle's API only exposes an `enable_gpu` boolean, with no way to request a
# card, and it consistently allocates a Tesla P100. P100 is sm_60; the stock
# build here is torch 2.10.0+cu128, whose arch list is sm_70..sm_120. The
# mismatch is not a warning -- every CUDA op raises
# "no kernel image is available for execution on the device".
#
# Measured fix: torch 2.4.1+cu121 ships sm_50..sm_90 and runs on this card.
# transformers is pinned alongside it because the stock transformers expects a
# much newer torch; 4.44.2 is the oldest release with Gemma-2 support.
_gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
print("allocated GPU:", _gpu)
if "P100" in _gpu or "K80" in _gpu:
    print("pre-Volta card: installing a torch build that targets sm_60...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.1", "--index-url",
                    "https://download.pytorch.org/whl/cu121"], check=True, timeout=2400)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.44.2"], check=True, timeout=1200)
    import importlib
    importlib.invalidate_caches()

import numpy as np

# Weights come from Kaggle's own mounted copy of Gemma-2, so no credential is
# needed and none is ever written into this file. A Hugging Face token is used
# only if one happens to be attached as a Kaggle secret AND the mount is absent;
# pushing a new kernel version resets secret attachments, so the mount is the
# path that actually survives an unattended loop.
import glob
# The mount nests a version directory under the variation, so the model root is
# wherever config.json actually is -- globbing the variation returned ['2'].
MODEL_PATH = None
_configs = [c for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
            if "gemma" in c.lower()]
if _configs:
    MODEL_PATH = os.path.dirname(sorted(_configs, key=len)[0])
else:
    print("kaggle/input tree:", glob.glob("/kaggle/input/*"))

if MODEL_PATH:
    print("using Kaggle-mounted weights:", MODEL_PATH)
    print("  contents:", sorted(os.listdir(MODEL_PATH))[:12])
else:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        MODEL_PATH = "google/gemma-2-2b"
        print("no mount found; falling back to Hugging Face with a Kaggle secret")
    except Exception as exc:
        raise SystemExit(
            f"no Gemma weights: mount missing and no HF_TOKEN secret ({type(exc).__name__}). "
            "Attach the model google/gemma-2/transformers/gemma-2-2b to this kernel.")

# The contract names a specific model. Verify the mounted weights are that
# model before spending an hour on them: a different Gemma variant would
# produce embeddings that look fine and score badly for no visible reason.
import json as _json
_cfg_path = os.path.join(str(MODEL_PATH), "config.json")
if os.path.exists(_cfg_path):
    _cfg = _json.load(open(_cfg_path))
    print("config:", {k: _cfg.get(k) for k in
                      ("model_type", "hidden_size", "num_hidden_layers",
                       "vocab_size", "intermediate_size")})
    assert _cfg.get("model_type") == "gemma2", f"expected gemma2, got {_cfg.get('model_type')}"
    assert _cfg.get("hidden_size") == 2304, f"hidden_size {_cfg.get('hidden_size')} != 2304"
    assert _cfg.get("num_hidden_layers") == 26, f"layers {_cfg.get('num_hidden_layers')} != 26"
    print("contract check: gemma2, 2304 hidden, 26 layers -- matches google/gemma-2-2b")
    print("weight files:", sorted(f for f in os.listdir(MODEL_PATH)
                                  if f.endswith((".safetensors", ".bin")))[:8])
else:
    raise SystemExit(f"no config.json under {MODEL_PATH}; cannot verify the contract")

import torch
print("torch", torch.__version__, "| arch list", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    raise SystemExit("no GPU: enable the GPU accelerator on this kernel")
_cap = torch.cuda.get_device_capability(0)
print("gpu", torch.cuda.get_device_name(0), _cap)
# Fail here, in one second, rather than after the corpus build and a partial
# extraction: a card outside the build's arch list raises on every CUDA op.
_arch = f"sm_{_cap[0]}{_cap[1]}"
if _arch not in torch.cuda.get_arch_list():
    raise SystemExit(f"{_arch} is not in this torch build's arch list "
                     f"{torch.cuda.get_arch_list()} -- every CUDA op would fail")
torch.zeros(4, device="cuda").sum().item()  # proves a kernel actually launches
print("CUDA smoke test passed")

import probe.corpus as corpus
import probe.extract as extract

CAP_SCALE = __CAP_SCALE__
OUT = "/kaggle/working"

t0 = time.perf_counter()
texts, labels, sources = corpus.build_corpus(cap_scale=CAP_SCALE, positive_rate=0.5)
families = [corpus.SOURCES_BY_NAME[s].group for s in sources]
labels = np.asarray(labels, dtype=np.int64)
print(f"corpus: {len(texts)} rows, {len(set(families))} families, "
      f"{time.perf_counter() - t0:.0f}s")

with open(os.path.join(OUT, "corpus.jsonl"), "w", encoding="utf-8") as fh:
    for t, y, s in zip(texts, labels, sources):
        fh.write(json.dumps({"text": t, "label": int(y), "source": s}) + "\\n")

# Scratch, not output: the raw float32 memmap is large and is not worth
# shipping when the float16 bundle carries the same signal.
t0 = time.perf_counter()
X = extract.extract(texts, "/kaggle/temp/gemma_l14.npy",
                    model_id=MODEL_PATH, token=os.environ.get("HF_TOKEN"),
                    batch_size=64)
X = np.asarray(X)
dt = time.perf_counter() - t0
print(f"extracted {X.shape} in {dt/60:.1f} min ({len(texts)/max(dt,1):.1f} rows/s)")
assert X.shape[1] == 2304, f"expected 2304 features, got {X.shape[1]}"
assert np.isfinite(X).all(), "non-finite embeddings"

extract.save_bundle(os.path.join(OUT, "bundle_l14.npz"), X, labels, sources,
                    families=families)
json.dump({"rows": int(len(texts)), "families": sorted(set(families)),
           "cap_scale": CAP_SCALE, "seconds": dt,
           "contract": {"model": str(MODEL_PATH), "layer": extract.HIDDEN_STATE_INDEX,
                        "max_length": extract.MAX_LENGTH, "pooling": "masked_mean"}},
          open(os.path.join(OUT, "extract_manifest.json"), "w"), indent=1)
print("EXTRACT_OK")
'''


SELECT_BODY = '''
import json, os, glob
import numpy as np

import probe.select as select
import probe.finalize as finalize

OUT = "/kaggle/working"
hits = glob.glob("/kaggle/input/**/bundle_l14.npz", recursive=True)
if not hits:
    raise SystemExit("bundle_l14.npz not found in /kaggle/input -- "
                     "is the extract kernel attached as a source?")
print("bundle:", hits[0])
data = np.load(hits[0], allow_pickle=True)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
sources = data["sources"]
families = data["families"] if "families" in data else sources
groups = dict(zip(sources.tolist(), families.tolist()))
print("loaded", X.shape, "families:", sorted(set(families.tolist())))

# Source hygiene before anything is fitted. The real extraction produced
# hu_berlin_toxicity with 3,828 rows and zero positives (its toxic half
# duplicates Civil Comments and was de-duplicated away), and per-source positive
# rates from 0.087 to 0.781 despite the builder targeting 0.50.
print("== source hygiene ==")
X, y, sources, hygiene = select.clean_sources(X, y, sources, groups,
                                              min_per_class=200, rebalance=True)
json.dump(hygiene, open(os.path.join(OUT, "source_hygiene.json"), "w"), indent=1)
groups = {s: groups[s] for s in set(sources.tolist())}

summary = select.evaluate_recipes(X, y, sources, groups, max_train_rows=40_000,
                                  verbose=True)
json.dump(summary, open(os.path.join(OUT, "loso_results.json"), "w"), indent=1)
print()
print(select.report(summary, top=25))

cross = select.evaluate_cross_source(X, y, sources, groups, max_train_rows=40_000,
                                     verbose=True)
json.dump(cross, open(os.path.join(OUT, "cross_source_results.json"), "w"), indent=1)
print()
print("cross-source pairs:", ", ".join(cross["pairs"]))
for row in cross["results"][:15]:
    print(f"  {row['key']:<32} worst={row['worst_accuracy']:.4f} "
          f"mean={row['mean_accuracy']:.4f}")

manifest = finalize.build_candidates(X, y, sources, groups, summary,
                                     os.path.join(OUT, "submissions"),
                                     cross=cross, ensemble_size=3, per_source=True)
print()
print(finalize.report(manifest))
print("SELECT_OK")
'''


GEMMA_KAGGLE_REF = "google/gemma-2/transformers/gemma-2-2b/2"


def _metadata(slug: str, title: str, gpu: bool, sources: list[str],
              models: list[str] | None = None) -> dict:
    return {
        "id": f"{USER}/{slug}",
        "title": title,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": sources,
        "model_sources": models or [],
    }


def build_kernel(kind: str, cap_scale: float = 1.0) -> Path:
    folder = WORK / kind
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "extract":
        body = EXTRACT_BODY.replace("__CAP_SCALE__", repr(float(cap_scale)))
        meta = _metadata(EXTRACT_SLUG, "latent-probe-extract", gpu=True, sources=[],
                         models=[GEMMA_KAGGLE_REF])
    else:
        body = SELECT_BODY
        meta = _metadata(SELECT_SLUG, "latent-probe-select", gpu=False,
                         sources=[f"{USER}/{EXTRACT_SLUG}"])
    (folder / "script.py").write_text(_inline_modules() + "\n\n" + body, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    size = (folder / "script.py").stat().st_size
    print(f"built {kind} kernel: {folder} ({size/1024:.0f} KB script)")
    return folder


def _api():
    import kaggle

    api = kaggle.KaggleApi()
    api.authenticate()
    return api


def push(kind: str, cap_scale: float = 1.0) -> str:
    folder = build_kernel(kind, cap_scale)
    api = _api()
    # sm_60 (P100) is outside this torch build's supported range (sm_70-sm_120),
    # so a P100 session warns and then fails on the first CUDA op. T4 is sm_75.
    try:
        result = api.kernels_push(str(folder.resolve()), acc="nvidiaTeslaT4")
    except TypeError:
        result = api.kernels_push(str(folder.resolve()))
    ref = getattr(result, "ref", None)
    error = getattr(result, "error", None)
    print(f"pushed: {ref}  version={getattr(result, 'versionNumber', None)}")
    if error:
        print(f"error: {error}")
    url = getattr(result, "url", None)
    if url:
        print(f"url: {url}")
    return ref or ""


def status(kind: str, wait: bool = False, timeout_s: int = 5400) -> str:
    slug = EXTRACT_SLUG if kind == "extract" else SELECT_SLUG
    ref = f"{USER}/{slug}"
    api = _api()
    started, last = time.time(), None
    while True:
        try:
            st = api.kernels_status(ref)
            state = str(getattr(st, "status", st))
            message = getattr(st, "failureMessage", "") or ""
        except Exception as exc:
            state, message = f"ERR {type(exc).__name__}", str(exc)[:150]
        if state != last:
            print(f"[{int(time.time()-started):>5}s] {state} {message}".rstrip(), flush=True)
            last = state
        done = any(k in state.lower() for k in ("complete", "error", "cancel"))
        if done or not wait or time.time() - started > timeout_s:
            return state
        time.sleep(20)


def pull(kind: str, out: Path) -> None:
    slug = EXTRACT_SLUG if kind == "extract" else SELECT_SLUG
    out.mkdir(parents=True, exist_ok=True)
    api = _api()
    api.kernels_output(f"{USER}/{slug}", path=str(out.resolve()))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    print(f"pulled {len(files)} files into {out}")
    for p in files:
        print(f"   {p.relative_to(out)}  {p.stat().st_size/1024:.0f} KB")


def log(kind: str) -> None:
    """Print the kernel's stdout, which is where the run's evidence lives."""
    slug = EXTRACT_SLUG if kind == "extract" else SELECT_SLUG
    tmp = WORK / f"_log_{kind}"
    tmp.mkdir(parents=True, exist_ok=True)
    _api().kernels_output(f"{USER}/{slug}", path=str(tmp.resolve()))
    for path in sorted(tmp.glob("*.log")):
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(path.read_text(encoding="utf-8")[-8000:])
            continue
        for entry in entries:
            text = entry.get("data", "") if isinstance(entry, dict) else str(entry)
            if text.strip():
                print(text.rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="push the GPU extraction kernel")
    p.add_argument("--cap-scale", type=float, default=1.0)
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("select", help="push the selection kernel")
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("status", help="poll a kernel")
    p.add_argument("kind", choices=["extract", "select"])
    p.add_argument("--wait", action="store_true")

    p = sub.add_parser("pull", help="download a kernel's output")
    p.add_argument("kind", choices=["extract", "select"])
    p.add_argument("--out", default="artifacts/kaggle")

    p = sub.add_parser("log", help="print a kernel's stdout")
    p.add_argument("kind", choices=["extract", "select"])

    p = sub.add_parser("build", help="generate a kernel folder without pushing")
    p.add_argument("kind", choices=["extract", "select"])
    p.add_argument("--cap-scale", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "build":
        build_kernel(args.kind, getattr(args, "cap_scale", 1.0))
    elif args.command in ("extract", "select"):
        push(args.command, getattr(args, "cap_scale", 1.0))
        if args.wait:
            status(args.command, wait=True)
    elif args.command == "status":
        status(args.kind, wait=args.wait)
    elif args.command == "pull":
        pull(args.kind, Path(args.out))
    elif args.command == "log":
        log(args.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
