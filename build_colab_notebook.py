"""Generate colab_probe_pipeline.ipynb from src/probe/*.py.

The notebook has to be self-contained -- uploading the repository to Colab is
one more thing to get wrong -- but duplicating the modules by hand would let the
notebook drift from the code the tests cover. So the modules are inlined
mechanically from their single source of truth, and regenerating is one command:

    python build_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src" / "probe"
OUT = ROOT / "colab_probe_pipeline.ipynb"

MODULES = ["corpus", "select", "submission", "finalize", "extract"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


def module_cell(name: str) -> dict:
    body = (SRC / f"{name}.py").read_text(encoding="utf-8")
    return code(f"%%writefile probe/{name}.py\n{body}")


CELLS = [
    md("""
# Latent Probe Challenge 17670 — extract, select, package

Runs the whole pipeline on one GPU session and hands back submission zips.

**Before running:** Runtime → Change runtime type → **T4 GPU**.
You need a Hugging Face token whose account has accepted the `google/gemma-2-2b`
licence.

The contract is fixed by the organizer and is not a tuning knob:
`google/gemma-2-2b`, hidden state **14**, **mean over all tokens**, truncation at
**64** tokens, 2304 features.

Expected wall clock on a T4: ~5 min setup, ~25-40 min extraction for ~140k rows,
~10 min selection, ~2 min packaging.
"""),
    code("""
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(gpu or "NO GPU DETECTED -- Runtime > Change runtime type > T4 GPU, then rerun")
"""),
    code("""
%pip -q install -U "transformers>=4.44" "datasets>=3.0" "huggingface_hub>=0.25" accelerate scikit-learn
"""),
    md("""
## Token

Preferred: add `HF_TOKEN` to Colab **Secrets** (the key icon in the left sidebar)
and enable it for this notebook. Otherwise it is prompted for and never written
to disk.
"""),
    code("""
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("token loaded from Colab secrets")
except Exception:
    import getpass
    os.environ["HF_TOKEN"] = getpass.getpass("HF token (needs gemma-2-2b access): ").strip()
    print("token loaded from prompt")
assert os.environ.get("HF_TOKEN"), "no token"
"""),
    md("""
## Optional: persist to Drive

Mount Drive so a disconnect does not cost the extraction. Skip this cell to keep
everything in the ephemeral session.
"""),
    code("""
WORK = "/content/probe_work"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/latent_probe_17670"
except Exception as exc:
    print("Drive not mounted, using ephemeral storage:", exc)
import os
os.makedirs(WORK, exist_ok=True)
print("work dir:", WORK)
"""),
    md("## Pipeline modules\n\nGenerated from `src/probe/` — edit there, not here."),
    code("""
import pathlib
pathlib.Path("probe").mkdir(exist_ok=True)
pathlib.Path("probe/__init__.py").write_text("")
print("package dir ready")
"""),
    *[module_cell(name) for name in MODULES],
    md("""
## 1. Build the corpus

`CAP_SCALE` trades corpus size against extraction time. 1.0 is roughly 140k rows
and 25-40 min on a T4. Drop to 0.25 for a fast end-to-end rehearsal.
"""),
    code("""
import importlib, numpy as np, json, os
import probe.corpus as corpus
importlib.reload(corpus)

CAP_SCALE = 1.0

texts, labels, sources = corpus.build_corpus(cap_scale=CAP_SCALE, positive_rate=0.5)
families = [corpus.SOURCES_BY_NAME[s].group for s in sources]
labels = np.asarray(labels, dtype=np.int64)

with open(os.path.join(WORK, "corpus.jsonl"), "w", encoding="utf-8") as fh:
    for t, y, s in zip(texts, labels, sources):
        fh.write(json.dumps({"text": t, "label": int(y), "source": s}) + "\\n")
print("rows:", len(texts), "| families:", sorted(set(families)))
"""),
    md("""
## 2. Extract embeddings

Resumable: rerunning after a disconnect continues from the journal rather than
starting over. The parity check must print a max error near zero before any rows
are written — that is what guarantees the fast path equals
`output_hidden_states=True` at index 14.
"""),
    code("""
import probe.extract as extract
importlib.reload(extract)

emb_path = os.path.join(WORK, f"gemma_l14_mean64_{len(texts)}.npy")
X = extract.extract(texts, emb_path, token=os.environ["HF_TOKEN"], batch_size=64)
X = np.asarray(X)
print(X.shape, X.dtype, "finite:", bool(np.isfinite(X).all()))
"""),
    code("""
bundle_path = os.path.join(WORK, "bundle_l14.npz")
extract.save_bundle(bundle_path, X, labels, sources, families=families)
"""),
    md("""
## 3. Leave-one-source-out selection

Each fold trains on every corpus but one and scores the held-out corpus after
resampling it to the hidden set's known 70.588% positive rate, so these numbers
are on the same footing as a leaderboard score.

Recipes are ranked by **worst** fold, not mean: the hidden corpus is one unknown
draw, so the recipe that survives its weakest fold is worth more than the one
with the best average.
"""),
    code("""
import probe.select as select
importlib.reload(select)

groups = dict(zip(sources, families))
summary = select.evaluate_recipes(X, labels, np.asarray(sources), groups,
                                  max_train_rows=40_000, verbose=True)
with open(os.path.join(WORK, "loso_results.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print()
print(select.report(summary, top=25))
"""),
    md("""
### Cross-source diagnostic

A tiebreaker, not a selection criterion. It scores folds whose positives and
negatives come from two corpora the probe has never seen, because the organizer
reaching 94.9% from ~5,440 training rows suggests their toxic and safe halves
differ by provenance, not only by meaning. A recipe that holds up here is the
one more likely to survive that.
"""),
    code("""
cross = select.evaluate_cross_source(X, labels, np.asarray(sources), groups,
                                     max_train_rows=40_000, verbose=True)
with open(os.path.join(WORK, "cross_source_results.json"), "w") as fh:
    json.dump(cross, fh, indent=1)
print()
print("pairs:", ", ".join(cross["pairs"]))
for row in cross["results"][:15]:
    print(f"  {row['key']:<32} worst={row['worst_accuracy']:.4f} mean={row['mean_accuracy']:.4f}")
"""),
    md("""
## 4. Package submissions

`best_single` and `best_ensemble` are the models to score. The `source_*` zips
are the leaderboard experiment from G4: one probe per training corpus, so a dev
submission tells us which public corpus the hidden data resembles.
"""),
    code("""
import probe.finalize as finalize
importlib.reload(finalize)

manifest = finalize.build_candidates(X, labels, np.asarray(sources), groups, summary,
                                     os.path.join(WORK, "submissions"),
                                     cross=cross, ensemble_size=3, per_source=True)
print(finalize.report(manifest))
"""),
    md("## 5. Download\n\nThe zips and the results JSON are small; the embedding bundle is not."),
    code("""
import shutil
shutil.make_archive(os.path.join(WORK, "probe_outputs"), "zip",
                    os.path.join(WORK, "submissions"))
for name in ("loso_results.json", "cross_source_results.json"):
    src_path = os.path.join(WORK, name)
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(WORK, "submissions", name))
shutil.make_archive(os.path.join(WORK, "probe_outputs"), "zip",
                    os.path.join(WORK, "submissions"))
print("bundle size (MB):", round(os.path.getsize(bundle_path) / 1e6, 1))

try:
    from google.colab import files
    files.download(os.path.join(WORK, "probe_outputs.zip"))
except Exception as exc:
    print("download from the Files pane instead:", exc)
"""),
    code("""
# Optional, and larger: the embeddings themselves, for local experimentation.
try:
    from google.colab import files
    files.download(bundle_path)
except Exception as exc:
    print(exc)
"""),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
