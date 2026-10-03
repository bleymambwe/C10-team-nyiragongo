"""Private, isolated Kaggle experiment; preserves the established kernels."""
import argparse
import base64
import json
from pathlib import Path

import run_kaggle as runner

SLUG = "latent-probe-corpus-transfer"
FOLDER = Path("results/kaggle/corpus-transfer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["build", "push", "status", "pull"])
    p.add_argument("--full", action="store_true")
    a = p.parse_args()
    ref = f"{runner.USER}/{SLUG}"
    if a.action in ("build", "push"):
        FOLDER.mkdir(parents=True, exist_ok=True)
        body = Path("experiments/corpus_transfer.py").read_text(encoding="utf-8")
        body = body.replace("from __future__ import annotations\n", "")
        body = body[:body.index('if __name__ == "__main__":')]
        opts = ', max_train=200000, source_cap=200000, within=True' if a.full else ''
        launch = "\nimport glob\nrun(Path(glob.glob('/kaggle/input/**/bundle_l14.npz', recursive=True)[0]), Path('/kaggle/working')" + opts + ")\n"
        prelude = "import os\nos.environ['OPENBLAS_NUM_THREADS']='4'\nos.environ['OMP_NUM_THREADS']='4'\n"
        routing = Path('src/probe/routing.py').read_bytes()
        inline = runner._inline_modules() + "\n(_PKG_ROOT / 'probe' / 'routing.py').write_bytes(base64.b64decode(" + repr(base64.b64encode(routing).decode()) + "))\n"
        (FOLDER / "script.py").write_text(prelude + inline + "\n" + body + launch, encoding="utf-8")
        meta = runner._metadata(SLUG, SLUG, False, [f"{runner.USER}/{runner.EXTRACT_SLUG}"])
        (FOLDER / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(f"built {FOLDER}")
    if a.action == "build":
        return
    api = runner._api()
    if a.action == "push":
        result = api.kernels_push(str(FOLDER.resolve()))
        print({k: getattr(result, k, None) for k in ("ref", "url", "versionNumber", "error")})
    elif a.action == "status":
        result = api.kernels_status(ref)
        print({"status": str(result.status), "failure": getattr(result, "failureMessage", None)})
    else:
        target = Path("artifacts/kaggle/corpus-transfer-full" if a.full else "artifacts/kaggle/corpus-transfer")
        target.mkdir(parents=True, exist_ok=True)
        api.kernels_output(ref, path=str(target.resolve()))
        print([str(x) for x in target.iterdir()])


if __name__ == "__main__":
    main()
