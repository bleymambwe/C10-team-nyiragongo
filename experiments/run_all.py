"""Run the full experiment suite in dependency order and rebuild the dashboard.

Each experiment writes results/data/<name>.json and is independently runnable.
Failures are reported and do not stop the suite -- the dashboard renders
whatever completed.

Run:  python experiments/run_all.py            (everything)
      python experiments/run_all.py e1 e4      (a subset)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SUITE = [
    ("e1", "e1_exact_decoupling.py", "Exact decoupling (Thm 2, Props 3-5, 7)", "~2 min, CPU"),
    ("e2", "e2_trained_toy.py", "Trained toy transformers + confound dial", "~45 min, CPU"),
    ("e3", "e3_gpt2_toxicity.py", "GPT-2 toxicity: layerwise certificate vs steering", "~90 min, CPU"),
    ("e5", "e5_real_data_multimodel.py", "RealToxicityPrompts, 3 models, all baselines", "~60 min, CPU"),
    ("e6", "e6_independence.py", "Read/write geometric independence", "~15 min, CPU"),
    ("e4", "e4_active_design.py", "Active intervention design", "~25 min, CPU"),
]


def main():
    want = [a.lower() for a in sys.argv[1:]]
    todo = [s for s in SUITE if not want or s[0] in want]
    print(f"running {len(todo)} experiment(s)\n")
    results = []
    for key, script, desc, cost in todo:
        print("=" * 78)
        print(f"{key.upper()}  {desc}   [{cost}]")
        print("=" * 78)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, script)],
                           cwd=ROOT)
        dt = time.time() - t0
        results.append((key, r.returncode == 0, dt))
        print(f"\n{key.upper()} {'OK' if r.returncode == 0 else 'FAILED'} in {dt/60:.1f} min\n")

    print("=" * 78)
    for key, ok, dt in results:
        print(f"  {key.upper():<4} {'ok' if ok else 'FAILED':<7} {dt/60:6.1f} min")
    print("=" * 78)

    print("\nrebuilding dashboard ...")
    subprocess.run([sys.executable, os.path.join(ROOT, "dashboard", "build.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
