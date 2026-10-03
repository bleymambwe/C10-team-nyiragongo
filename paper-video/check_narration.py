"""Check that spoken numbers in the narration match src/facts.json.

The narration is the one place in this project where measurements are written
out by hand -- Kokoro needs words, not figures. That makes it the one place
voice and visuals can silently disagree. This asserts the key claims match.

Run:  python paper-video/check_narration.py
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
facts = json.load(open(os.path.join(HERE, "src", "facts.json")))

src = open(os.path.join(HERE, "generate_voiceover.py"), encoding="utf-8").read()
start = src.index("SCRIPT = [")
end = src.index("]\n\n\ndef main")
ns: dict = {}
exec(src[start:end + 1], ns)
SCRIPT = dict(ns["SCRIPT"])

e2, e3, e4, e6 = facts["e2"], facts["e3"], facts["e4"], facts["e6"]

# (scene, phrase that must appear, the value it asserts, actual value, tolerance)
CHECKS = [
    ("gpt2", "six percent", 6.0, 100 * e3["eff_probe"], 0.5),
    ("gpt2", "ninety nine point seven percent", 99.7, 100 * e3["eff_dark"], 0.15),
    ("gpt2", "zero point zero five six", 0.056, e3["mean_rho"], 0.001),
    ("lowrank", "zero point one eight percent", 0.18, 100 * e3["mean_reff"], 0.01),
    ("independence", "zero point zero three one", 0.031, e6["m"], 0.001),
    ("independence", "zero point zero three two", 0.032, e6["u"], 0.001),
    ("notcircular", "zero point nine two", 0.92, e3["mean_sp_s"], 0.01),
    ("budget", "one point three measurements", 1.3, e4["cert_mean"], 0.05),
    ("budget", "twenty nine", 29.0, e4["rand_mean"], 0.5),
    ("budget", "twenty two fold", 22.0, e4["speedup"], 0.6),
    ("trained", "zero point one five", 0.15, e2["rho_by_level"][0], 0.005),
    ("trained", "twelve percent", 12.0, e2["eff_by_level"][0], 0.3),
    ("trained", "twelve transformers", 12, e2["n_runs"], 0),
    ("failures", "four thousandths", 0.004, abs(e2["rho_delta"]), 0.0006),
    ("failures", "one fifth", 0.2, e2["delta_in_sd"], 0.06),
]

fails, missing = [], []
for scene, phrase, claimed, actual, tol in CHECKS:
    text = SCRIPT.get(scene, "")
    if phrase not in text:
        missing.append((scene, phrase))
        continue
    if abs(claimed - actual) > tol:
        fails.append((scene, phrase, claimed, actual))

# every scene id must have a component registered
video = open(os.path.join(HERE, "src", "Video.tsx"), encoding="utf-8").read()
registered = set(re.findall(r"^\s+(\w+): S\d+\w+,", video, flags=re.M))
unregistered = [s for s in SCRIPT if s not in registered]

print(f"checked {len(CHECKS)} spoken figures across {len(SCRIPT)} scenes")
if missing:
    print("\nPHRASE NOT FOUND (narration reworded? update this check):")
    for s, p in missing:
        print(f"  [{s}] expected to contain: '{p}'")
if fails:
    print("\nMISMATCH (voice disagrees with facts.json):")
    for s, p, c, a in fails:
        print(f"  [{s}] says '{p}' ({c}) but data says {a:.4f}")
if unregistered:
    print("\nSCENE WITHOUT A COMPONENT:", unregistered)
if not (missing or fails or unregistered):
    print("all spoken figures agree with the data; every scene is registered")

sys.exit(1 if (missing or fails or unregistered) else 0)
