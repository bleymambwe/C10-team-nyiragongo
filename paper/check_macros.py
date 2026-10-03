"""Verify every measurement macro used in the preprint is defined by the data.

Guards the reproducibility claim: if a macro appears in preprint.tex but not in
numbers.tex, LaTeX silently renders nothing useful and a number quietly vanishes
from the paper. This turns that into a hard failure.

Run:  python paper/check_macros.py
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

tex = open(os.path.join(HERE, "preprint.tex"), encoding="utf-8").read()
nums = open(os.path.join(HERE, "numbers.tex"), encoding="utf-8").read()

defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", nums))
local = set(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", tex, flags=re.M))

# macros the body uses that look like generated measurements
used = set(re.findall(r"\\((?:Eone|Ethree|Efour|Efive|Esix|Pseven|Pthree|Pfive)[A-Za-z]*)", tex))

missing = sorted(u for u in used if u not in defined)
unused = sorted(d for d in defined if d not in used)

print(f"defined in numbers.tex : {len(defined)}")
print(f"used in preprint.tex   : {len(used)}")
print(f"local \\newcommand      : {len(local)}")

if missing:
    print("\nMISSING (used but not defined):")
    for m in missing:
        print("  \\" + m)
else:
    print("\nno missing macros")

if unused:
    print(f"\ndefined but unused ({len(unused)}): " + ", ".join("\\" + u for u in unused))

sys.exit(1 if missing else 0)
