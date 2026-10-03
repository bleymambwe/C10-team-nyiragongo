"""Write docs/03_VIDEO_TRANSCRIPT.md from the narration script and manifest.

Timecodes are computed with the same accounting the composition uses, so the
transcript cannot drift from the rendered video.

Run:  python paper-video/make_transcript.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FPS = 30
TRANSITION_FRAMES = 12

TITLES = {
    "title": "Opening",
    "problem": "The habit: one vector, two jobs",
    "two_geometries": "Two geometries, one residual stream",
    "certificate": "The steerability certificate",
    "regions": "Four regions of a residual stream",
    "theorem": "Theorem 2 - exact decoupling",
    "leakage": "Proposition 7 - finite-sample leakage",
    "lowrank": "Proposition 3 - effective rank",
    "gpt2": "Experiment E3 - toxicity in GPT-2",
    "notcircular": 'The "gradients obviously steer better" objection',
    "independence": "Experiment E6 - structural independence",
    "realdata": "Experiment E5 - real data, three models",
    "trained": "Experiment E2 - trained from scratch",
    "budget": "Experiment E4 - active intervention design",
    "failures": "What did not work",
    "limitations": "Limitations",
    "safety": "Consequence for monitoring",
    "next": "Next steps",
    "close": "The claim in one sentence",
}


def load_script():
    """Read SCRIPT out of generate_voiceover.py without importing kokoro."""
    src = open(os.path.join(HERE, "generate_voiceover.py"), encoding="utf-8").read()
    start = src.index("SCRIPT = [")
    end = src.index("]\n\n\ndef main")
    ns: dict = {}
    exec(src[start:end + 1], ns)
    return ns["SCRIPT"]


def main():
    script = load_script()
    man = json.load(open(os.path.join(HERE, "src", "narration.json")))

    frames = [round((s["durationSec"] + 0.55) * FPS) for s in man["scenes"]]
    starts, t = [], 0
    for i, f in enumerate(frames):
        starts.append(t)
        t += f - (TRANSITION_FRAMES if i < len(frames) - 1 else 0)

    def ts(fr):
        sec = fr / FPS
        return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

    out = [
        "# Video transcript - *Seeing Is Not Steering*",
        "",
        f"Runtime **{t / FPS / 60:.2f} min** - {len(script)} scenes - 1920x1080, 30 fps.",
        "Narration synthesised locally with Kokoro-82M (`af_heart`, speed "
        f"{man.get('speed', 0.98)}); no network, no API key.",
        "Rendered with Remotion. Every on-screen figure reads "
        "`paper-video/src/facts.json`, generated from `results/data/`.",
        "",
        "| # | Time | Scene |",
        "|---|---|---|",
    ]
    for i, (sid, _) in enumerate(script):
        out.append(f"| {i:02d} | `{ts(starts[i])}` | {TITLES.get(sid, sid)} |")
    out += ["", "---", ""]
    for i, (sid, text) in enumerate(script):
        out.append(f"## {i:02d} - {ts(starts[i])} - {TITLES.get(sid, sid)}")
        out.append("")
        out.append(text.strip())
        out.append("")

    path = os.path.join(ROOT, "docs", "03_VIDEO_TRANSCRIPT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {path}")
    print(f"  {len(script)} scenes, runtime {t / FPS / 60:.2f} min")


if __name__ == "__main__":
    main()
