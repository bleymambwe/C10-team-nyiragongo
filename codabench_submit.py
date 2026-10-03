"""Drive a real Chrome to submit to CodaBench 17670, and track the budget.

Why a browser and not the API: CodaBench's submission flow uploads to object
storage with a presigned URL and then creates the submission record. The public
REST API exposes reads without authentication but not that flow, and the
competition page itself is a JavaScript shell -- there is no form to POST. So
the reliable path is the one a person would take, automated.

Credentials are never handled here. `login` opens a headed browser against a
dedicated profile directory and waits for you to sign in yourself; the session
cookie then persists in that profile for later runs. Nothing reads or stores a
password, and the profile lives outside the repository.

Two guards, because submissions are rate-limited and irreversible:

  * `--dry-run` (the default for `submit`) walks the entire path, finds the file
    input and reports what it found, and stops before uploading. Use it to prove
    the selectors work without spending one of the day's five.
  * an actual upload needs `--yes`, and is refused once the local log shows five
    submissions inside the last 24 hours.

    python codabench_submit.py login
    python codabench_submit.py status
    python codabench_submit.py submit artifacts/colab/best_single.zip          # dry run
    python codabench_submit.py submit artifacts/colab/best_single.zip --yes    # real
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

COMPETITION = 17670
DEV_PHASE = 29478
FINAL_PHASE = 29479
COMPETITION_URL = f"https://www.codabench.org/competitions/{COMPETITION}/"
DAILY_LIMIT = 5

REPO = Path(__file__).parent
LOG_PATH = REPO / "results" / "submission_log.json"
# Kept out of the repository: it holds a live login session.
PROFILE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "codabench_probe_profile"

# Playwright lives in the Browse Automation venv on this machine.
BROWSE_VENV = Path(r"C:\Users\BLESSING MAMBWE\Pictures\Projects\Browse Automation\.venv\Scripts\python.exe")


# --------------------------------------------------------------------------
# Budget log
# --------------------------------------------------------------------------


def load_log() -> list[dict]:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    return []


def save_log(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def recent_count(rows: list[dict], hours: int = 24) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return sum(1 for r in rows if datetime.fromisoformat(r["utc"]) > cutoff)


def budget_report(rows: list[dict]) -> str:
    used = recent_count(rows)
    lines = [f"submissions in the last 24h: {used}/{DAILY_LIMIT}"]
    if used >= DAILY_LIMIT:
        oldest = min(datetime.fromisoformat(r["utc"]) for r in rows[-DAILY_LIMIT:])
        lines.append(f"  budget frees up at {(oldest + timedelta(hours=24)).isoformat(timespec='minutes')}")
    for r in rows[-6:]:
        lines.append(f"  {r['utc'][:16]}  {r.get('label', ''):<22} {Path(r['zip']).name}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Public leaderboard (no auth needed)
# --------------------------------------------------------------------------


def fetch_leaderboard() -> list[tuple[float, str, str]]:
    url = "https://www.codabench.org/api/leaderboards/19655/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=45))
    rows = []
    for s in data["submissions"]:
        acc = [x["score"] for x in s["scores"] if x["column_key"] == "accuracy"][0]
        rows.append((float(acc), s["owner"], s["created_when"][:16]))
    rows.sort(reverse=True)
    return rows


def phase_deadlines() -> dict:
    req = urllib.request.Request(f"https://www.codabench.org/api/competitions/{COMPETITION}/",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=45))
    return {p["name"]: (p["start"], p["end"]) for p in data["phases"]}


# --------------------------------------------------------------------------
# Browser driving. Runs in a subprocess against the venv that has Playwright.
# --------------------------------------------------------------------------


DRIVER = r'''
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

cfg = json.loads(sys.argv[1])
action, profile_dir, url = cfg["action"], cfg["profile_dir"], cfg["url"]
zip_path, do_upload, phase = cfg["zip_path"], cfg["do_upload"], cfg["phase"]
result = {"action": action, "ok": False, "notes": []}


def note(msg):
    result["notes"].append(msg)
    print(msg, flush=True)


with sync_playwright() as pw:
    context = pw.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False,
        args=["--disable-blink-features=AutomationControlled"],
        viewport={"width": 1500, "height": 950},
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.set_default_timeout(45000)
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(6000)

    # Signed-out is the reliable thing to detect: the header shows Login and
    # Sign-up. Probing for profile links instead gives false positives, because
    # the leaderboard is full of links to other people's profiles.
    signed_in = True
    for probe in ["header >> text=Login", "nav >> text=Login", "text=Sign-up"]:
        try:
            if page.locator(probe).first.is_visible(timeout=2500):
                signed_in = False
                note(f"signed-OUT indicator present: {probe}")
                break
        except Exception:
            continue
    result["signed_in"] = signed_in
    if signed_in:
        note("appears signed in")

    if action == "login":
        if signed_in:
            note("already signed in on this profile; nothing to do")
        else:
            page.goto("https://www.codabench.org/accounts/login/",
                      wait_until="domcontentloaded")
            note("Sign in to CodaBench in the window that opened, then leave it alone.")
            note("Waiting up to 10 minutes...")
            # "My Submissions" is visible signed-out too, so it proves nothing.
            # Sign-in is the *disappearance* of the Login / Sign-up header links.
            for _ in range(120):
                page.wait_for_timeout(5000)
                try:
                    still_out = any(
                        page.locator(p).first.is_visible(timeout=1200)
                        for p in ["text=Sign-up", "header >> text=Login"]
                    )
                except Exception:
                    still_out = True
                if not still_out:
                    signed_in = True
                    break

        result["signed_in"] = signed_in
        if signed_in:
            note("signed in; session stored in the profile directory")
            # Being signed in is not the same as being allowed to submit: this
            # competition gates participation on organizer approval.
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(6000)
            for probe in ["text=My Submissions", "a:has-text('My Submissions')"]:
                try:
                    page.locator(probe).first.click(timeout=6000)
                    break
                except Exception:
                    continue
            page.wait_for_timeout(5000)
            body = page.inner_text("body")
            inputs = page.locator("input[type=file]").count()
            result["file_inputs"] = inputs
            if inputs:
                result["registered"] = True
                note(f"registered participant: upload control present ({inputs} file input(s))")
            elif "not yet registered" in body or "requires approval" in body:
                result["registered"] = False
                note("SIGNED IN BUT NOT AN APPROVED PARTICIPANT -- request registration on "
                     "the competition page; the organizer must approve it before any submission")
            else:
                result["registered"] = None
                note("signed in; could not confirm participant status from the page")
            page.screenshot(path=str(Path(profile_dir).parent / "codabench_login_state.png"),
                            full_page=True)
            result["ok"] = True
        else:
            note("timed out waiting for sign-in")
        context.close()
        print("RESULT " + json.dumps(result))
        raise SystemExit(0)

    # Open the submissions tab.
    for probe in ["text=My Submissions", "a:has-text('My Submissions')", "#participate-tab"]:
        try:
            page.locator(probe).first.click(timeout=6000)
            note(f"clicked {probe}")
            break
        except Exception:
            continue
    page.wait_for_timeout(5000)

    # The Development and Testing phases have separate upload panels. Landing on
    # the wrong one would spend a submission against the wrong phase, so the
    # phase is selected explicitly rather than relying on the default.
    # The phase switch is `<div class="ui button">Development Phase </div>` --
    # a div, so it has no button role, and its text has a trailing space, so an
    # exact text match misses it too. Both cost a dry run each to discover.
    phase_clicked = False
    try:
        control = page.locator(f'div.ui.button:has-text("{phase}")').first
        if "active" in (control.get_attribute("class") or ""):
            note(f"phase already active: {phase}")
        else:
            control.click(timeout=5000)
            page.wait_for_timeout(3000)
            note(f"selected phase: {phase}")
        phase_clicked = True
    except Exception as exc:
        note(f"WARNING: could not select phase {phase!r}: {type(exc).__name__}")
    result["phase"] = phase
    result["phase_clicked"] = phase_clicked

    # An upload into the wrong phase is unrecoverable, so when we are actually
    # uploading, refuse rather than trust that the default happened to be right.
    if do_upload and not phase_clicked:
        note("ABORT: refusing to upload without having positively selected the phase")
        context.close()
        print("RESULT " + json.dumps(result))
        raise SystemExit(0)

    # The page states the real budget; trust it over any local bookkeeping.
    import re as _re
    body = page.inner_text("body")
    day = _re.search(r"(\d+)\s*out of\s*(\d+)", body)
    allm = _re.findall(r"(\d+)\s*out of\s*(\d+)", body)
    if allm:
        result["budget"] = [f"{a}/{b}" for a, b in allm]
        note("platform counters (day, total): " + ", ".join(result["budget"]))
        try:
            used_today, limit_today = int(allm[0][0]), int(allm[0][1])
            result["used_today"] = used_today
            result["limit_today"] = limit_today
            if do_upload and used_today >= limit_today:
                note(f"ABORT: platform reports {used_today}/{limit_today} used today")
                context.close()
                print("RESULT " + json.dumps(result))
                raise SystemExit(0)
        except (ValueError, IndexError):
            pass

    inputs = page.locator("input[type=file]")
    count = inputs.count()
    note(f"file inputs found: {count}")
    result["file_inputs"] = count

    # Existing submission ids, so a new one can be detected after upload.
    # The page has seven tables; `inner_text("table")` raises on the ambiguity
    # and would silently yield no ids. One of the seven is the *leaderboard*,
    # which carries other people's submission ids -- scraping that would report
    # a stranger's upload as ours. So: find the table with a "File name" header.
    def submission_ids():
        found = set()
        try:
            tables = page.locator("table")
            for i in range(tables.count()):
                text = tables.nth(i).inner_text()
                if "File name" not in text:
                    continue
                found |= set(_re.findall(r"\b\d{6}\b", text))
        except Exception as exc:
            note(f"could not read the submissions table: {type(exc).__name__}")
        return found

    before = submission_ids()
    result["ids_before"] = len(before)

    page.screenshot(path=str(Path(profile_dir).parent / "codabench_submit_page.png"),
                    full_page=True)
    note("screenshot written next to the profile dir")

    if count == 0:
        # CodaBench builds the upload control only for a registered participant,
        # and registration for this competition needs organizer approval, so a
        # signed-in account is not by itself enough.
        if not signed_in:
            note("no file input because the session is signed out -- run: login")
        else:
            body = page.inner_text("body")[:400].replace("\n", " ")
            if "not yet registered" in body or "requires approval" in body:
                note("signed in but not an approved participant for this competition")
            else:
                note("signed in and registered, but no file input found -- layout changed")
    elif not do_upload:
        note("DRY RUN: file input located; stopping before upload")
        result["ok"] = True
    else:
        # CodaBench starts the upload from the file input's change event. There
        # is no submit button to press -- the twelve "Submit" strings on the page
        # are table headers and tab labels, and clicking one would do something
        # other than what we mean.
        inputs.first.set_input_files(zip_path)
        note(f"attached {zip_path}; upload starts on selection")

        new_id = None
        for _ in range(40):  # up to ~3 minutes for upload plus scoring to appear
            page.wait_for_timeout(5000)
            added = submission_ids() - before
            if added:
                new_id = sorted(added)[-1]
                break
        page.wait_for_timeout(3000)
        page.screenshot(path=str(Path(profile_dir).parent / "codabench_after_submit.png"),
                        full_page=True)
        if new_id:
            result["ok"] = True
            result["submission_id"] = new_id
            note(f"submission created: id {new_id}")
        else:
            note("no new submission row appeared -- check codabench_after_submit.png "
                 "before retrying, so a silent success is not submitted twice")

    context.close()
print("RESULT " + json.dumps(result))
'''


def run_driver(action: str, zip_path: str = "", do_upload: bool = False,
               phase: str = "Development Phase") -> dict:
    python = str(BROWSE_VENV) if BROWSE_VENV.exists() else sys.executable
    payload = json.dumps({"action": action, "profile_dir": str(PROFILE_DIR),
                          "url": COMPETITION_URL, "zip_path": zip_path,
                          "do_upload": do_upload, "phase": phase})
    script = REPO / "results" / "_codabench_driver.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(DRIVER, encoding="utf-8")
    proc = subprocess.run([python, str(script), payload], capture_output=True, text=True,
                          timeout=900)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"ok": False, "notes": ["driver produced no result"]}


# --------------------------------------------------------------------------


def cmd_status(_args) -> int:
    print("=== leaderboard (last submission per user) ===")
    try:
        for acc, owner, when in fetch_leaderboard():
            mark = "  <-- us" if owner == "bleymambwe" else ""
            print(f"  {acc:.4f}  {int(round(acc * 1700)):>4}/1700  {owner:<16} {when}{mark}")
    except Exception as e:
        print(f"  leaderboard unavailable: {type(e).__name__}: {e}")
    print()
    try:
        for name, (start, end) in phase_deadlines().items():
            print(f"  {name}: {start} -> {end}")
    except Exception as e:
        print(f"  phases unavailable: {type(e).__name__}")
    print()
    print("=== local submission budget ===")
    print(budget_report(load_log()))
    return 0


def cmd_login(_args) -> int:
    print(f"profile: {PROFILE_DIR}")
    result = run_driver("login")
    return 0 if result.get("ok") else 1


def cmd_submit(args) -> int:
    zip_path = Path(args.zip).resolve()
    if not zip_path.exists():
        print(f"no such file: {zip_path}")
        return 1

    # Never upload something that would be rejected or that we have not verified.
    sys.path.insert(0, str(REPO / "src"))
    from probe.submission import smoke_test  # noqa: E402

    import zipfile
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(archive.namelist())
    if names != ["classifier.py", "trained_probe.joblib"]:
        print(f"REFUSED: zip root must be exactly the two required files, got {names}")
        return 1
    try:
        check = smoke_test(zip_path, n=1700)
        print(f"smoke test ok: {check}")
    except Exception as e:
        print(f"REFUSED: smoke test failed: {type(e).__name__}: {e}")
        return 1

    rows = load_log()
    used = recent_count(rows)
    if args.yes and used >= DAILY_LIMIT:
        print(f"REFUSED: {used}/{DAILY_LIMIT} submissions already used in the last 24h")
        print(budget_report(rows))
        return 1

    print(f"\n{'UPLOADING' if args.yes else 'DRY RUN'}: {zip_path.name}")
    print(budget_report(rows))
    result = run_driver("submit", str(zip_path), do_upload=bool(args.yes),
                        phase=args.phase)

    if args.yes and result.get("ok"):
        rows.append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "zip": str(zip_path), "label": args.label or zip_path.stem,
                     "phase": args.phase,
                     "submission_id": result.get("submission_id")})
        save_log(rows)
        print("\nlogged. Re-check the score with: python codabench_submit.py status")
    return 0 if result.get("ok") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login", help="open a browser and sign in once")
    sub.add_parser("status", help="leaderboard, deadlines and remaining budget")
    s = sub.add_parser("submit", help="upload a submission zip")
    s.add_argument("zip")
    s.add_argument("--yes", action="store_true", help="actually upload (default is a dry run)")
    s.add_argument("--label", default="", help="short name for the local log")
    s.add_argument("--phase", default="Development Phase",
                   choices=["Development Phase", "Testing Phase"],
                   help="which phase panel to upload into")

    args = parser.parse_args()
    return {"login": cmd_login, "status": cmd_status, "submit": cmd_submit}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
