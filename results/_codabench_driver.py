
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
