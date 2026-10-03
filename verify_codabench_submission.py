"""Read-only verification of one known CodaBench submission ID."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


TERMINAL = {"Finished", "Failed", "Cancelled", "Canceled"}


def fetch(submission_id: int) -> dict:
    request = urllib.request.Request(
        f"https://www.codabench.org/api/submissions/{submission_id}/",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def summarize(data: dict) -> dict:
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "submission_id": data.get("id"),
        "phase": data.get("phase"),
        "phase_name": data.get("phase_name"),
        "filename": data.get("filename"),
        "owner": data.get("owner"),
        "created_when": data.get("created_when"),
        "status": data.get("status"),
        "status_details": data.get("status_details"),
        "scores": {item.get("column_key"): item.get("score")
                   for item in data.get("scores", [])},
        "leaderboard": data.get("leaderboard"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--expected-phase", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wait", type=int, default=720)
    args = parser.parse_args()

    deadline = time.monotonic() + args.max_wait
    while True:
        try:
            data = fetch(args.submission_id)
        except urllib.error.HTTPError as exc:
            # Fresh CodaBench submissions can briefly return 404 between the UI
            # row appearing and the public detail endpoint becoming visible.
            if exc.code == 404 and time.monotonic() < deadline:
                print(json.dumps({"submission_id": args.submission_id,
                                  "status": "API visibility pending"}), flush=True)
                time.sleep(5)
                continue
            raise
        result = summarize(data)
        result["phase_verified"] = data.get("phase") == args.expected_phase
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if not result["phase_verified"]:
            return 3
        if data.get("status") in TERMINAL:
            return 0 if data.get("status") == "Finished" else 4
        if time.monotonic() >= deadline:
            return 5
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
