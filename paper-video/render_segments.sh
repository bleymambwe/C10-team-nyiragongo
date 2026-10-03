#!/bin/bash
# Render the video in frame-range segments, then concatenate.
#
# A single `remotion render` of ~22k frames takes over an hour here and cannot
# resume: stopping it discards everything. Segments are individually durable,
# so an interruption costs at most one chunk, and re-running skips whatever is
# already on disk.
#
# Segments are retried: the composition pulls IBM Plex from Google Fonts at
# render time, so a transient "NetworkError" can kill an otherwise fine chunk.
set -u
cd "$(dirname "$0")"

TOTAL=$(node -e "
const n=require('./src/narration.json');
const FPS=30,T=12;
const f=n.scenes.map(s=>Math.round((s.durationSec+0.55)*FPS));
console.log(f.reduce((a,b)=>a+b,0)-T*(f.length-1));
")
CHUNK=${CHUNK:-2000}
RETRIES=${RETRIES:-4}
CONC=${CONC:-2}   # 4 cores; 3 concurrent Chromes can miss the 30s startup timeout
SEGDIR=../out/segments
mkdir -p "$SEGDIR" ../out

echo "total frames: $TOTAL, chunk size: $CHUNK, retries: $RETRIES, concurrency: $CONC"
LIST="$SEGDIR/list.txt"
: > "$LIST"

noise='npm notice|ExperimentalWarning|trace-warnings|^\(Use'


# A segment is valid only if it decodes and has exactly the expected frames.
# `test -s` is not enough: an interrupted render leaves a non-empty but
# truncated mp4 (no moov atom, or a few frames), which then corrupts the concat.
valid_segment() {
  local f=$1 want=$2
  [ -s "$f" ] || return 1
  local got
  got=$(ffprobe -v error -count_frames -select_streams v:0         -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null | tr -d ',')
  [ -n "$got" ] && [ "$got" = "$want" ]
}

i=0
start=0
while [ "$start" -lt "$TOTAL" ]; do
  end=$((start + CHUNK - 1))
  [ "$end" -ge "$TOTAL" ] && end=$((TOTAL - 1))
  out=$(printf "%s/seg%02d.mp4" "$SEGDIR" "$i")

  want=$((end - start + 1))
  if valid_segment "$out" "$want"; then
    echo "seg $i ($start-$end) already rendered, skipping"
  else
    ok=0
    for attempt in $(seq 1 "$RETRIES"); do
      echo "=== seg $i: frames $start-$end (attempt $attempt/$RETRIES) ==="
      npx remotion render PaperVideo "$out" \
        --frames="$start-$end" --concurrency="$CONC" 2>&1 |
        grep -viE "$noise" | grep -iE "^.*(Error|Encoded [0-9]+/|\+ )" | tail -3
      if valid_segment "$out" "$want"; then ok=1; break; fi
      echo "  attempt $attempt produced no valid segment ($want frames expected); retrying"
      rm -f "$out"
      sleep $((attempt * 15))
    done
    if [ "$ok" -ne 1 ]; then
      echo "SEGMENT $i FAILED (frames $start-$end) after $RETRIES attempts."
      echo "Re-run this script; completed segments are skipped."
      exit 1
    fi
  fi
  echo "file '$(basename "$out")'" >> "$LIST"
  i=$((i + 1))
  start=$((end + 1))
done

echo "=== concatenating $i segments ==="
ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy ../out/seeing-is-not-steering.mp4 2>&1 | tail -2
echo "=== result ==="
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_type,codec_name \
  -of default=noprint_wrappers=1 ../out/seeing-is-not-steering.mp4
echo "SEGMENTED_RENDER_DONE"
