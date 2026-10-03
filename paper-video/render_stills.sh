#!/bin/bash
mkdir -p ../stills
python - <<'PY' > /tmp/scenes.txt
import json
n=json.load(open("src/narration.json"))
FPS=30
for i,s in enumerate(n["scenes"]):
    frames=round((s["durationSec"]+0.55)*FPS)
    fr=min(frames-8, int(9.5*FPS))
    print(f'{i:02d}-{s["id"].replace("_","-")} {fr}')
PY
while read id fr; do
  echo "--- $id @ $fr ---"
  npx remotion still "$id" "../stills/$id.png" --frame=$fr --scale=0.45 2>&1 | grep -iE "error occurred|Error:" | head -2
done < /tmp/scenes.txt
echo ALL_STILLS_DONE
