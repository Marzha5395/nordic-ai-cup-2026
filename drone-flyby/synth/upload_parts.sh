#!/bin/bash
# Upload dataset tar parts (+ manifest last) to the Colab session, 3 in parallel,
# skipping parts already present, refreshing the proxy token periodically.
# Usage: synth/upload_parts.sh <parts-dir> <prefix>   e.g. data/parts2 yolo2_data
DIR=${1:?parts dir}; PFX=${2:?prefix}
cd "$DIR" || exit 1
CPY=~/.local/share/uv/tools/google-colab-cli/bin/python
REFRESH=/home/simonph/progproject/Ai/drone-flyby/synth/refresh_colab_token.py
timeout 90 $CPY $REFRESH >/dev/null 2>&1
have=$(timeout 60 ~/.local/bin/colab --auth=oauth2 ls -s trainer /content 2>/dev/null | grep -o "${PFX}.tar.part[0-9]*" | sort -u)
up() { f="$1"; for a in 1 2 3; do timeout 180 ~/.local/bin/colab --auth=oauth2 upload -s trainer "$PWD/$f" "/content/$f" >/dev/null 2>&1 && { echo "ok $f"; return 0; }; sleep 3; done; echo "FAILED $f"; }
export -f up
n=0
for f in $(ls ${PFX}.tar.part*); do
  echo "$have" | grep -q "^$f$" && continue
  n=$((n+1))
  if [ $((n % 30)) -eq 0 ]; then timeout 90 $CPY $REFRESH >/dev/null 2>&1; fi
  echo "$f"
done | xargs -P 3 -I{} bash -c 'up {}'
timeout 90 $CPY $REFRESH >/dev/null 2>&1
up ${PFX}.manifest.json
echo "upload finished $(date +%H:%M): remote has $(timeout 60 ~/.local/bin/colab --auth=oauth2 ls -s trainer /content 2>/dev/null | grep -c ${PFX}) ${PFX} files"
