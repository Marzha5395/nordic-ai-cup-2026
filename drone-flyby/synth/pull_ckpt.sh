#!/bin/bash
# Pull training artefacts from the Colab session every 10 minutes (up to 5 h),
# refreshing the runtime proxy token first (the CLI never refreshes it; it expires after 60 min).
# Usage: synth/pull_ckpt.sh <run-name>   (e.g. y11s_960)
RUN=${1:-y11s_960}
cd /home/simonph/progproject/Ai/drone-flyby
CPY=~/.local/share/uv/tools/google-colab-cli/bin/python
mkdir -p weights/colab/$RUN
for i in $(seq 1 30); do
  sleep 600
  ts=$(date +%H:%M)
  timeout 90 $CPY synth/refresh_colab_token.py >/dev/null 2>&1 || echo "$ts token refresh FAILED"
  timeout 120 ~/.local/bin/colab --auth=oauth2 download -s trainer /content/runs/$RUN/results.csv weights/colab/$RUN/results.csv >/dev/null 2>&1 \
    && echo "$ts results.csv ok ($(wc -l < weights/colab/$RUN/results.csv) lines)" || echo "$ts results.csv missing"
  for w in last best; do
    timeout 300 ~/.local/bin/colab --auth=oauth2 download -s trainer /content/runs/$RUN/weights/$w.pt weights/colab/$RUN/$w.pt.tmp >/dev/null 2>&1 \
      && mv weights/colab/$RUN/$w.pt.tmp weights/colab/$RUN/$w.pt && echo "$ts $w.pt ok ($(stat -c %s weights/colab/$RUN/$w.pt) bytes)" || echo "$ts $w.pt missing"
  done
done
