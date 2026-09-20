#!/bin/bash
# Wait for run 1 to print "training done", then upload /content/STOP so its
# hold loop ends and the queued run-2 cell starts.
cd /home/simonph/progproject/Ai/drone-flyby
CPY=~/.local/share/uv/tools/google-colab-cli/bin/python
until grep -a -q "training done" logs/colab_train_y11s.log; do sleep 60; done
echo "$(date +%H:%M) run 1 finished; downloading final weights before releasing the hold"
timeout 90 $CPY synth/refresh_colab_token.py >/dev/null 2>&1
mkdir -p weights/colab/y11s_960
for w in best last; do
  timeout 300 ~/.local/bin/colab --auth=oauth2 download -s trainer /content/runs/y11s_960/weights/$w.pt weights/colab/y11s_960/$w.pt \
    && echo "$(date +%H:%M) $w.pt downloaded ($(stat -c %s weights/colab/y11s_960/$w.pt) bytes)"
done
timeout 120 ~/.local/bin/colab --auth=oauth2 download -s trainer /content/runs/y11s_960/results.csv weights/colab/y11s_960/results.csv
touch /tmp/STOP && timeout 120 ~/.local/bin/colab --auth=oauth2 upload -s trainer /tmp/STOP /content/STOP && echo "$(date +%H:%M) STOP uploaded"
