#!/usr/bin/env bash
# Usage: run_ablations.sh <weights> <scene1> [scene2 ...]
# Runs synth/eval_policy.py over a fixed config grid and prints a markdown table.
set -u
cd "$(dirname "$0")/.."

WEIGHTS="$1"; shift
STEM=$(basename "$WEIGHTS" .pt)
OUTDIR="logs/ablations"
mkdir -p "$OUTDIR"

CONFIGS=(
  "base|"
  "no_refresh|DRONE_L1_REFRESH=0"
  "sweep_l1|DRONE_SWEEP_LEVEL=1"
  "no_second|DRONE_SECOND_CLASS=0"
  "tta|DRONE_TTA=1"
  "l0up|DRONE_L0_UPSCALE=2"
  "conf10|DRONE_CONF=0.10"
  "conf05|DRONE_CONF=0.05"
  "tau05|DRONE_CONF_TAU=0.5"
  "second15|DRONE_SECOND_P=0.15"
  "hybrid|DRONE_SWEEP_LEVEL=1 DRONE_L2_DIPS=1"
  "l1_nodips|DRONE_SWEEP_LEVEL=1 DRONE_L2_DIPS=0"
  "l2base|DRONE_SWEEP_LEVEL=2"
  "hybrid_t|DRONE_SWEEP_LEVEL=1 DRONE_L2_DIPS=1 DRONE_DIP_MIN_TOTAL=0.4"
)

if [ -n "${ABL_CONFIGS:-}" ]; then
  keep=" $ABL_CONFIGS "
  sel=()
  for entry in "${CONFIGS[@]}"; do
    case "$keep" in *" ${entry%%|*} "*) sel+=("$entry");; esac
  done
  CONFIGS=("${sel[@]}")
fi

ROWS=()
for scene in "$@"; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"
    envs="${entry#*|}"
    log="$OUTDIR/${STEM}_${scene}_${name}.txt"
    echo ">>> $scene / $name  ($envs)"
    env DRONE_WEIGHTS="$WEIGHTS" $envs \
      .venv/bin/python synth/eval_policy.py --scene "$scene" --quiet \
      > "$log" 2>&1
    ROWS+=("$scene|$name|$log")
  done
done

python3 - "${ROWS[@]}" <<'EOF'
import re, sys

def field(txt, pat, cast=float):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else None

print("| config | scene | mAP | FP/frame | ghosts | never hit | ms |")
print("|---|---|---|---|---|---|---|")
for row in sys.argv[1:]:
    scene, name, log = row.split("|")
    txt = open(log).read()
    map_ = field(txt, r"mAP@0\.5 = ([0-9.]+)")
    ms = field(txt, r"predict ms mean (\d+)")
    fp = field(txt, r"false positives[^:]*: \d+ over \d+ frames \(([0-9.]+)/frame\)")
    ghosts = field(txt, r"'ghost': (\d+)", int)
    never = field(txt, r"never hit: (\d+)", int)
    print(f"| {name} | {scene} | {map_} | {fp} | {ghosts} | {never} | {ms} |")
EOF
