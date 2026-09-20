"""Colab probe: install deps, report hardware, then hold the kernel busy with a
heartbeat so we can see whether a busy session survives idle pruning."""
import os
import subprocess
import sys
import time


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


t0 = time.time()
print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"), flush=True)
print("cpus", os.cpu_count(), "py", sys.version.split()[0], flush=True)
print(sh("pip install -q ultralytics 2>&1 | tail -1"), flush=True)
print(sh("python -c \"import ultralytics, torch; print('ultralytics', ultralytics.__version__, "
         "'torch', torch.__version__, 'cuda', torch.cuda.is_available())\""), flush=True)
print(sh("df -h /content | tail -1"), flush=True)
print(f"setup done in {time.time() - t0:.0f}s", flush=True)

minutes = int(os.environ.get("PROBE_MINUTES", "25"))
for i in range(minutes):
    time.sleep(60)
    print(f"heartbeat {i + 1}/{minutes} min", flush=True)
print("probe finished", flush=True)
