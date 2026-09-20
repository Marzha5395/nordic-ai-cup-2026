"""Training driver for the Colab T4 VM (run via `colab exec -s trainer -f`).

The Colab runtime is recycled after ~10 minutes of kernel idleness, so this
one cell must own the kernel for the whole job: it polls until the dataset
upload is complete (manifest + all parts present), reassembles and untars
/content/yolo_data.tar.part* into /content/data/yolo, trains, then keeps the
kernel busy for HOLD_MINUTES so the weights can be downloaded.
If /content/runs/<name>/weights/last.pt exists the run is resumed instead, so
a fresh VM can continue an interrupted run after data and last.pt are
uploaded again.

Settings come from environment variables (set them in a preamble cell sent
together with this file): TRAIN_MODEL (yolo11s.pt), TRAIN_EPOCHS (24),
TRAIN_BATCH (16), TRAIN_NAME (y11s_960), TRAIN_IMGSZ (960), WAIT_MINUTES
(75), HOLD_MINUTES (150), TRAIN_DATA_NAME (yolo: the tar's top-level dir and
the part-file prefix), TRAIN_WARMUP_EPOCHS (3.0), TRAIN_STOP_FILE.
"""
import glob
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

DATA_NAME = os.environ.get("TRAIN_DATA_NAME", "yolo")   # tar top-level dir + part prefix
DATA = Path("/content/data") / DATA_NAME
PARTS = f"/content/{DATA_NAME}_data.tar.part*"
MANIFEST = Path(f"/content/{DATA_NAME}_data.manifest.json")
TAR = f"/content/{DATA_NAME}_data.tar"
RUNS = Path("/content/runs")
STOP = Path(os.environ.get("TRAIN_STOP_FILE", "/content/STOP"))

MODEL = os.environ.get("TRAIN_MODEL", "yolo11s.pt")
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "24"))
BATCH = int(os.environ.get("TRAIN_BATCH", "16"))
NAME = os.environ.get("TRAIN_NAME", "y11s_960")
IMGSZ = int(os.environ.get("TRAIN_IMGSZ", "960"))
WAIT_MINUTES = float(os.environ.get("WAIT_MINUTES", "75"))
HOLD_MINUTES = float(os.environ.get("HOLD_MINUTES", "150"))
WARMUP_EPOCHS = float(os.environ.get("TRAIN_WARMUP_EPOCHS", "3.0"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wait_for_upload():
    """Block (kernel busy) until the manifest and every listed part are here."""
    t0 = time.time()
    last_report = 0
    while time.time() - t0 < WAIT_MINUTES * 60:
        parts = sorted(glob.glob(PARTS))
        have = sum(os.path.getsize(p) for p in parts)
        if MANIFEST.exists():
            m = json.loads(MANIFEST.read_text())
            if len(parts) == m["parts"] and have == m["total_bytes"]:
                log(f"upload complete: {len(parts)} parts, {have} bytes")
                return m
        if time.time() - last_report >= 60:
            log(f"waiting for upload: {len(parts)} parts, {have / 1e6:.0f} MB so far")
            last_report = time.time()
        time.sleep(15)
    sys.exit("upload did not complete in time")


def extract():
    if (DATA / "data.yaml").exists():
        log("data already extracted, skipping reassembly")
        return
    manifest = wait_for_upload()
    parts = sorted(glob.glob(PARTS))
    log(f"reassembling {len(parts)} parts -> {TAR}")
    h = hashlib.sha256()
    with open(TAR, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while chunk := f.read(1 << 22):
                    out.write(chunk)
                    h.update(chunk)
    if manifest.get("sha256") and h.hexdigest() != manifest["sha256"]:
        sys.exit(f"sha256 mismatch: {h.hexdigest()} != {manifest['sha256']}")
    log("sha256 verified")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    log(f"untarring into {DATA.parent}")
    with tarfile.open(TAR) as tf:
        tf.extractall(DATA.parent, filter="data")
    os.remove(TAR)
    for p in parts:
        os.remove(p)
    n_train = len(list((DATA / "images/train").glob("*.jpg")))
    n_val = len(list((DATA / "images/val").glob("*.jpg")))
    log(f"extracted: train={n_train} val={n_val}")


def fix_yaml():
    """Point data.yaml at this VM's paths; optionally add extra train dirs
    (TRAIN_EXTRA_TRAIN, comma-separated absolute image dirs)."""
    y = DATA / "data.yaml"
    lines = y.read_text().splitlines()
    extra = [e for e in os.environ.get("TRAIN_EXTRA_TRAIN", "").split(",") if e]
    out = []
    for ln in lines:
        if ln.startswith("path:"):
            ln = f"path: {DATA}"
        elif ln.startswith("train:") and extra:
            dirs = [str(DATA / "images/train")] + extra
            ln = "train: [" + ", ".join(dirs) + "]"
        out.append(ln)
    y.write_text("\n".join(out) + "\n")
    log(f"data.yaml path -> {DATA}" + (f" (+ extra train dirs {extra})" if extra else ""))


def hold():
    """Keep the kernel busy so the runtime is not recycled before downloads."""
    t0 = time.time()
    while time.time() - t0 < HOLD_MINUTES * 60 and not STOP.exists():
        log(f"holding runtime ({(time.time() - t0) / 60:.0f} min); create {STOP} to release")
        time.sleep(120)
    log("hold finished")


def ensure_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        import subprocess
        log("installing ultralytics")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    import torch
    import ultralytics
    log(f"ultralytics {ultralytics.__version__} torch {torch.__version__} cuda {torch.cuda.is_available()}")


def main():
    t0 = time.time()
    try:
        ensure_ultralytics()
        extract()
        fix_yaml()
        from ultralytics import YOLO

        last = RUNS / NAME / "weights" / "last.pt"
        if last.exists():
            log(f"resuming from {last}")
            YOLO(str(last)).train(resume=True)
        else:
            log(f"training {MODEL} imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH} name={NAME}")
            YOLO(MODEL).train(
                data=str(DATA / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS,
                batch=BATCH, device=0, workers=2, amp=True,
                project=str(RUNS), name=NAME, exist_ok=True,
                degrees=0, mosaic=1.0, scale=0.3, fliplr=0.5, flipud=0.5,
                translate=0.1, close_mosaic=4, patience=100, cos_lr=True,
                warmup_epochs=WARMUP_EPOCHS, plots=False, verbose=True)
        log(f"training done in {(time.time() - t0) / 60:.1f} min")
    except BaseException as exc:  # noqa: BLE001 - keep the runtime alive for diagnosis
        log(f"FAILED: {exc!r}")
    hold()


if __name__ == "__main__":
    main()
