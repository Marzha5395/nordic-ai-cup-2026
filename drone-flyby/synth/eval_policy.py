"""In-process policy evaluation with per-object diagnostics.

Drives example.predict over a scene exactly like local_evaluator (same Camera
rules, same crops/PNG payloads, same scorer) but without HTTP, then explains
the score: per-class AP, per-object hit statistics (frames visible, frames
with a matching prediction at IoU>=0.5 of the right class, first hit delay),
and where the misses come from.

    DRONE_WEIGHTS=weights/colab/last.pt .venv/bin/python synth/eval_policy.py --scene synth250
    ... --frames 120            # first 120 frames only
    ... --dump out.json         # save per-frame predictions
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dtos import DroneFlybyPredictRequestDto  # noqa: E402
from local_evaluator import Camera, CameraRejection, build_request, render_view, score  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame, validate_response  # noqa: E402


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="synth250")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    import policy_controller as example  # noqa: E402  (loads the detector)

    frames = frame_numbers(args.scene)
    if args.frames:
        frames = frames[:args.frames]
    camera, feedback = Camera(), None
    preds, refused, times, path = {}, 0, [], []
    for idx, fr in enumerate(frames):
        img = load_frame(fr, args.scene)
        payload = build_request(fr, idx, camera, render_view(img, camera), feedback)
        path.append((fr, camera.resolution_level, camera.center_x, camera.center_y))
        t0 = time.monotonic()
        resp = example.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        times.append((time.monotonic() - t0) * 1000)
        validate_response(resp)
        assert resp.request_id == payload["request_id"] and resp.frame == fr
        preds[fr] = [{"object_id": a.object_id,
                      "bbox": [a.bbox[0] * 3840, a.bbox[1] * 2160, a.bbox[2] * 3840, a.bbox[3] * 2160],
                      "confidence": float(a.confidence)} for a in resp.annotations]
        feedback = None
        if resp.requested_view is not None:
            rv = resp.requested_view
            try:
                camera.apply(rv.resolution_level, rv.center_x, rv.center_y)
            except CameraRejection as exc:
                refused += 1
                feedback = {"frame": fr, "requested_view": rv.model_dump(), "reason": str(exc)}
                print(f"frame {fr}: REFUSED {exc}")

    # ---- scoring (the scene may have more frames than we ran: score only ours) ----
    gt = {fr: load_annotations(fr, args.scene) for fr in frames}
    if args.frames:
        # local_evaluator.score scores every frame of the scene; restrict by
        # feeding empty predictions for frames we skipped is wrong, so instead
        # build a temporary sub-scene view via monkeypatching frame_numbers.
        import local_evaluator
        local_evaluator.frame_numbers = lambda scene: frames
    m, ap_cls = score(args.scene, preds)
    print(f"\n=== {args.scene}: mAP@0.5 = {m:.3f}  (frames {len(frames)}, refused moves {refused}, "
          f"predict ms mean {np.mean(times):.0f} max {np.max(times):.0f})")
    print("per-class AP:", " ".join(f"{k}={v:.2f}" for k, v in sorted(ap_cls.items(), key=lambda kv: kv[1])))

    # ---- per-object diagnostics (needs "oid" in the GT) ----
    if not any("oid" in a for a in gt[frames[0]]):
        print("(no oid in GT; skipping per-object diagnostics)")
        return
    vis = defaultdict(list)      # oid -> [(frame, gt)]
    for fr in frames:
        for a in gt[fr]:
            vis[a["oid"]].append((fr, a))
    # camera coverage per object: how many frames its GT box was inside a view at each level
    region_of = {}
    for fr, lvl, cx, cy in path:
        half = {0: (1920, 1080), 1: (960, 540), 2: (480, 270)}[lvl]
        region_of[fr] = (lvl, [cx - half[0], cy - half[1], cx + half[0], cy + half[1]])
    looks = defaultdict(lambda: [0, 0, 0])       # oid -> frames fully inside a view, per level
    for fr in frames:
        lvl, r = region_of[fr]
        for a in gt[fr]:
            b = a["bbox"]
            if b[0] >= r[0] and b[1] >= r[1] and b[2] <= r[2] and b[3] <= r[3]:
                looks[a["oid"]][lvl] += 1
    rows, fp_kind = [], defaultdict(int)
    for fr in frames:
        matched, matched_any = set(), set()
        gts = gt[fr]
        for p in sorted(preds[fr], key=lambda p: -p["confidence"]):
            ious = [(iou(p["bbox"], a["bbox"]), a) for a in gts]
            same = [(v, a) for v, a in ious if a["object_id"] == p["object_id"] and a["oid"] not in matched]
            best = max(same, key=lambda t: t[0], default=(0.0, None))
            if best[0] >= 0.5:
                matched.add(best[1]["oid"])
                matched_any.add(best[1]["oid"])
                continue
            other = max(ious, key=lambda t: t[0], default=(0.0, None))
            if other[0] >= 0.5 and other[1]["object_id"] != p["object_id"]:
                fp_kind["wrong_class"] += 1
                matched_any.add(other[1]["oid"])
            elif other[0] >= 0.5:
                fp_kind["duplicate"] += 1
            elif other[0] >= 0.2:
                fp_kind["loose_box(0.2<=iou<0.5)"] += 1
            else:
                fp_kind["ghost"] += 1
        for a in gts:
            rows.append((a["oid"], fr, a["oid"] in matched, a["oid"] in matched_any))
    hit, hit_any = defaultdict(list), defaultdict(list)
    for oid, fr, ok, ok_any in rows:
        hit[oid].append(ok)
        hit_any[oid].append(ok_any)
    fp_total = sum(fp_kind.values())
    print(f"\nfalse positives (class-aware, IoU<0.5): {fp_total} over {len(frames)} frames "
          f"({fp_total / len(frames):.1f}/frame); GT boxes {sum(len(v) for v in gt.values())}")
    print("  by kind:", dict(fp_kind))
    print(f"{'oid':>4} {'class':<16} {'vis':>4} {'hit':>4} {'rate':>5} {'anycls':>6} {'looks L0/1/2':>13}  first_hit  size@first note")
    weak, weak_any = defaultdict(list), defaultdict(list)
    for oid in sorted(vis, key=lambda o: vis[o][0][0]):
        h, ha = hit[oid], hit_any[oid]
        first = next((i for i, ok in enumerate(h) if ok), None)
        fr0, a0 = vis[oid][0]
        w, hgt = a0["bbox"][2] - a0["bbox"][0], a0["bbox"][3] - a0["bbox"][1]
        rate, rate_any = sum(h) / len(h), sum(ha) / len(ha)
        weak[a0["object_id"]].append(rate)
        weak_any[a0["object_id"]].append(rate_any)
        lk = looks[oid]
        note = "NEVER" if first is None else ("late" if first > 12 else "")
        if first is None and sum(lk) == 0:
            note += " (no full look)"
        elif first is None and rate_any > 0.3:
            note += " (class wrong)"
        if rate < 0.6 or not args.quiet:
            print(f"{oid:>4} {a0['object_id']:<16} {len(h):>4} {sum(h):>4} {rate:5.2f} {rate_any:6.2f} "
                  f"{lk[0]:>4}/{lk[1]:>3}/{lk[2]:>3}  {'-' if first is None else first:>9}  {w}x{hgt}@{fr0} {note}")
    print("\nper-class mean hit rate (class-aware):",
          " ".join(f"{c}={np.mean(weak[c]):.2f}" for c in sorted(weak, key=lambda c: np.mean(weak[c]))))
    print("per-class mean hit rate (any class):  ",
          " ".join(f"{c}={np.mean(weak_any[c]):.2f}" for c in sorted(weak, key=lambda c: np.mean(weak[c]))))
    n_obj = len(vis)
    print(f"objects: {n_obj}; never hit: {sum(1 for o in vis if not any(hit[o]))}; "
          f"never hit any-class: {sum(1 for o in vis if not any(hit_any[o]))}; "
          f"objects with zero full L2 looks: {sum(1 for o in vis if looks[o][2] == 0)}")
    lv = defaultdict(int)
    for _, lvl, _, _ in path:
        lv[lvl] += 1
    print("camera levels used:", dict(lv))
    if args.dump:
        Path(args.dump).write_text(json.dumps({"preds": preds, "path": path, "map": m, "ap": ap_cls}))


if __name__ == "__main__":
    main()
