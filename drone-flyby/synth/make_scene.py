"""Render a long synthetic flight in the evaluator's scene format.

A ground canvas (Inria tiles at a fixed GSD) gets sprites pasted on it; every
frame is a perspective view of that canvas through the camera model fitted to
Helsinki (synth/helsinki_camera.json), moved by the fitted per-frame ground
translation. GT boxes are the AABB of each sprite's oriented footprint, the
same convention the detector labels use.

    .venv/bin/python synth/make_scene.py --name synth250 --frames 250 --seed 1

Output: src/<name>/{images,annotations}/frame_XXXXXX.{png,json},
run_metadata.json and review_frame0.jpg. Run local_evaluator.py --scene <name>.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from dtos import OBJECT_CLASSES  # noqa: E402
from make_dataset import footprint_for  # noqa: E402

SPRITES_DIR = ROOT / "data/sprites"
INRIA_DIR = ROOT / "data/backgrounds/inria"
CAM = json.loads((ROOT / "synth/helsinki_camera.json").read_text())
FW, FH = CAM["image_width"], CAM["image_height"]
INRIA_GSD = 0.3            # m/px of the Inria tiles
CANVAS_GSD = 0.19          # m/px of the ground canvas (about the frame-centre GSD)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ground_to_image():
    """3x3 map from ground metres (X, Y, 1) to image px, same model as the fit."""
    K = np.array([[CAM["focal_px"], 0, CAM["cx"]], [0, CAM["focal_px"], CAM["cy"]], [0, 0, 1]])
    R = rot_y(CAM["roll_rad"]) @ rot_x(CAM["pitch_rad"])
    return K @ R @ np.diag([1.0, 1.0, CAM["height_m"]])


def translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def apply_h(H, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def local_gsd(G, cx, cy):
    """Metres per source px around image point (cx, cy) (geometric mean of axes)."""
    Ginv = np.linalg.inv(G)
    p = apply_h(Ginv, np.array([[cx, cy], [cx + 1, cy], [cx, cy + 1]], float))
    return math.sqrt(np.hypot(*(p[1] - p[0])) * np.hypot(*(p[2] - p[0])))


def transform_sprite(sp, corners, scale, theta, rng):
    """flip? -> scale -> rotate; returns (rgba, corners) with corners following."""
    corners = corners.copy()
    if rng.random() < 0.5:
        sp = cv2.flip(sp, 1)
        corners[:, 0] = sp.shape[1] - corners[:, 0]
    h, w = sp.shape[:2]
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    sp = cv2.resize(sp, (sw, sh), interpolation=cv2.INTER_LINEAR)
    corners *= scale
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), theta, 1.0)
    rc = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], float) @ M[:, :2].T + M[:, 2]
    mn, mx = rc.min(0), rc.max(0)
    M[0, 2] -= mn[0]
    M[1, 2] -= mn[1]
    rw, rh = int(math.ceil(mx[0] - mn[0])), int(math.ceil(mx[1] - mn[1]))
    sp = cv2.warpAffine(sp, M, (rw, rh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sp, corners @ M[:, :2].T + M[:, 2]


def composite(canvas, sp, ox, oy):
    h, w = sp.shape[:2]
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(canvas.shape[1], ox + w), min(canvas.shape[0], oy + h)
    if x2 <= x1 or y2 <= y1:
        return
    a = sp[y1 - oy:y2 - oy, x1 - ox:x2 - ox, 3:4].astype(np.float32) / 255.0
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (a * sp[y1 - oy:y2 - oy, x1 - ox:x2 - ox, :3] + (1 - a) * roi).astype(np.uint8)


def build_canvas(x0, y0, cw, ch, rng):
    """Fill a cw x ch canvas (px at CANVAS_GSD) with Inria tiles, band by band."""
    tiles = sorted(INRIA_DIR.glob("*.tif"))
    rng.shuffle(tiles)
    canvas = np.empty((ch, cw, 3), np.uint8)
    f = INRIA_GSD / CANVAS_GSD                       # canvas px per tile px (~1.58)
    y, ti = 0, 0
    while y < ch:
        tile = cv2.imread(str(tiles[ti % len(tiles)]), cv2.IMREAD_COLOR)
        ti += 1
        th, tw = tile.shape[:2]
        band = min(ch - y, int(th * f))
        ww, wh = min(tw, int(math.ceil(cw / f))), min(th, int(math.ceil(band / f)))
        ox, oy = rng.integers(0, tw - ww + 1), rng.integers(0, th - wh + 1)
        piece = cv2.resize(tile[oy:oy + wh, ox:ox + ww], (cw, band), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            piece = cv2.flip(piece, 1)
        canvas[y:y + band] = piece
        y += band
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synth250")
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--density", type=float, default=11.0, help="objects per frame area")
    ap.add_argument("--step-scale", type=float, default=1.0, help="multiply the per-frame ground step")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = ROOT / "src" / args.name
    if out.exists():
        sys.exit(f"{out} exists; pick another --name")
    (out / "images").mkdir(parents=True)
    (out / "annotations").mkdir()

    G = ground_to_image()
    tx, ty = -CAM["tx_m"] * args.step_scale, -CAM["ty_m"] * args.step_scale  # ground shift per frame
    Ginv = np.linalg.inv(G)
    img_corners = np.array([[0, 0], [FW, 0], [FW, FH], [0, FH]], float)
    # Ground extent swept by all frames: frame t sees ground shifted by -t*(tx,ty)
    g0 = apply_h(Ginv, img_corners)
    gN = g0 - (args.frames - 1) * np.array([tx, ty])
    allc = np.vstack([g0, gN])
    margin = 40.0
    X0, Y0 = allc[:, 0].min() - margin, allc[:, 1].min() - margin
    X1, Y1 = allc[:, 0].max() + margin, allc[:, 1].max() + margin
    cw, ch = int(math.ceil((X1 - X0) / CANVAS_GSD)), int(math.ceil((Y1 - Y0) / CANVAS_GSD))
    print(f"ground extent {X1 - X0:.0f} x {Y1 - Y0:.0f} m -> canvas {cw} x {ch} px "
          f"({cw * ch * 3 / 1e6:.0f} MB)", flush=True)
    canvas = build_canvas(X0, Y0, cw, ch, rng)
    S = np.array([[CANVAS_GSD, 0, X0], [0, CANVAS_GSD, Y0], [0, 0, 1]])  # canvas px -> ground m

    # frame ground area (shoelace) for the object budget
    x, y = g0[:, 0], g0[:, 1]
    frame_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    n_obj = int(round(args.density * (cw * ch * CANVAS_GSD ** 2) / frame_area))

    idx = json.loads((SPRITES_DIR / "index.json").read_text())
    by_class = {}
    for r in idx:
        by_class.setdefault(r["object_id"], []).append(r)
    objects, occupied = [], []
    for _ in range(n_obj):
        cls = OBJECT_CLASSES[rng.integers(len(OBJECT_CLASSES))]
        rec = by_class[cls][rng.integers(len(by_class[cls]))]
        sp = cv2.imread(str(SPRITES_DIR / rec["file"]), cv2.IMREAD_UNCHANGED)
        fp = footprint_for(rec, sp)
        c, s = math.cos(fp["phi"]), math.sin(fp["phi"])
        corners = np.array([[fp["cx"] + dx * fp["a"] / 2 * c - dy * fp["b"] / 2 * s,
                             fp["cy"] + dx * fp["a"] / 2 * s + dy * fp["b"] / 2 * c]
                            for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        gx1, gy1, gx2, gy2 = rec["gt_bbox"]
        gsd_here = local_gsd(G, (gx1 + gx2) / 2, (gy1 + gy2) / 2)
        scale = gsd_here / CANVAS_GSD * rng.uniform(0.92, 1.08)
        sp_t, corners_t = transform_sprite(sp, corners, scale, rng.uniform(0, 360), rng)
        h, w = sp_t.shape[:2]
        for _try in range(30):
            ox, oy = int(rng.integers(0, cw - w)), int(rng.integers(0, ch - h))
            box = [ox + corners_t[:, 0].min() - 10, oy + corners_t[:, 1].min() - 10,
                   ox + corners_t[:, 0].max() + 10, oy + corners_t[:, 1].max() + 10]
            if all(box[2] <= o[0] or o[2] <= box[0] or box[3] <= o[1] or o[3] <= box[1] for o in occupied):
                composite(canvas, sp_t, ox, oy)
                occupied.append(box)
                objects.append((cls, corners_t + [ox, oy]))
                break

    totals = Counter(c for c, _ in objects)
    per_frame = []
    for t in range(args.frames):
        M = G @ translation(t * tx, t * ty) @ S         # canvas px -> frame t image px
        frame = cv2.warpPerspective(canvas, M, (FW, FH), flags=cv2.INTER_LINEAR)
        anns = []
        for oid, (cls, corners) in enumerate(objects):
            ph = np.hstack([corners, np.ones((4, 1))]) @ M.T
            if (ph[:, 2] <= 1e-6).any():
                continue                                  # behind the camera / across the horizon
            p = ph[:, :2] / ph[:, 2:3]
            if p[:, 0].max() - p[:, 0].min() > 600 or p[:, 1].max() - p[:, 1].min() > 600:
                continue
            x1, y1 = max(0.0, p[:, 0].min()), max(0.0, p[:, 1].min())
            x2, y2 = min(float(FW), p[:, 0].max()), min(float(FH), p[:, 1].max())
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                anns.append({"object_id": cls, "bbox": [int(round(x1)), int(round(y1)),
                                                       int(round(x2)), int(round(y2))],
                             "oid": oid})
        per_frame.append(len(anns))
        cv2.imwrite(str(out / "images" / f"frame_{t:06d}.png"), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        (out / "annotations" / f"frame_{t:06d}.json").write_text(json.dumps({
            "frame": t, "pose": {"x": -t * tx, "y": -t * ty, "z": -CAM["height_m"]},
            "annotations": anns}, indent=1))
        if t == 0:
            rev = frame.copy()
            for a in anns:
                x1, y1, x2, y2 = a["bbox"]
                cv2.rectangle(rev, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(rev, a["object_id"], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imwrite(str(out / "review_frame0.jpg"), cv2.resize(rev, (FW // 2, FH // 2), interpolation=cv2.INTER_AREA))
        if (t + 1) % 25 == 0:
            print(f"frame {t + 1}/{args.frames}", flush=True)

    (out / "run_metadata.json").write_text(json.dumps({
        "capture": {"camera_name": "SynthCamera", "altitude_m": CAM["height_m"], "num_frames": args.frames,
                    "step_m": float(math.hypot(tx, ty))},
        "total_objects": len(objects), "object_totals": dict(sorted(totals.items()))}, indent=2))
    print(f"objects placed {len(objects)}/{n_obj}; per-frame visible mean {np.mean(per_frame):.1f} "
          f"min {min(per_frame)} max {max(per_frame)}")
    print("class totals:", dict(sorted(totals.items())))


if __name__ == "__main__":
    main()
