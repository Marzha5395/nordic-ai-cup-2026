"""Generate synthetic YOLO detection dataset: composite extracted sprites onto
Helsinki frames / Inria tiles, render L0/L1/L2 views, emit YOLO labels."""
import argparse
import json
import math
import resource
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES  # noqa: E402

IMG_DIR = ROOT / "src/helsinki/images"
ANN_DIR = ROOT / "src/helsinki/annotations"
INRIA_DIR = ROOT / "data/backgrounds/inria"
SPRITES_DIR = ROOT / "data/sprites"
CW, CH = 3840, 2160          # canvas
VW, VH = 960, 540            # view

# ---------------- per-worker globals ----------------
G = {}


def worker_init(seed, out_dir):
    G["seed"] = seed
    G["out"] = Path(out_dir)
    idx = json.loads((SPRITES_DIR / "index.json").read_text())
    sprites = {}
    for r in idx:
        sprites.setdefault(r["object_id"], []).append(r)
    G["sprites"] = sprites
    G["classes"] = sorted(sprites)
    G["cls_idx"] = {c: OBJECT_CLASSES.index(c) for c in OBJECT_CLASSES}
    G["helsinki"] = sorted(IMG_DIR.glob("frame_*.png"))
    G["inria"] = sorted(p for p in INRIA_DIR.glob("*.tif")
                        if not p.stem.endswith("21"))
    G["cache"] = {}
    G["footprints"] = {}


def load_img_cached(path):
    key = str(path)
    c = G["cache"]
    if key not in c:
        img = cv2.imread(key, cv2.IMREAD_COLOR)
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            img = cv2.cvtColor(
                cv2.convertScaleAbs(img), cv2.COLOR_BGRA2BGR
                if img.ndim == 3 and img.shape[2] == 4 else cv2.COLOR_GRAY2BGR)
        c[key] = img
        if len(c) > 2:
            c.pop(next(iter(c)))
    return c[key]


def photometric(img, rng):
    b = rng.uniform(0.85, 1.15)
    c = rng.uniform(0.85, 1.15)
    lut = np.clip(((np.arange(256, dtype=np.float32) - 128.0) * c + 128.0) * b,
                  0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def jitter_sprite_rgb(sp, rng):
    """Photometric jitter on RGB where alpha>0; optional blur on RGB."""
    rgb = sp[:, :, :3].astype(np.float32)
    b, c = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    rgb = (rgb - 128.0) * c + 128.0
    rgb = rgb * b + rng.uniform(-8, 8, 3)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        sigma = rng.uniform(0.3, 0.8)
        k = max(3, int(math.ceil(sigma * 4)) | 1)
        rgb = cv2.GaussianBlur(rgb, (k, k), sigma)
    return np.dstack([rgb, sp[:, :, 3]])


def footprint_for(rec, sp):
    """Oriented footprint rect (centre = GT box centre, axis = mask principal
    axis) whose AABB equals the Helsinki GT box at zero rotation."""
    gx1, gy1, gx2, gy2 = rec["gt_bbox"]
    ox0, oy0 = rec["sprite_origin"]
    gt = [gx1 - ox0, gy1 - oy0, gx2 - ox0, gy2 - oy0]
    W, H = gt[2] - gt[0], gt[3] - gt[1]
    fp = {"cx": (gt[0] + gt[2]) / 2.0, "cy": (gt[1] + gt[3]) / 2.0,
          "a": float(W), "b": float(H), "phi": 0.0, "fallback": True}
    ys, xs = np.where(sp[:, :, 3] > 127)
    if len(xs) < 12:
        return fp
    dx, dy = xs - xs.mean(), ys - ys.mean()
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)],
                    [np.mean(dx * dy), np.mean(dy * dy)]])
    vals, vecs = np.linalg.eigh(cov)
    aniso = math.sqrt(vals[1] / max(vals[0], 1e-6))
    if aniso < 1.25:
        return fp
    phi = math.atan2(vecs[1, 1], vecs[0, 1]) % math.pi
    c, s = abs(math.cos(phi)), abs(math.sin(phi))
    det = c * c - s * s
    if abs(det) < 0.30:
        return fp
    a = (W * c - H * s) / det
    b = (H * c - W * s) / det
    if a <= 2 or b <= 2 or a / b > 6 or b / a > 6:
        return fp
    fp.update(a=a, b=b, phi=phi, fallback=False)
    return fp


def footprint_corners(fp):
    ux, uy = math.cos(fp["phi"]), math.sin(fp["phi"])
    vx, vy = -uy, ux
    c = np.array([fp["cx"], fp["cy"]])
    u, v = np.array([ux, uy]), np.array([vx, vy])
    return np.array([c + sa * fp["a"] / 2 * u + sb * fp["b"] / 2 * v
                     for sa in (-1, 1) for sb in (-1, 1)])


def transform_sprite(sp, corners, rng):
    """flip -> jitter -> scale -> rotate. Returns (rgba, label) where label
    is the AABB of the transformed footprint corners."""
    corners = corners.copy()
    if rng.random() < 0.5:
        sp = cv2.flip(sp, 1)
        corners[:, 0] = sp.shape[1] - corners[:, 0]
    sp = jitter_sprite_rgb(sp, rng)
    s = rng.uniform(0.7, 1.4)
    h, w = sp.shape[:2]
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    sp = cv2.resize(sp, (sw, sh), interpolation=cv2.INTER_LINEAR)
    corners = corners * s
    theta = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), theta, 1.0)
    corners_px = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64)
    rc = corners_px @ M[:, :2].T + M[:, 2]
    mn, mx = rc.min(0), rc.max(0)
    M[0, 2] -= mn[0]
    M[1, 2] -= mn[1]
    rw, rh = int(math.ceil(mx[0] - mn[0])), int(math.ceil(mx[1] - mn[1]))
    sp = cv2.warpAffine(sp, M, (rw, rh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tc = corners @ M[:, :2].T + M[:, 2]
    label = [tc[:, 0].min(), tc[:, 1].min(), tc[:, 0].max(), tc[:, 1].max()]
    return sp, label


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def inside(c, b):
    return b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]


def composite(canvas, sp, cx, cy):
    """Alpha-composite sp centred at (cx,cy). Returns paste offset (ox,oy)."""
    h, w = sp.shape[:2]
    ox, oy = int(round(cx - w / 2)), int(round(cy - h / 2))
    x1, y1 = max(0, ox), max(0, oy)
    x2, y2 = min(CW, ox + w), min(CH, oy + h)
    if x2 <= x1 or y2 <= y1:
        return ox, oy
    a = sp[y1 - oy:y2 - oy, x1 - ox:x2 - ox, 3:4].astype(np.float32) / 255.0
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (a * sp[y1 - oy:y2 - oy, x1 - ox:x2 - ox, :3]
                            + (1 - a) * roi).astype(np.uint8)
    return ox, oy


def render_view(canvas, boxes, level, rng, obj_centres):
    """boxes: list of (cls, x1,y1,x2,y2) in canvas px. Returns (img, labels)."""
    if level == 0:
        img = cv2.resize(canvas, (VW, VH), interpolation=cv2.INTER_AREA)
        t = lambda x1, y1, x2, y2: (x1 / 4, y1 / 4, x2 / 4, y2 / 4)  # noqa: E731
    elif level == 1:
        if obj_centres and rng.random() < 0.7:
            oc = obj_centres[rng.integers(len(obj_centres))]
            cx = np.clip(oc[0] + rng.uniform(-600, 600), 960, 2880)
            cy = np.clip(oc[1] + rng.uniform(-350, 350), 540, 1620)
        else:
            cx, cy = rng.uniform(960, 2880), rng.uniform(540, 1620)
        x1, y1 = int(round(cx - 960)), int(round(cy - 540))
        img = cv2.resize(canvas[y1:y1 + 1080, x1:x1 + 1920], (VW, VH),
                         interpolation=cv2.INTER_AREA)
        t = lambda a, b, c, d: ((a - x1) / 2, (b - y1) / 2,  # noqa: E731
                                (c - x1) / 2, (d - y1) / 2)
    else:
        if obj_centres and rng.random() < 0.7:
            oc = obj_centres[rng.integers(len(obj_centres))]
            cx = np.clip(oc[0] + rng.uniform(-300, 300), 480, 3360)
            cy = np.clip(oc[1] + rng.uniform(-170, 170), 270, 1890)
        else:
            cx, cy = rng.uniform(480, 3360), rng.uniform(270, 1890)
        x1, y1 = int(round(cx - 480)), int(round(cy - 270))
        img = canvas[y1:y1 + 540, x1:x1 + 960]
        t = lambda a, b, c, d: (a - x1, b - y1, c - x1, d - y1)  # noqa: E731
    labels = []
    for cls, bx1, by1, bx2, by2 in boxes:
        area0 = max(0.0, (bx2 - bx1) * (by2 - by1))
        vx1, vy1, vx2, vy2 = t(bx1, by1, bx2, by2)
        vx1, vy1 = max(0.0, vx1), max(0.0, vy1)
        vx2, vy2 = min(float(VW), vx2), min(float(VH), vy2)
        w, h = vx2 - vx1, vy2 - vy1
        if w < 2 or h < 2:
            continue
        scale = 4 if level == 0 else (2 if level == 1 else 1)
        if w * h * scale * scale < 0.4 * area0:
            continue
        labels.append((cls, vx1 + w / 2, vy1 + h / 2, w, h))
    return img, labels


def do_canvas(cid):
    rng = np.random.default_rng(G["seed"] * 1_000_003 + cid)
    helsinki_bg = rng.random() < 0.4
    gt_boxes = []  # canvas-space label boxes already present
    if helsinki_bg:
        fp = G["helsinki"][rng.integers(len(G["helsinki"]))]
        canvas = load_img_cached(fp).copy()
        ap = ANN_DIR / (fp.stem + ".json")
        for a in json.loads(ap.read_text())["annotations"]:
            x1, y1, x2, y2 = a["bbox"]
            gt_boxes.append((a["object_id"], float(x1), float(y1), float(x2), float(y2)))
        K = rng.integers(3, 21)
    else:
        tp = G["inria"][rng.integers(len(G["inria"]))]
        tile = load_img_cached(tp)
        f = rng.uniform(1.2, 1.7)
        th, tw = tile.shape[:2]
        ww, wh = min(tw, int(round(CW / f))), min(th, int(round(CH / f)))
        oy = rng.integers(0, th - wh + 1)
        ox = rng.integers(0, tw - ww + 1)
        canvas = cv2.resize(tile[oy:oy + wh, ox:ox + ww], (CW, CH),
                            interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            canvas = cv2.flip(canvas, 1)
        if rng.random() < 0.5:
            canvas = cv2.flip(canvas, 0)
        K = rng.integers(6, 26)
    canvas = photometric(canvas, rng)

    boxes = list(gt_boxes)
    obj_centres = []
    occupied = [b[1:] for b in gt_boxes]
    for _ in range(K):
        cls = G["classes"][rng.integers(len(G["classes"]))]
        rec = G["sprites"][cls][rng.integers(len(G["sprites"][cls]))]
        sp = cv2.imread(str(SPRITES_DIR / rec["file"]), cv2.IMREAD_UNCHANGED)
        fp = G["footprints"].get(rec["file"])
        if fp is None:
            fp = footprint_for(rec, sp)
            G["footprints"][rec["file"]] = fp
        sp, label = transform_sprite(sp, footprint_corners(fp), rng)
        placed = False
        for _try in range(10):
            ccx, ccy = rng.uniform(0, CW), rng.uniform(0, CH)
            sh, sw = sp.shape[:2]
            ox, oy = ccx - sw / 2, ccy - sh / 2
            lb = [label[0] + ox, label[1] + oy, label[2] + ox, label[3] + oy]
            # clip to canvas, area test
            cx1, cy1 = max(0.0, lb[0]), max(0.0, lb[1])
            cx2, cy2 = min(float(CW), lb[2]), min(float(CH), lb[3])
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.3 * (lb[2] - lb[0]) * (lb[3] - lb[1]):
                continue
            ctr = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
            bad = any(iou(lb, ob) > 0.05 or inside(ctr, ob) for ob in occupied)
            if bad:
                continue
            composite(canvas, sp, ccx, ccy)
            boxes.append((cls, cx1, cy1, cx2, cy2))
            occupied.append([cx1, cy1, cx2, cy2])
            obj_centres.append(ctr)
            placed = True
            break
        if not placed:
            continue

    split = "val" if cid < G["nval"] else "train"
    out_rows = []
    for level in (0, 1, 2):
        img, labels = render_view(canvas, boxes, level, rng, obj_centres)
        name = f"c{cid:05d}_L{level}"
        cv2.imwrite(str(G["out"] / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for cls, cx, cy, w, h in labels:
            lines.append(f"{G['cls_idx'][cls]} {cx / VW:.6f} {cy / VH:.6f} "
                         f"{w / VW:.6f} {h / VH:.6f}")
            out_rows.append((level, cls, w, h))
        (G["out"] / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
    return out_rows


def check_footprints():
    worker_init(0, ROOT)
    recs = [r for rs in G["sprites"].values() for r in rs]
    stats = {}
    bad = 0
    t45 = math.radians(45)
    for rec in recs:
        sp = cv2.imread(str(SPRITES_DIR / rec["file"]), cv2.IMREAD_UNCHANGED)
        fp = footprint_for(rec, sp)
        corners = footprint_corners(fp)
        gx1, gy1, gx2, gy2 = rec["gt_bbox"]
        ox0, oy0 = rec["sprite_origin"]
        gt = [gx1 - ox0, gy1 - oy0, gx2 - ox0, gy2 - oy0]
        aabb = [corners[:, 0].min(), corners[:, 1].min(),
                corners[:, 0].max(), corners[:, 1].max()]
        err = max(abs(p - g) for p, g in zip(aabb, gt))
        if err > 0.5:
            print(f"VIOLATION {rec['file']}: aabb={aabb} gt={gt} "
                  f"err={err:.2f}px")
            bad += 1
        a, b, phi = fp["a"], fp["b"], fp["phi"]
        w45 = a * abs(math.cos(phi - t45)) + b * abs(math.sin(phi - t45))
        h45 = a * abs(math.sin(phi - t45)) + b * abs(math.cos(phi - t45))
        r_fp = w45 * h45 / (a * b)
        W, H = gt[2] - gt[0], gt[3] - gt[1]
        wA = W * math.cos(t45) + H * math.sin(t45)
        r_A = wA * wA / (W * H)
        st = stats.setdefault(rec["object_id"],
                              {"n": 0, "fb": 0, "a": [], "b": [],
                               "phi": [], "rfp": [], "rA": []})
        st["n"] += 1
        st["fb"] += fp["fallback"]
        st["a"].append(a)
        st["b"].append(b)
        if not fp["fallback"]:
            st["phi"].append(math.degrees(phi))
        st["rfp"].append(r_fp)
        st["rA"].append(r_A)
    print(f"{len(recs)} records, {bad} AABB violations (>0.5 px)")
    print(f"{'class':<17} {'n':>4} {'fb':>4} {'med a x b':>12} "
          f"{'phi_deg':>7} {'fp45':>6} {'A45':>6}")
    for cls in sorted(stats):
        st = stats[cls]
        med_phi = np.median(st["phi"]) if st["phi"] else 0.0
        print(f"{cls:<17} {st['n']:>4} {st['fb']:>4} "
              f"{np.median(st['a']):>5.0f}x{np.median(st['b']):<6.0f} "
              f"{med_phi:>7.1f} {np.median(st['rfp']):>6.2f} "
              f"{np.median(st['rA']):>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvases", type=int, default=2500)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-footprints", action="store_true")
    args = ap.parse_args()

    if args.check_footprints:
        check_footprints()
        return

    out = ROOT / args.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    nval = int(args.canvases * args.val_frac)
    G["nval"] = nval  # set for initializer via global copy

    init_seed = args.seed
    def init():
        worker_init(init_seed, out)
        G["nval"] = nval

    rows = []
    with Pool(args.workers, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(do_canvas, range(args.canvases))):
            rows.extend(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.canvases} canvases", flush=True)

    (out / "data.yaml").write_text(
        f"path: {out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(OBJECT_CLASSES)))

    # stats
    per_level = Counter()
    per_class = Counter()
    sizes = {0: [], 1: [], 2: []}
    for level, cls, w, h in rows:
        per_level[level] += 1
        per_class[cls] += 1
        sizes[level].append((w, h))
    print("\nlabel stats:")
    for lv in (0, 1, 2):
        ws = [s[0] for s in sizes[lv]] or [0]
        hs = [s[1] for s in sizes[lv]] or [0]
        print(f"  L{lv}: boxes={per_level[lv]} "
              f"w min/med/max={min(ws):.0f}/{np.median(ws):.0f}/{max(ws):.0f} "
              f"h {min(hs):.0f}/{np.median(hs):.0f}/{max(hs):.0f}")
    print("  boxes per class:", dict(sorted(per_class.items())))

    # review sheet: 12 random train images, 4 per level
    rng = np.random.default_rng(args.seed + 7)
    train_imgs = sorted((out / "images/train").glob("*.jpg"))
    picks = {0: [], 1: [], 2: []}
    for lv in (0, 1, 2):
        cands = [p for p in train_imgs if p.stem.endswith(f"_L{lv}")]
        picks[lv] = [cands[i] for i in
                     rng.choice(len(cands), size=min(4, len(cands)), replace=False)]
    tiles = []
    for lv in (0, 1, 2):
        for p in picks[lv]:
            im = cv2.imread(str(p))
            lt = out / "labels/train" / (p.stem + ".txt")
            for line in lt.read_text().splitlines():
                ci, cx, cy, w, h = line.split()
                ci = int(ci)
                cx, cy, w, h = float(cx) * VW, float(cy) * VH, float(w) * VW, float(h) * VH
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(im, OBJECT_CLASSES[ci], (x1, max(8, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            tiles.append(im)
    sheet = np.vstack([np.hstack(tiles[i * 4:(i + 1) * 4]) for i in range(3)])
    cv2.imwrite(str(out / "review_labels.png"), sheet)
    print(f"wrote {out/'review_labels.png'}")

    # review_zoom: 8 random train labels from L1/L2 of camouflage-prone classes
    zoom_cls = {"tank", "small_plane", "helicopter", "ta-ta",
                "medium_launcher", "jet_plane"}
    cand = []
    for lt in sorted((out / "labels/train").glob("*.txt")):
        lv = lt.stem[-1]
        if lv not in "12":
            continue
        for li, line in enumerate(lt.read_text().splitlines()):
            ci = int(line.split()[0])
            if OBJECT_CLASSES[ci] in zoom_cls:
                cand.append((lt, li, line))
    rngz = np.random.default_rng(args.seed + 11)
    tiles = []
    for lt, li, line in [cand[i] for i in
                         rngz.choice(len(cand), size=min(8, len(cand)), replace=False)]:
        ci, cx, cy, w, h = line.split()
        ci = int(ci)
        cx, cy, w, h = float(cx) * VW, float(cy) * VH, float(w) * VW, float(h) * VH
        im = cv2.imread(str(out / "images/train" / (lt.stem + ".jpg")))
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        m = 40
        cc = im[max(0, y1 - m):min(VH, y2 + m), max(0, x1 - m):min(VW, x2 + m)].copy()
        z = cv2.resize(cc, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, ((x1 - max(0, x1 - m)) * 4, (y1 - max(0, y1 - m)) * 4),
                      ((x2 - max(0, x1 - m)) * 4, (y2 - max(0, y1 - m)) * 4),
                      (0, 0, 255), 2)
        cv2.putText(z, f"{OBJECT_CLASSES[ci]} {lt.stem}#{li}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(z)
    hmax = max(t.shape[0] for t in tiles)
    wmax = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, wmax - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(30, 30, 30))
             for t in tiles]
    zsheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out / "review_zoom.png"), zsheet)
    print(f"wrote {out/'review_zoom.png'}")

    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"peak RSS: self={self_rss:.0f} MB, max child={child_rss:.0f} MB")


if __name__ == "__main__":
    main()
