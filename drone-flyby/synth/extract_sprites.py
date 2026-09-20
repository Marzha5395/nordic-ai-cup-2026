"""Extract RGBA object sprites from Helsinki frames.

Candidate masks in order: SAM box prompt, SAM box+center point, grabCut,
sharpness (Laplacian variance). Each candidate is validated; the first valid
one wins. Accepted sprites get flat-background colour suppression, then a
per-class area-consistency filter drops outliers.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "src/helsinki/images"
ANN_DIR = ROOT / "src/helsinki/annotations"
OUT_DIR = ROOT / "data/sprites"
FRAME_W, FRAME_H = 3840, 2160
PAD = 3
AREA_LO, AREA_HI = 0.04, 0.75
BORDER_FG_MAX = 0.35


def load_sam():
    for name in ("sam2.1_b.pt", "sam_b.pt", "mobile_sam.pt"):
        p = ROOT / "weights" / name
        if p.exists():
            try:
                return SAM(str(p)), name
            except Exception as e:
                print(f"SAM load failed for {name}: {e}")
    raise RuntimeError("no SAM weights available")


def clip_rect(x1, y1, x2, y2, w, h):
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))


def components(mask, box_local):
    """Clip to GT box + PAD, close 3x3, keep largest comp + comps >5% of it."""
    h, w = mask.shape
    bx1, by1, bx2, by2 = box_local
    clip = np.zeros((h, w), bool)
    cx1, cy1, cx2, cy2 = clip_rect(bx1 - PAD, by1 - PAD, bx2 + PAD, by2 + PAD, w, h)
    clip[cy1:cy2, cx1:cx2] = True
    m = (mask.astype(bool) & clip).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    keep = np.zeros_like(m)
    keep[labels == biggest] = 255
    thr = 0.05 * areas.max()
    for i in range(1, n):
        if i != biggest and stats[i, cv2.CC_STAT_AREA] > thr:
            keep[labels == i] = 255
    return keep


def is_valid(mask255, box_local, bw, bh):
    """mask255 is already component-filtered. Area + border tests."""
    area = int(mask255.sum() // 255)
    gt_area = bw * bh
    if not (AREA_LO * gt_area <= area <= AREA_HI * gt_area):
        return False
    h, w = mask255.shape
    bx1, by1, bx2, by2 = box_local
    ex1, ey1, ex2, ey2 = clip_rect(bx1 - PAD, by1 - PAD, bx2 + PAD, by2 + PAD, w, h)
    if ex2 - ex1 < 3 or ey2 - ey1 < 3:
        return False
    border = np.zeros((h, w), bool)
    border[ey1, ex1:ex2] = border[ey2 - 1, ex1:ex2] = True
    border[ey1:ey2, ex1] = border[ey1:ey2, ex2 - 1] = True
    nb = int(border.sum())
    if nb == 0:
        return False
    if (mask255[border] > 0).sum() / nb >= BORDER_FG_MAX:
        return False
    return True


def sam_mask(sam, crop, box_local, with_point):
    bx1, by1, bx2, by2 = box_local
    kw = dict(bboxes=[box_local], verbose=False)
    if with_point:
        kw["points"] = [[(bx1 + bx2) / 2, (by1 + by2) / 2]]
        kw["labels"] = [1]
    res = sam(crop, device="cpu", **kw)
    if res[0].masks is None or not len(res[0].masks.data):
        return None
    m = res[0].masks.data[0].cpu().numpy()
    if m.shape != crop.shape[:2]:
        m = cv2.resize(m.astype(np.float32), (crop.shape[1], crop.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def grabcut_mask(crop, box_local):
    h, w = crop.shape[:2]
    bx1, by1, bx2, by2 = box_local
    rect = (max(0, bx1), max(0, by1), max(1, bx2 - bx1), max(1, by2 - by1))
    m = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, m, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), bool)
    return (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)


def sharpness_mask(crop, box_local):
    """High local Laplacian variance inside the box vs a ring outside it."""
    h, w = crop.shape[:2]
    bx1, by1, bx2, by2 = box_local
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    mean = cv2.blur(lap, (5, 5))
    var = cv2.blur(lap * lap, (5, 5)) - mean * mean
    box = np.zeros((h, w), np.uint8)
    box[by1:by2, bx1:bx2] = 255
    outer = cv2.dilate(box, np.ones((13, 13), np.uint8))  # ~6px ring
    ring = (outer > 0) & (box == 0)
    if ring.sum() < 20:
        return np.zeros((h, w), bool)
    thr = 2.0 * float(np.median(var[ring]))
    fg = (var > thr) & (box > 0)
    m = fg.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 6:
            out[labels == i] = 255
    return cv2.dilate(out, np.ones((3, 3), np.uint8)) > 0


def suppress_flat_bg(alpha, crop, tight_local):
    """If the ring around the tight bbox is a uniform colour, fade matching px."""
    h, w = alpha.shape
    tx1, ty1, tx2, ty2 = tight_local
    tight = np.zeros((h, w), np.uint8)
    tight[ty1:ty2, tx1:tx2] = 255
    outer = cv2.dilate(tight, np.ones((13, 13), np.uint8))  # 6px out
    inner = cv2.dilate(tight, np.ones((5, 5), np.uint8))   # 2px out
    ring = (outer > 0) & (inner == 0) & (alpha == 0)
    if ring.sum() < 30:
        return alpha
    px = crop[ring].astype(np.float32)
    per_ch_std = px.std(axis=0)
    if per_ch_std.mean() >= 18:
        return alpha
    m = px.mean(axis=0)
    s = np.maximum(px.std(axis=0), 6.0)
    # Only touch a 2 px band just inside the mask edge, and only when the
    # object's interior clearly contrasts with the ring colour. Camouflaged
    # objects (tank on grass) must keep their body pixels.
    fg = alpha > 0
    interior = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band = fg & ~interior
    if interior.sum() < 10:
        return alpha
    obj = crop[interior].astype(np.float32)
    contrast = np.sqrt((((obj.mean(axis=0) - m) / s) ** 2).sum())
    if contrast < 3.0:
        return alpha
    d = np.sqrt((((crop.astype(np.float32) - m) / s) ** 2).sum(axis=2))
    gain = np.clip((d - 1.5) / 1.5, 0, 1)
    gain[~band] = 1.0
    return (alpha.astype(np.float32) * gain).astype(np.uint8)


def main():
    sam, sam_name = load_sam()
    print(f"SAM model: {sam_name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0
    dbg = defaultdict(list)  # class -> list of (title, crop, {method: mask})
    frames = sorted(IMG_DIR.glob("frame_*.png"))
    for fidx, img_path in enumerate(frames):
        ann_path = ANN_DIR / (img_path.stem + ".json")
        if not ann_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        anns = json.loads(ann_path.read_text())["annotations"]
        for ai, a in enumerate(anns):
            oid = a["object_id"]
            x1, y1, x2, y2 = [int(round(v)) for v in a["bbox"]]
            if not (x1 >= 2 and y1 >= 2 and x2 <= FRAME_W - 3 and y2 <= FRAME_H - 3):
                skipped += 1
                continue
            bw, bh = x2 - x1, y2 - y1
            cx1, cy1, cx2, cy2 = clip_rect(x1 - bw, y1 - bh, x2 + bw, y2 + bh,
                                          FRAME_W, FRAME_H)
            crop = frame[cy1:cy2, cx1:cx2]
            box_local = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]

            cands = {}
            order = []
            for name, fn in (("sam_box", lambda: sam_mask(sam, crop, box_local, False)),
                             ("sam_box_pt", lambda: sam_mask(sam, crop, box_local, True)),
                             ("grabcut", lambda: grabcut_mask(crop, box_local)),
                             ("sharp", lambda: sharpness_mask(crop, box_local))):
                order.append(name)
                try:
                    raw = fn()
                except Exception as e:
                    print(f"  {name} error f{fidx} {oid}: {e}")
                    raw = None
                if raw is None:
                    cands[name] = None
                    continue
                m = components(raw, box_local)
                cands[name] = m
                if is_valid(m, box_local, bw, bh):
                    mask, method = m, name
                    break
            else:
                mask, method = None, None

            if oid in ("helicopter", "medium_launcher"):
                dbg[oid].append((f"f{fidx} ann{ai}", crop, dict(cands), box_local))

            if mask is None:
                print(f"  no valid mask: {oid} f{fidx} gt={bw}x{bh}")
                continue

            # sprite window = GT box + PAD clamped
            sx1, sy1, sx2, sy2 = clip_rect(x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD,
                                          FRAME_W, FRAME_H)
            alpha = mask[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1].copy()
            sub = crop[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1]

            ys, xs = np.where(alpha > 0)
            if not len(xs):
                continue
            tight_l = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            alpha = suppress_flat_bg(alpha, sub, tight_l)
            alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8))
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

            rgb = frame[sy1:sy2, sx1:sx2]
            sprite = np.dstack([rgb, alpha])
            od = OUT_DIR / oid
            od.mkdir(exist_ok=True)
            fn = f"{oid}_f{fidx:02d}.png"
            cv2.imwrite(str(od / fn), sprite)

            ys, xs = np.where(alpha > 127)
            if len(xs):
                tight = [sx1 + int(xs.min()), sy1 + int(ys.min()),
                         sx1 + int(xs.max()) + 1, sy1 + int(ys.max()) + 1]
            else:
                tight = [x1, y1, x2, y2]
            records.append(dict(
                object_id=oid, frame=fidx, file=f"{oid}/{fn}",
                gt_bbox=[x1, y1, x2, y2], sprite_origin=[sx1, sy1],
                tight_bbox=tight, mask_area_px=int((alpha > 127).sum()),
                method=method,
                pad=[tight[0] - x1, tight[1] - y1, x2 - tight[2], y2 - tight[3]],
            ))
        print(f"frame {fidx}: done ({len(records)} sprites so far)")

    # ---- per-class consistency filter ----
    by_class = defaultdict(list)
    for r in records:
        by_class[r["object_id"]].append(r)
    dropped = []
    for c, recs in by_class.items():
        if len(recs) < 4:
            continue
        ratios = np.array([r["mask_area_px"] /
                           ((r["gt_bbox"][2] - r["gt_bbox"][0]) *
                            (r["gt_bbox"][3] - r["gt_bbox"][1])) for r in recs])
        med = np.median(ratios)
        for r, rt in zip(recs, ratios):
            if abs(rt - med) / med > 0.45:
                dropped.append((r["file"], f"area_ratio={rt:.2f} med={med:.2f}"))
                p = OUT_DIR / r["file"]
                if p.exists():
                    p.unlink()
                records.remove(r)
    for f, why in dropped:
        print(f"DROPPED {f}: {why}")

    (OUT_DIR / "index.json").write_text(json.dumps(records, indent=1))
    print(f"total sprites={len(records)} skipped_border={skipped} dropped={len(dropped)}")

    print("\nper-class counts and winning methods:")
    for c in sorted(by_class):
        recs = [r for r in records if r["object_id"] == c]
        meths = defaultdict(int)
        for r in recs:
            meths[r["method"]] += 1
        print(f"  {c:<16} n={len(recs):<3} {dict(meths)}")

    # ---- padding table ----
    print(f"\n{'class':<16} n  medGT_w medGT_h medTight_w medTight_h")
    for c in sorted(by_class):
        v = np.array([[r["gt_bbox"][2] - r["gt_bbox"][0],
                       r["gt_bbox"][3] - r["gt_bbox"][1],
                       r["tight_bbox"][2] - r["tight_bbox"][0],
                       r["tight_bbox"][3] - r["tight_bbox"][1]]
                      for r in records if r["object_id"] == c])
        if not len(v):
            continue
        m = np.median(v, axis=0)
        print(f"{c:<16} {len(v):<3} {m[0]:7.1f} {m[1]:7.1f} {m[2]:9.1f} {m[3]:9.1f}")

    # ---- debug sheets for classes with <3 sprites ----
    counts = {c: len([r for r in records if r["object_id"] == c]) for c in by_class}
    for c, items in dbg.items():
        if counts.get(c, 0) >= 3:
            continue
        rows = []
        meths = order
        for title, crop, cands, _bl in items:
            panels = [crop]
            for mname in meths:
                m = cands.get(mname)
                if m is None:
                    p = np.zeros_like(crop)
                    cv2.putText(p, "none", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                else:
                    p = crop.copy()
                    p[m > 0] = (0.4 * p[m > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
                cv2.putText(p, mname, (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 255), 1)
                panels.append(p)
            hm = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, hm - p.shape[0], 0, 0,
                                         cv2.BORDER_CONSTANT, value=(30, 30, 30))
                      for p in panels]
            row = np.hstack(panels)
            cv2.putText(row, title, (4, hm - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)
            rows.append(row)
        W = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
                for r in rows]
        cv2.imwrite(str(OUT_DIR / f"debug_{c}.png"), np.vstack(rows))
        print(f"wrote debug sheet {OUT_DIR/f'debug_{c}.png'}")

    # ---- sheet_masks: largest instance per class ----
    tiles = []
    for c in sorted(by_class):
        recs = [r for r in records if r["object_id"] == c]
        if not recs:
            continue
        r = max(recs, key=lambda r: (r["gt_bbox"][2] - r["gt_bbox"][0]) *
                (r["gt_bbox"][3] - r["gt_bbox"][1]))
        fr = cv2.imread(str(IMG_DIR / f"frame_{r['frame']:06d}.png"))
        x1, y1, x2, y2 = r["gt_bbox"]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1, cx2, cy2 = clip_rect(x1 - bw, y1 - bh, x2 + bw, y2 + bh,
                                      FRAME_W, FRAME_H)
        crop = fr[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1),
                      (0, 0, 255), 1)
        sp = cv2.imread(str(OUT_DIR / r["file"]), cv2.IMREAD_UNCHANGED)
        comp = np.full(sp.shape[:2] + (3,), 128, np.uint8)
        a = sp[:, :, 3:4].astype(np.float32) / 255
        comp = (sp[:, :, :3] * a + comp * (1 - a)).astype(np.uint8)
        tw, th = r["tight_bbox"][2] - r["tight_bbox"][0], r["tight_bbox"][3] - r["tight_bbox"][1]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        comp = cv2.resize(comp, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        hmax = max(crop.shape[0], comp.shape[0])
        def pad_h(im):
            if im.shape[0] < hmax:
                im = cv2.copyMakeBorder(im, 0, hmax - im.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(40, 40, 40))
            return im
        tile = np.hstack([pad_h(crop), np.full((hmax, 4, 3), 255, np.uint8), pad_h(comp)])
        cv2.putText(tile, f"{c} {r['method']} tight={tw}x{th} gt={bw}x{bh}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(tile)
    W = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, W - t.shape[1], cv2.BORDER_CONSTANT,
                                value=(20, 20, 20)) for t in tiles]
    cv2.imwrite(str(OUT_DIR / "sheet_masks.png"), np.vstack(tiles))

    # ---- sheet_all ----
    cell, cols = 96, 10
    rows_n = math.ceil(len(records) / cols)
    sheet = np.full((rows_n * cell, cols * cell, 3), 128, np.uint8)
    for i, r in enumerate(records):
        sp = cv2.imread(str(OUT_DIR / r["file"]), cv2.IMREAD_UNCHANGED)
        h, w = sp.shape[:2]
        s = min((cell - 18) / w, (cell - 18) / h)
        sp = cv2.resize(sp, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA)
        comp = np.full(sp.shape[:2] + (3,), 128, np.uint8)
        a = sp[:, :, 3:4].astype(np.float32) / 255
        comp = (sp[:, :, :3] * a + comp * (1 - a)).astype(np.uint8)
        ry, cx = divmod(i, cols)
        oy, ox = ry * cell + 16, cx * cell + (cell - comp.shape[1]) // 2
        sheet[oy:oy + comp.shape[0], ox:ox + comp.shape[1]] = comp
        cv2.putText(sheet, f"{r['object_id'][:12]} f{r['frame']:02d}",
                    (cx * cell + 2, ry * cell + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 0, 0), 1)
    cv2.imwrite(str(OUT_DIR / "sheet_all.png"), sheet)
    print(f"\nwrote {OUT_DIR/'sheet_masks.png'} and {OUT_DIR/'sheet_all.png'}")


if __name__ == "__main__":
    main()
