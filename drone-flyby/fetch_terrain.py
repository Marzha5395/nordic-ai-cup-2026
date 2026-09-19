"""Download object-free aerial terrain for training and for held-out testing.

The 25 reference frames show one place. A detector that only ever sees that
place learns it, and on new ground it reports rocks, sheds and shadows as
objects. Real orthophotos from several open services give it other ground to
learn "background" from, and a held-out set of places to measure false alarms
on.

The split is by place, never by neighbouring crops: every region is either
wholly ``train`` or wholly ``test``. Nothing here is used at inference time.

Sources (all open, no key):
  * PDOK luchtfoto (Netherlands), 8-25 cm
  * Maa- ja Ruumiamet fotokaart (Estonia), about 20-40 cm
  * USGS NAIP Plus (United States), 30-60 cm
"""

import argparse
import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

# name, source, split, x_min, x_max, y_min, y_max, samples
# PDOK boxes are in EPSG:28992 metres, Estonia in EPSG:3301 metres, NAIP in degrees.
REGIONS = [
    ('nl_utrecht', 'pdok', 'train', 125000, 150000, 440000, 465000, 18),
    ('nl_veluwe', 'pdok', 'train', 175000, 205000, 450000, 480000, 18),
    ('nl_groningen', 'pdok', 'train', 220000, 250000, 565000, 595000, 14),
    ('nl_zeeland', 'pdok', 'test', 40000, 70000, 380000, 405000, 10),
    ('nl_drenthe', 'pdok', 'test', 225000, 255000, 520000, 545000, 10),
    ('ee_harju', 'estonia', 'train', 520000, 580000, 6560000, 6590000, 22),
    ('ee_parnu', 'estonia', 'train', 520000, 560000, 6450000, 6490000, 18),
    ('ee_tartu', 'estonia', 'train', 640000, 680000, 6440000, 6480000, 18),
    ('ee_ida', 'estonia', 'test', 680000, 720000, 6560000, 6590000, 12),
    ('ee_voru', 'estonia', 'test', 660000, 700000, 6400000, 6420000, 10),
    ('us_sonoran', 'naip', 'train', -114.5, -111.0, 32.0, 34.5, 10),
    ('us_corn_belt', 'naip', 'train', -96.0, -90.0, 40.0, 43.0, 10),
    ('us_appalachia', 'naip', 'train', -84.0, -79.0, 35.0, 39.0, 10),
    ('us_rockies', 'naip', 'train', -108.5, -105.0, 37.5, 41.0, 10),
    ('us_urban_la', 'naip', 'train', -118.5, -117.0, 33.5, 34.5, 8),
    ('us_maine', 'naip', 'test', -70.5, -68.0, 44.0, 46.5, 8),
    ('us_texas', 'naip', 'test', -101.0, -97.0, 29.0, 33.0, 8),
    ('us_great_lakes', 'naip', 'test', -89.0, -84.0, 43.0, 46.5, 8),
    # Cities, ports and industry: the validation flight is urban (marina, rail yards, highways, construction),
    # and the first detectors raised false alarms on buildings and red roofs. Appended last, so the earlier
    # tiles and their recorded scales are unchanged.
    ('nl_rotterdam_port', 'pdok', 'train', 60000, 95000, 430000, 445000, 16),
    ('nl_amsterdam', 'pdok', 'train', 115000, 130000, 480000, 492000, 14),
    ('ee_tallinn', 'estonia', 'train', 535000, 552000, 6583000, 6595000, 14),
    ('nl_denhaag', 'pdok', 'test', 75000, 85000, 450000, 460000, 8),
    ('nl_eindhoven', 'pdok', 'test', 155000, 165000, 380000, 390000, 8),
]

NAIP = 'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage'
PDOK = 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0'
ESTONIA = 'https://kaart.maaamet.ee/wms/fotokaart'


def download(url, timeout=60):
    request = urllib.request.Request(url, headers={'User-Agent': 'nordic-ai-cup-terrain/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


def fetch(source, x, y, pixels, gsd):
    half = pixels * gsd / 2
    if source == 'pdok':
        layer = 'Actueel_orthoHR' if gsd < 0.2 else 'Actueel_ortho25'
        return download(f'{PDOK}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS=EPSG:28992'
                        f'&BBOX={x - half},{y - half},{x + half},{y + half}&WIDTH={pixels}&HEIGHT={pixels}&FORMAT=image/jpeg')
    if source == 'estonia':
        # The service stamps a small credit line along the bottom edge; fetch extra and cut it off.
        extra = 64
        image = download(f'{ESTONIA}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=EESTIFOTO&STYLES=&SRS=EPSG:3301'
                         f'&BBOX={x - half},{y - half - extra * gsd},{x + half},{y + half}&WIDTH={pixels}&HEIGHT={pixels + extra}&FORMAT=image/jpeg')
        return None if image is None else image[:pixels]
    radius = 6378137.0
    mx = math.radians(x) * radius
    my = math.log(math.tan(math.pi / 4 + math.radians(y) / 2)) * radius
    half_mercator = half / math.cos(math.radians(y))
    return download(f'{NAIP}?bbox={mx - half_mercator},{my - half_mercator},{mx + half_mercator},{my + half_mercator}'
                    f'&bboxSR=102100&imageSR=102100&size={pixels},{pixels}&format=jpg&f=image')


def usable(image):
    if image is None or image.size == 0:
        return False
    if float(image.std()) < 10:
        return False
    # White or black margins mean the request ran off the edge of the coverage.
    blank = float(((image.min(axis=2) > 245) | (image.max(axis=2) < 8)).mean())
    return blank < 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / '.training' / 'terrain')
    parser.add_argument('--pixels', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260919)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    jobs = []
    for name, source, split, x0, x1, y0, y1, samples in REGIONS:
        for index in range(samples):
            # Ground sample distance around the reference scene's ~0.2 m per source pixel.
            # NAIP is never sharper than 0.3-0.6 m, so it is not requested finer.
            low, high = (0.3, 0.6) if source == 'naip' else (0.12, 0.35)
            gsd = float(rng.uniform(low, high))
            jobs.append((split, f'{name}_{index:03d}', source, float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)), gsd))

    def run(job):
        split, name, source, x, y, gsd = job
        destination = args.output / split / f'{name}.jpg'
        if destination.exists():
            return split, True
        for attempt in range(3):
            try:
                image = fetch(source, x, y, args.pixels, gsd)
                break
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                image = None
                print(f'  {name} attempt {attempt + 1}: {error}', flush=True)
        if not usable(image):
            return split, False
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return split, True

    # Record every tile's ground sample distance, so it can be rescaled to the source frame's.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gsd.json').write_text(json.dumps({f'{split}/{name}.jpg': gsd for split, name, _, _, _, gsd in jobs}, indent=0))

    kept = {'train': 0, 'test': 0}
    with ThreadPoolExecutor(args.workers) as pool:
        for done, (split, ok) in enumerate(pool.map(run, jobs), 1):
            kept[split] += int(ok)
            if done % 25 == 0:
                print(f'  {done}/{len(jobs)} requested', flush=True)
    print(f'kept {kept["train"]} train and {kept["test"]} test tiles in {args.output}')


if __name__ == '__main__':
    main()
