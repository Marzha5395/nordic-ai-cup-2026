"""Download aerial terrain to train and to test against.

The reference scene is 25 frames of one place. A detector trained only on it
learns that place: on the validation flight it answered with 20 detections on
the first frames and 80 on the last, which is false tracks accumulating on
terrain it had never seen.

So: real aerial imagery, from USGS, which is public domain and served for
exactly this kind of use. The locations are spread across deserts, farmland,
forest, mountains, snow, coast, wetland and city, because what causes a false
alarm is a small sharp thing that is not an object -- a rock, a shed, a parked
car, a shadow -- and one terrain type only has its own.

Half the locations are held out. Nothing in ``test`` is ever trained on, which
makes ``training/eval_unseen.py`` the only honest number in this repository.
"""

import argparse
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

SERVICE = (
    'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/'
    'ImageServer/exportImage'
)

# Rough boxes over places whose ground cover differs, with a weight for how
# many samples to draw from each. NAIP covers the conterminous United States,
# which is enough variety on its own: Sonoran desert to Maine forest.
REGIONS = [
    # name,                 lat_min, lat_max, lon_min, lon_max, samples
    ('sonoran_desert',        32.0,  34.5, -114.5, -111.0, 22),
    ('great_basin',           38.5,  41.5, -119.0, -115.0, 18),
    ('high_plains_farm',      37.0,  41.0, -102.0,  -97.0, 22),
    ('corn_belt',             40.0,  43.0,  -96.0,  -90.0, 20),
    ('mississippi_delta',     30.0,  33.0,  -92.0,  -89.5, 14),
    ('appalachian_forest',    35.0,  39.0,  -84.0,  -79.0, 20),
    ('maine_forest',          44.0,  46.5,  -70.5,  -68.0, 14),
    ('rockies',               37.5,  41.0, -108.5, -105.0, 20),
    ('cascades',              44.0,  48.0, -122.5, -120.0, 16),
    ('sierra',                36.5,  39.5, -120.5, -118.5, 14),
    ('florida_coast',         25.5,  29.0,  -82.5,  -80.0, 14),
    ('carolina_coast',        33.0,  35.5,  -79.5,  -76.5, 12),
    ('texas_rangeland',       29.0,  33.0, -101.0,  -97.0, 18),
    ('california_central',    35.5,  38.5, -121.5, -119.5, 16),
    ('great_lakes',           43.0,  46.5,  -89.0,  -84.0, 14),
    ('northern_plains',       45.0,  48.5, -103.0,  -97.0, 16),
    ('pacific_northwest_ag',  45.5,  47.5, -123.5, -121.5, 12),
    ('utah_canyon',           37.0,  39.5, -112.0, -109.0, 16),
    ('urban_sprawl',          33.5,  34.5, -118.5, -117.0, 14),
    ('urban_midwest',         41.5,  42.2,  -88.2,  -87.5, 10),
]

EARTH_RADIUS = 6378137.0


def mercator(lat: float, lon: float):
    x = math.radians(lon) * EARTH_RADIUS
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * EARTH_RADIUS
    return x, y


def fetch(latitude, longitude, pixels, ground_sample_distance, timeout=40):
    """Export one square image centred on a point, at a chosen resolution."""
    x, y = mercator(latitude, longitude)
    # Web Mercator over-states distance by 1/cos(lat); correct so the requested
    # ground sample distance is the real one.
    half = 0.5 * pixels * ground_sample_distance / math.cos(math.radians(latitude))
    bbox = f'{x - half},{y - half},{x + half},{y + half}'
    url = (
        f'{SERVICE}?bbox={bbox}&bboxSR=102100&imageSR=102100'
        f'&size={pixels},{pixels}&format=jpg&f=image'
    )
    request = urllib.request.Request(url, headers={'User-Agent': 'nordic-ai-cup/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return image


def usable(image) -> bool:
    """Reject the blank tiles the service returns where it has no coverage."""
    if image is None or image.size == 0:
        return False
    if float(image.std()) < 14.0:
        return False
    # A large black margin means the request ran off the edge of the coverage.
    dark = float((image.max(axis=2) < 12).mean())
    return dark < 0.08


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(PROJECT / 'external'))
    parser.add_argument('--pixels', type=int, default=1792)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--test-fraction', type=float, default=0.30)
    arguments = parser.parse_args()

    output = Path(arguments.output)
    (output / 'train').mkdir(parents=True, exist_ok=True)
    (output / 'test').mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(arguments.seed)
    jobs = []
    for name, lat_min, lat_max, lon_min, lon_max, samples in REGIONS:
        for index in range(samples):
            latitude = float(rng.uniform(lat_min, lat_max))
            longitude = float(rng.uniform(lon_min, lon_max))
            # Vary the ground sample distance so the detector does not learn one
            # relationship between object size and the size of the things around
            # it. The reference frames sit at 0.19 m per pixel.
            gsd = float(rng.uniform(0.15, 0.60))
            # Split by region and index, so a held-out location is a different
            # place and not a neighbouring crop of a trained one.
            split = 'test' if (index % 10) < int(arguments.test_fraction * 10) else 'train'
            jobs.append((split, f'{name}_{index:03d}', latitude, longitude, gsd))

    def run(job):
        split, name, latitude, longitude, gsd = job
        destination = output / split / f'{name}.jpg'
        if destination.exists():
            return split, True
        try:
            image = fetch(latitude, longitude, arguments.pixels, gsd)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            print(f'  {name}: {error}', flush=True)
            return split, False
        if not usable(image):
            return split, False
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return split, True

    kept = {'train': 0, 'test': 0}
    with ThreadPoolExecutor(arguments.workers) as pool:
        for done, (split, ok) in enumerate(pool.map(run, jobs), 1):
            kept[split] += int(ok)
            if done % 25 == 0:
                print(f'  {done}/{len(jobs)} requested', flush=True)

    print(f'kept {kept["train"]} train and {kept["test"]} test mosaics '
          f'of {arguments.pixels}x{arguments.pixels} in {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
