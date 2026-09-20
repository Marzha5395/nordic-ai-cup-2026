"""Record every view the evaluation service sends us.

The service does not publish the validation images; it pushes them to our
endpoint one frame at a time. The use case explicitly allows keeping them
("You are allowed to record and keep the validation sequence"), and they are
the single most valuable thing available: the supplied Helsinki scene is 25
heavily overlapping frames of one location, which is the weakest part of the
training data. A 249-frame sequence over somewhere else is exactly the
background diversity that is missing.

Recording happens on a background thread with a bounded queue. Writing a PNG
takes tens of milliseconds and the frame budget is 333 ms, so it must not sit
in the request path; and if the queue ever backs up, frames are dropped rather
than allowed to slow the answer down. Nothing here can raise into ``predict``.

Enable by setting ``DRONE_RECORD_DIR``. Off by default, so an evaluation run is
never affected by it unless asked for.
"""

import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RECORD_DIR = os.environ.get('DRONE_RECORD_DIR', '')
QUEUE_LIMIT = int(os.environ.get('DRONE_RECORD_QUEUE', '64'))


class Recorder:
    """Write transmitted views and their geometry to disk, off the hot path."""

    def __init__(self, directory: str, queue_limit: int = QUEUE_LIMIT):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_limit)
        self._thread = threading.Thread(target=self._run, name='drone-recorder',
                                        daemon=True)
        self._thread.start()
        self.written = 0
        self.dropped = 0
        logger.info('Recording views to %s', self.directory)

    def submit(self, sequence_id: str, frame: int, image: np.ndarray,
               metadata: dict) -> None:
        try:
            self._queue.put_nowait((sequence_id, frame, image, metadata))
        except queue.Full:
            # Never let disk I/O cost us a frame.
            self.dropped += 1

    def _run(self) -> None:
        while True:
            sequence_id, frame, image, metadata = self._queue.get()
            try:
                self._write(sequence_id, frame, image, metadata)
                self.written += 1
            except Exception:
                logger.exception('Could not record frame %s', frame)
            finally:
                self._queue.task_done()

    def _write(self, sequence_id: str, frame: int, image: np.ndarray,
               metadata: dict) -> None:
        # The sequence id is a server-generated uuid; keep it as the folder
        # name so separate attempts do not overwrite each other.
        safe = ''.join(c for c in sequence_id if c.isalnum() or c in '-_')[:64]
        directory = self.directory / (safe or 'unknown')
        (directory / 'views').mkdir(parents=True, exist_ok=True)
        (directory / 'meta').mkdir(parents=True, exist_ok=True)
        stem = f'{frame:06d}_L{metadata["resolution_level"]}'
        cv2.imwrite(str(directory / 'views' / f'{stem}.png'), image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        with open(directory / 'meta' / f'{stem}.json', 'w') as handle:
            json.dump(metadata, handle, indent=1)


_recorder: Optional[Recorder] = None


def get_recorder() -> Optional[Recorder]:
    """The process-wide recorder, or None when recording is off."""
    global _recorder
    if not RECORD_DIR:
        return None
    if _recorder is None:
        try:
            _recorder = Recorder(RECORD_DIR)
        except Exception:
            logger.exception('Could not start the recorder; continuing without it')
            return None
    return _recorder


def record(request, image: np.ndarray, annotations) -> None:
    """Best-effort capture of one request. Never raises."""
    recorder = get_recorder()
    if recorder is None:
        return
    try:
        view = request.view
        recorder.submit(request.sequence_id, request.frame, image, {
            'sequence_id': request.sequence_id,
            'frame': request.frame,
            'frame_index': request.frame_index,
            'resolution_level': view.resolution_level,
            'center_x': view.center_x,
            'center_y': view.center_y,
            'source_region_xyxy': list(view.source_region_xyxy),
            'original_width': request.original_width,
            'original_height': request.original_height,
            # Our own answer, so the recording can be inspected later and used
            # as a starting point for pseudo-labels.
            'predictions': [
                {'object_id': a.object_id,
                 'bbox': [float(c) for c in a.bbox],
                 'confidence': float(a.confidence)}
                for a in annotations
            ],
        })
    except Exception:
        logger.exception('Could not queue frame %s for recording', request.frame)
