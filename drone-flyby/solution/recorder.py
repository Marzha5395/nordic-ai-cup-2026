"""Keep a copy of every attempt: the view that arrived and the answer given.

The validation sequence may be recorded and kept, and it is the only look
there is at terrain the scored flight might resemble. The reference scene is
one place; a recording of the validation flight is another, and it is what
says whether the detector sees objects there or sees objects everywhere.

Writing happens on a background thread from a bounded queue, so a slow disk
costs recordings rather than frames. Nothing here may raise into a request.

Set ``DRONE_RECORD=0`` to turn it off, ``DRONE_RECORD_DIR`` to move it.
"""

import base64
import json
import logging
import os
import queue
import re
import threading
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get('DRONE_RECORD', '1') not in {'0', 'false', 'False', ''}
_DIRECTORY = Path(os.environ.get('DRONE_RECORD_DIR', config.PROJECT_ROOT / 'recordings'))

_queue: 'queue.Queue' = queue.Queue(maxsize=512)
_thread = None
_thread_lock = threading.Lock()


def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(name))[:80] or 'sequence'


def _writer() -> None:
    while True:
        item = _queue.get()
        try:
            _write(*item)
        except Exception:                            # noqa: BLE001 - never fatal
            logger.exception('could not write a recording')
        finally:
            _queue.task_done()


def _write(request: dict, image: str, response: dict) -> None:
    directory = _DIRECTORY / _safe(request.get('sequence_id', 'sequence'))
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{int(request.get('frame_index', 0)):04d}_{int(request.get('frame', 0)):04d}"
    if image:
        (directory / f'{stem}.png').write_bytes(base64.b64decode(image))
    with open(directory / f'{stem}.json', 'w') as handle:
        json.dump({'request': request, 'response': response}, handle)


def _ensure_thread() -> None:
    global _thread
    if _thread is not None:
        return
    with _thread_lock:
        if _thread is None:
            _thread = threading.Thread(target=_writer, name='drone-recorder', daemon=True)
            _thread.start()


def record(request, response) -> None:
    """Queue one request and its answer for writing. Never raises."""
    if not _ENABLED:
        return
    try:
        if str(request.sequence_id).startswith('__'):
            return                                   # the warm-up, not an attempt
        payload = request.model_dump(mode='json')
        image = payload.get('view', {}).pop('image', '')
        _ensure_thread()
        _queue.put_nowait((payload, image, response.model_dump(mode='json')))
    except queue.Full:
        logger.warning('recording queue full, frame %s not recorded', request.frame)
    except Exception:                                # noqa: BLE001 - never fatal
        logger.exception('could not queue a recording')
