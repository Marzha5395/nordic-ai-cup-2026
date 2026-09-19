"""Keep what the evaluator sends, so a validation flight can be looked at afterwards.

The README allows recording the validation sequence. Off unless ``DRONE_RECORD_DIR``
is set. Writing happens on a background thread and every error is swallowed:
recording must never slow down or break an answer.
"""

import base64
import json
import logging
import os
from pathlib import Path
import queue
import threading


logger = logging.getLogger(__name__)
_queue = None
_lock = threading.Lock()


def _writer(root):
    while True:
        request, response = _queue.get()
        try:
            folder = root / request.sequence_id
            folder.mkdir(parents=True, exist_ok=True)
            stem = f'{request.frame_index:05d}_frame{request.frame:06d}'
            (folder / f'{stem}.png').write_bytes(base64.b64decode(request.view.image))
            meta = request.model_dump(exclude={'view': {'image'}})
            meta['response'] = response.model_dump() if response is not None else None
            (folder / f'{stem}.json').write_text(json.dumps(meta))
        except Exception:
            logger.exception('Recording failed; continuing without it')


def record(request, response):
    global _queue
    directory = os.environ.get('DRONE_RECORD_DIR')
    if not directory:
        return
    try:
        with _lock:
            if _queue is None:
                _queue = queue.Queue(maxsize=2000)
                threading.Thread(target=_writer, args=(Path(directory),), daemon=True).start()
        _queue.put_nowait((request, response))
    except Exception:
        logger.exception('Recording failed; continuing without it')
