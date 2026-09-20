"""Keep what the evaluator sends, so a validation attempt can be looked at afterwards.

Off unless ``MEDICAL_RECORD_DIR`` is set. Writing happens on a background thread and every error is
swallowed: recording must never slow down or break an answer. Nothing here runs during scoring.
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
        index, request, response = _queue.get()
        try:
            root.mkdir(parents=True, exist_ok=True)
            stem = f'{index:04d}_{Path(request.audio_filename).stem[:60]}'
            (root / f'{stem}.mp3').write_bytes(base64.b64decode(request.audio_base64))
            (root / f'{stem}.json').write_text(json.dumps({
                'audio_filename': request.audio_filename,
                'questions': list(request.questions),
                'answers': list(response.answers),
                'evidence_start': list(response.evidence_start),
                'evidence_end': list(response.evidence_end),
            }, indent=1))
        except Exception:
            logger.exception('Recording failed; continuing without it')


def record(request, response):
    global _queue
    directory = os.environ.get('MEDICAL_RECORD_DIR')
    if not directory:
        return
    try:
        with _lock:
            if _queue is None:
                _queue = queue.Queue(maxsize=500)
                threading.Thread(target=_writer, args=(Path(directory),), daemon=True).start()
            index = _queue.qsize() + getattr(record, 'seen', 0)
            record.seen = getattr(record, 'seen', 0) + 1
        _queue.put_nowait((index, request, response))
    except Exception:
        logger.exception('Recording failed; continuing without it')
