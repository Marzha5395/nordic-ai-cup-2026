"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9054/predict``
rather than just the host.

This server runs Astra's solution (``solver.MedicalSolver``: faster-whisper
speech recognition, then a local llama.cpp language model that answers every
question with a verbatim evidence quote). ``example.py`` and ``main.py`` are the
separate Qwen/transformers pipeline and are not used here. Prepare the models
first with ``python prepare_models.py --build-cuda`` (see AGENTS.md).
"""

from contextlib import asynccontextmanager
import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from recorder import record
from solver import fallback_response, get_solver
from utils import audio_duration_seconds, decode_audio, validate_response

HOST = '0.0.0.0'
PORT = 9054

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Load and warm both models before the first request: there is no warm-up
    # allowance, and the first inference is the slowest.
    solver = get_solver()
    try:
        yield
    finally:
        solver.llm.close()
        get_solver.cache_clear()


app = FastAPI(lifespan=lifespan)
start_time = time.time()


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation.

    The whole conversation and all of its questions arrive together, so the
    expensive half — transcription — is paid once here and shared by every
    answer. Never raises: an error falls back to answering no everywhere,
    which still returns a valid body for all ten questions.
    """
    try:
        audio_bytes = decode_audio(request.audio_base64)
        duration = audio_duration_seconds(audio_bytes)
        logger.info(
            '%s (%.1f s, %.1f MB): %d questions',
            request.audio_filename,
            duration if duration is not None else float('nan'),
            len(audio_bytes) / 1e6,
            len(request.questions),
        )
        response = get_solver().predict(audio_bytes, request.questions)
    except Exception:
        logger.exception('Prediction failed for %s', request.audio_filename)
        response = fallback_response(request.questions)

    answer = ASRQuestionResponseDto(
        answers=response.answers,
        evidence_start=response.evidence_start,
        evidence_end=response.evidence_end,
    )
    record(request, answer)
    return answer


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    """Answer every question about one conversation."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently score every
    # question about this conversation wrong.
    validate_response(response, expected_count=len(request.questions))

    return response


@app.get('/api')
def hello():
    return {
        'service': 'medical-appointment-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
