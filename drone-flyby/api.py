"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from example import predict
from solution.runtime import detector_status
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        # 'ready' before an attempt, or every frame is answered empty.
        'detector': detector_status(),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    status = detector_status()
    if status != 'ready':
        logger.error('detector is %s -- every frame will be answered empty', status)
    uvicorn.run('api:app', host=HOST, port=PORT)
