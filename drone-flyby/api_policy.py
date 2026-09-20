"""The endpoint the evaluation service calls.

You should not need to change much in here. This variant serves the memory policy in ``policy_controller.py``
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
from policy_controller import predict
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

    # Every rule this checks is a rule the evaluator also enforces. An invalid
    # response would cost the whole frame, so fall back to an empty answer
    # (keeping the camera command) instead of failing the request.
    try:
        validate_response(response)
    except Exception:
        logger.exception('invalid response on frame %s; sending empty '
                         'annotations', request.frame)
        response = DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=[], requested_view=response.requested_view)
        try:
            validate_response(response)
        except Exception:
            logger.exception('camera command also invalid on frame %s; '
                             'holding', request.frame)
            response.requested_view = None

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
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api_policy:app', host=HOST, port=PORT)
