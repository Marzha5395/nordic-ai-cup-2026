"""The detector and camera policy the server calls.

The work is in the ``solution`` package; this module is the seam the template
defines, kept thin on purpose:

* ``solution.detector``  -- a YOLO model with a P2/4 head, run on the 960x540
  view and answering in source pixels;
* ``solution.motion``    -- how fast the ground drifts, from phase correlation
  between consecutive views;
* ``solution.tracking``  -- the world model, which is what makes it possible to
  answer for the whole source frame while looking at a quarter of it;
* ``solution.planner``   -- where to point the camera next, from a staleness
  map of the ground.

The model is loaded at import so that the first inference -- always the
slowest -- happens while the server is starting rather than during the first
scored frame.
"""

import logging

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from solution.runtime import predict as _predict, warm_up

logger = logging.getLogger(__name__)


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame: report detections and pick the next camera position."""
    return _predict(request)


# Load the model and run a complete synthetic request through it now, so the
# first scored frame is not the one that pays for it.
warm_up()
