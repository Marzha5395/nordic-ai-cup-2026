"""Every number worth arguing about, in one place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

WEIGHTS_PATH = PROJECT_ROOT / 'weights' / 'detector.pt'
DETECTOR_IMAGE_SIZE = 960
# Low, deliberately. A weak detection that a track later confirms is worth far
# more than the false positive it might have been, and confidence is decided by
# the world model rather than by the detector.
DETECTOR_CONFIDENCE = 0.05
DETECTOR_NMS_IOU = 0.55
DETECTOR_MAX_DETECTIONS = 120
DETECTOR_HALF_PRECISION = True

# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

# The ground drifts under the camera along a straight line, but the field is
# not uniform: it is linear in image position and spans roughly 56 to 76 source
# pixels per frame across one frame. See solution/motion.py for the numbers.
MAXIMUM_DRIFT_PER_FRAME = 260.0
FLOW_TILE_PIXELS = 256                      # view pixels per correlated tile
FLOW_SAMPLE_HISTORY = 2000
# The gradients hold over a sequence; the offset does not, so it is re-measured
# from the samples of the last few frames only.
FLOW_OFFSET_FRAMES = 8
# Barely any regularisation: measured against the annotated displacements of
# the reference scene, a ridge of 1.0 doubles the error and 5.0 triples it.
# Just enough to keep the first frame's couple of dozen tiles, which all sit in
# one quarter of the frame, from running away with six parameters.
FLOW_RIDGE = 0.25
# Source pixels of extra drift per 1000 px of image position. The reference
# scene sits at about 6 horizontally and 13 vertically.
FLOW_MAXIMUM_GRADIENT = 25.0
PHASE_CORRELATION_MINIMUM_OVERLAP = 160     # source pixels, per axis
PHASE_CORRELATION_MINIMUM_RESPONSE = 0.03
# What is left after the affine field is removed: object height and local
# relief. Small, so the per-track correction is allowed only a small range.
TRACK_BIAS_GAIN = 0.40
TRACK_POSITION_GAIN = 0.85                  # alpha of the alpha-beta filter
TRACK_MAXIMUM_BIAS = 6.0                    # source pixels per frame

# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #

MATCH_IOU = 0.15
MATCH_DISTANCE_RATIO = 0.80                 # of the mean box size
SAME_CLASS_MATCH_BONUS = 0.25
SIZE_SMOOTHING = 0.45
MAXIMUM_MISSES = 4
UNCONFIRMED_MISSES = 2                      # a one-sighting track dies faster
CONFIDENCE_STALENESS_DECAY = 0.030          # per frame since the last sighting
MINIMUM_OUTPUT_CONFIDENCE = 0.02
OUTPUT_NMS_IOU_SAME_CLASS = 0.45
OUTPUT_NMS_IOU_ANY_CLASS = 0.70
SECOND_CLASS_MINIMUM_SHARE = 0.25
SECOND_CLASS_CONFIDENCE_FACTOR = 0.25
MAXIMUM_ANNOTATIONS = 500

# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

# Level 1 is the only level that can keep up. Its view covers a quarter of the
# frame and may move 1102 px per frame, which is four times the rate at which
# new ground arrives; level 2 covers a sixteenth and may move 551 px, which is
# barely the rate at which new ground arrives and leaves nothing for turning
# around. Level 0 sees everything and resolves almost none of it.
PREFERRED_LEVEL = 1
# How much of a patch's staleness one look at that level clears. Level 1 is
# deliberately not 1.0: the detector misses things at half scale, so a patch
# that has been looked at once is not finished with.
LEVEL_DETECTION_QUALITY = {0: 0.30, 1: 0.75, 2: 1.00}
COVERAGE_CELL = 64                          # source pixels
# An object crosses the frame in about thirty frames, so ground nobody has
# looked at for eighteen is nearly as likely to be hiding something as ground
# nobody has ever looked at. Capping the age there stops the policy from
# chasing the frame edge at the expense of everything behind it: on the
# reference scene, a cap of 45 scores 0.73 and anything from 15 to 30 scores
# 0.75-0.80.
COVERAGE_MAXIMUM_AGE = 18.0
CANDIDATE_STEP = 64                         # source pixels between candidate centers
TRACK_REVISIT_WEIGHT = 45.0
UNCONFIRMED_TRACK_WEIGHT = 2.5
# Utility charged per pixel the camera centre travels. A greedy staleness
# maximiser with no such cost jumps across the frame whenever two candidates
# are nearly tied, which leaves no overlap between consecutive views.
MOVE_COST = 0.85
# Tie-breaking noise on the initial staleness map, as a fraction of the cap.
COVERAGE_INITIAL_JITTER = 0.15
COVERAGE_TIE_BREAK_SEED = 0
