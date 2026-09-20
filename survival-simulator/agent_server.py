"""Agent endpoint.

The evaluator stops a game once the accumulated wait for this endpoint passes
1200 seconds (raised from 600). A game is 30000 ticks, so the whole round trip
-- network plus server -- has to average under 40 ms. The server side is kept as small as
possible: the body is decoded with the standard JSON parser instead of being
validated field by field, the policy runs directly on the event loop (no
thread-pool hop), and the answer is written back as pre-encoded bytes.

Run it with a single worker; the policy keeps per-game state:
    uvicorn agent_server:app --host 0.0.0.0 --port 9052 --no-access-log
"""
import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response

from src.utils.DTOs import StepResponse
from src.utils.controllers.survival_policy import SurvivalPolicy

HOST = "0.0.0.0"
PORT = 9052

logger = logging.getLogger(__name__)
app = FastAPI(title="Survival Simulator Agent Endpoint")
policy = SurvivalPolicy()


def _hold_still(agent_status):
    return [
        {"agent_id": agent["agent_id"], "move_distance": 0.0, "move_direction": 0.0,
         "turn_angle": 0.0, "spawn_agent": False}
        for agent in agent_status
        if isinstance(agent, dict) and "agent_id" in agent
    ]


def _decide(payload: dict) -> list:
    agent_status = payload.get("agent_status") or []
    if payload.get("game_status") == "game_over":
        policy.reset()
        return []
    try:
        return policy.decide(agent_status, float(payload.get("sim_time", 0.0)))
    except Exception:  # an error response ends nothing but costs the tick; never send one
        logger.exception("policy failed; holding still this tick")
        policy.reset()
        return _hold_still(agent_status)


@app.post("/predict")
async def predict_endpoint(request: Request) -> Response:
    """Receives the current simulation state and returns actions for all agents."""
    try:
        payload = json.loads(await request.body())
    except ValueError:
        return Response(b'{"actions":[]}', media_type="application/json")
    if not isinstance(payload, dict):
        return Response(b'{"actions":[]}', media_type="application/json")
    # Async handlers run one at a time on the event loop, so the policy is never
    # entered concurrently and needs no lock.
    body = json.dumps({"actions": _decide(payload)}, separators=(",", ":")).encode()
    return Response(body, media_type="application/json")


def predict(step: StepResponse) -> dict:
    """The same decision for an already-parsed request (used by the tests)."""
    return {"actions": _decide(step.model_dump())}


@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, access_log=False, log_level="warning")
