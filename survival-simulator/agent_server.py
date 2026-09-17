from threading import Lock
from fastapi import FastAPI, Body
from src.utils.DTOs import StepResponse
from src.utils.controllers.survival_policy import SurvivalPolicy

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")
policy = SurvivalPolicy()
policy_lock = Lock()


@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    with policy_lock:  # deterministic for testing
        if step.game_status == "game_over":
            policy.reset()
            actions = []
        else:
            actions = policy.decide([agent.model_dump() for agent in step.agent_status], step.sim_time)

    # Must return {"actions": [...]} format
    return {"actions": actions}


@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
