import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import random
import time

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
from src.utils.controllers.dummy_agent_policy import action_decision
from src.utils.controllers.survival_policy import DEFAULT_POPULATION, DEFAULT_TUNING


def validate_seed(
    seed,
    policy_name="survival",
    duration=3000.0,
    progress=0.0,
    population=DEFAULT_POPULATION,
    share_maps=True,
    predictive_escape=True,
    camping=True,
    world_memory=False,
    tuning=None,
    settings=None,
):
    started = time.perf_counter()
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    policy = None
    if policy_name == "survival":
        from src.utils.controllers.survival_policy import SurvivalPolicy

        policy = SurvivalPolicy(
            population=population,
            share_maps=share_maps,
            predictive_escape=predictive_escape,
            camping=camping,
            world_memory=world_memory,
            tuning=tuning,
        )
    elif policy_name == "forager":
        from src.utils.controllers.forager_policy import ForagerPolicy

        policy = ForagerPolicy(settings=settings)
    actions = []
    peak_population = len(sim.env.agents)
    decision_seconds = 0.0
    max_decision_seconds = 0.0
    next_progress = progress
    fruit_score = 0.0
    predator_penalty = 0.0
    previous_score = 0.0
    ticks = 0
    deaths = {"starvation": 0, "predation": 0}
    last_deaths = []
    harvest_energy = 0.0
    harvest_count = 0
    rest_ticks = 0
    agent_ticks = 0
    origins = {}
    localization_error = 0.0
    localization_max = 0.0
    while sim.env.agents and sim.env.time < duration:
        previous_agents = list(sim.env.agents)
        previous_fruits = list(sim.env.fruits)
        state = sim.step(actions)
        for fruit in previous_fruits:
            if fruit.fruit_id not in sim.env.fruits_dict and fruit.age <= 100:
                harvest_energy += fruit.energy
                harvest_count += 1
        for agent in previous_agents:
            if agent.agent_id not in sim.env.agents_dict:
                cause = "starvation" if agent.energy <= 0 else "predation"
                deaths[cause] += 1
                last_deaths.append(
                    {
                        "id": agent.agent_id,
                        "cause": cause,
                        "time": round(sim.env.time, 1),
                        "age": round(agent.age, 1),
                        "speed": round(agent.speed, 2),
                        "hearing": round(agent.hearing_radius, 1),
                        "vision": round(agent.vision_radius, 1),
                        "nearest_food": round(min((float(((fruit.x - agent.x) ** 2 + (fruit.y - agent.y) ** 2) ** 0.5)
                                                   for fruit in sim.env.fruits), default=9999), 1),
                        "memory": {
                            "fruits": len(policy.agents[agent.agent_id].fruits),
                            "trees": len(policy.agents[agent.agent_id].trees),
                            "retired": policy.agents[agent.agent_id].retired,
                            "stuck": policy.agents[agent.agent_id].stuck,
                        } if policy_name == "survival" and agent.agent_id in policy.agents else {
                            "localized": policy.agents[agent.agent_id].localized,
                            "senescent": policy.agents[agent.agent_id].senescent,
                            "target": policy.agents[agent.agent_id].target_kind,
                            "stuck": policy.agents[agent.agent_id].stuck,
                        } if policy is not None and agent.agent_id in policy.agents else None,
                    }
                )
                last_deaths[:] = last_deaths[-5:]
        ticks += 1
        delta = state["score"] - previous_score - sim.dt
        fruit_score += max(0.0, delta)
        predator_penalty += max(0.0, -delta)
        previous_score = state["score"]
        peak_population = max(peak_population, state["num_agents"])
        for agent in sim.env.agents:
            origins.setdefault(agent.agent_id, (agent.x, agent.y, agent.direction))
        before = time.perf_counter()
        if policy is None:
            decisions = [action_decision(agent, rng) for agent in state["observations"]]
        else:
            decisions = [
                ActionRequest(**action) for action in policy.decide(state["observations"], state["sim_time"])
            ]
        elapsed = time.perf_counter() - before
        decision_seconds += elapsed
        max_decision_seconds = max(max_decision_seconds, elapsed)
        actions = [(action.agent_id, action) for action in decisions]
        rest_ticks += sum(action.move_distance == 0 for action in decisions)
        agent_ticks += len(decisions)
        if policy_name == "forager":
            for agent in sim.env.agents:
                memory = policy.agents[agent.agent_id]
                if memory.localized:
                    error = math.hypot(memory.last_x - agent.x, memory.last_y - agent.y)
                    localization_error += error
                    localization_max = max(localization_max, error)
        elif policy is not None:
            for agent in sim.env.agents:
                memory = policy.agents[agent.agent_id]
                ox, oy, heading = origins[agent.agent_id]
                ex = ox + math.cos(heading) * memory.last_x - math.sin(heading) * memory.last_y
                ey = oy + math.sin(heading) * memory.last_x + math.cos(heading) * memory.last_y
                error = math.hypot(ex - agent.x, ey - agent.y)
                localization_error += error
                localization_max = max(localization_max, error)
        if progress and sim.env.time >= next_progress:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "time": round(sim.env.time, 1),
                        "score": round(state["score"], 2),
                        "alive": state["num_agents"],
                        "energy": round(sum(a.energy for a in sim.env.agents), 1),
                        "trees": len(sim.env.trees),
                        "fruits": len(sim.env.fruits),
                        "predators": len(sim.env.predators),
                        "deaths": deaths,
                        "agents": [
                            {
                                "id": a.agent_id,
                                "energy": round(a.energy, 1),
                                "age": round(a.age, 1),
                                "speed": round(a.speed, 1),
                                "vision": round(a.vision_radius, 1),
                            }
                            for a in sim.env.agents
                        ]
                        if len(sim.env.agents) <= 3
                        else [],
                    }
                ),
                flush=True,
            )
            next_progress += progress
    result = {
        "seed": seed,
        "policy": policy_name,
        "population_target": population,
        "share_maps": share_maps,
        "predictive_escape": predictive_escape,
        "camping": camping,
        "world_memory": world_memory,
        "tuning": policy.tuning if policy_name == "survival" else {},
        "settings": policy.settings if policy_name == "forager" else {},
        "time": round(sim.env.time, 1),
        "score": round(sim.env.score, 3),
        "alive": len(sim.env.agents),
        "peak_population": peak_population,
        "births": sim.env._next_agent_id - 5,
        "harvest_count": harvest_count,
        "mean_fruit_energy": round(harvest_energy / max(1, harvest_count), 2),
        "rest_fraction": round(rest_ticks / max(1, agent_ticks), 3),
        "localization_mean_error": round(localization_error / max(1, agent_ticks), 3),
        "localization_max_error": round(localization_max, 3),
        "positive_score_delta": round(fruit_score, 3),
        "negative_score_delta": round(predator_penalty, 3),
        "decision_ms_mean": round(1000 * decision_seconds / max(1, ticks), 3),
        "decision_ms_max": round(1000 * max_decision_seconds, 3),
        "decision_seconds": round(decision_seconds, 3),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "deaths": deaths,
        "last_deaths": last_deaths,
    }
    return result


def run_seed(args):
    return validate_seed(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--policy", choices=["survival", "forager", "dummy"], default="survival")
    parser.add_argument("--duration", type=float, default=3000.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--no-sharing", action="store_true")
    parser.add_argument("--no-prediction", action="store_true")
    parser.add_argument("--no-camping", action="store_true")
    parser.add_argument("--world-memory", action="store_true")
    parser.add_argument("--tuning", type=json.loads)
    parser.add_argument("--override-tuning", type=json.loads)
    parser.add_argument("--settings", type=json.loads)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.override_tuning is not None:
        args.tuning = dict(DEFAULT_TUNING if args.tuning is None else args.tuning)
        args.tuning.update(args.override_tuning)
    runs = [
        (
            seed,
            args.policy,
            args.duration,
            args.progress,
            args.population,
            not args.no_sharing,
            not args.no_prediction,
            not args.no_camping,
            args.world_memory,
            args.tuning,
            args.settings,
        )
        for seed in args.seeds
    ]
    if args.workers == 1:
        results = []
        for run in runs:
            result = run_seed(run)
            print(json.dumps(result), flush=True)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = []
            for result in pool.map(run_seed, runs):
                print(json.dumps(result), flush=True)
                results.append(result)
    if args.output:
        with open(args.output, "x") as output:
            json.dump(results, output, indent=2)
    print(
        json.dumps(
            {
                "runs": len(results),
                "mean_score": round(sum(r["score"] for r in results) / len(results), 3),
                "mean_survival_time": round(sum(r["time"] for r in results) / len(results), 1),
                "min_survival_time": min(r["time"] for r in results),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
