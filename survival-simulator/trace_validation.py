import argparse
from collections import defaultdict, deque
import json
import math

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
from src.utils.controllers.predator_avoidance import segment_distance, wrap
from src.utils.controllers.survival_policy import SurvivalPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--duration", type=float, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tuning", type=json.loads)
    parser.add_argument("--orientations", action="store_true")
    args = parser.parse_args()
    sim = SimulationCore(seed=args.seed)
    policy = SurvivalPolicy(tuning=args.tuning)
    traces = defaultdict(lambda: deque(maxlen=40))
    actions = []
    tick = 0
    origins = {}
    with open(args.output, "x") as output:
        while sim.env.agents and sim.env.time < args.duration:
            previous = list(sim.env.agents)
            state = sim.step(actions)
            for agent in previous:
                if agent.agent_id in sim.env.agents_dict or agent.energy > 0 or agent.age > 65:
                    continue
                nearest = min((math.hypot(f.x - agent.x, f.y - agent.y) for f in sim.env.fruits), default=9999)
                if nearest < 40 and not args.orientations:
                    summary = {"seed": args.seed, "id": agent.agent_id, "time": sim.env.time,
                               "age": agent.age, "nearest_food": nearest, "score": sim.env.score}
                    print(json.dumps(summary), file=output)
                    for frame in traces[agent.agent_id]:
                        print(json.dumps(frame), file=output)
                    print(json.dumps(summary), flush=True)
                    return
            for agent in sim.env.agents:
                origins.setdefault(agent.agent_id, agent.direction)
            old_headings = {key: memory.heading for key, memory in policy.agents.items()}
            decisions = policy.decide(state["observations"], state["sim_time"])
            actions = [(a["agent_id"], ActionRequest(**a)) for a in decisions]
            if args.orientations:
                for agent in sim.env.agents:
                    memory = policy.agents[agent.agent_id]
                    bad = []
                    for group in memory.edges.values():
                        for edge in group:
                            angle = math.atan2(edge[3] - edge[1], edge[2] - edge[0]) + origins[agent.agent_id]
                            error = (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4
                            if abs(error) > 0.0001:
                                bad.append([edge, error])
                    if bad:
                        summary = {"id": agent.agent_id, "time": sim.env.time, "age": agent.age,
                                   "newborn": agent.agent_id not in old_headings,
                                   "origin": origins[agent.agent_id], "bad_edges": bad[:3],
                                   "neighbors": [o for o in policy.statuses[agent.agent_id]["observations"] if o["type"] == "Agent"],
                                   "parents": [{"id": a.agent_id, "real_heading": a.direction,
                                                "origin": origins[a.agent_id], "memory_heading": old_headings.get(a.agent_id)}
                                               for a in sim.env.agents]}
                        print(json.dumps(summary), file=output)
                        print(json.dumps(summary), flush=True)
                        return
            if tick % 5 == 0:
                for action in decisions:
                    agent = sim.env.agents_dict[action["agent_id"]]
                    status = policy.statuses[agent.agent_id]
                    memory = policy.agents[agent.agent_id]
                    target = memory.target
                    foods = sorted(sim.env.fruits, key=lambda f: math.hypot(f.x - agent.x, f.y - agent.y))[:2]
                    observations = status["observations"]
                    edges = [o["coords"] for o in observations if o["type"] == "Edge"]
                    edges.sort(key=lambda edge: segment_distance(0, 0, (*edge[0], *edge[1])))
                    heading = memory.heading - action["turn_angle"]
                    c, s = math.cos(heading), math.sin(heading)
                    map_edges = sorted((edge for group in memory.edges.values() for edge in group),
                                       key=lambda edge: segment_distance(memory.last_x, memory.last_y, edge))[:3]
                    local_edges = []
                    for edge in map_edges:
                        points = []
                        for px, py in (edge[:2], edge[2:]):
                            dx, dy = px - memory.last_x, py - memory.last_y
                            points.append([round(dx * c + dy * s, 2), round(-dx * s + dy * c, 2)])
                        local_edges.append(points)
                    frame = {
                        "t": round(sim.env.time, 1), "energy": round(agent.energy, 3), "age": round(agent.age, 1),
                        "biome": status["biome"], "speed": agent.speed,
                        "pose": [round(float(v), 3) for v in (agent.x, agent.y, agent.direction)],
                        "action": action, "retired": memory.retired, "senescent": memory.senescent,
                        "target_polar": [round(math.hypot(target.x - memory.last_x, target.y - memory.last_y), 2),
                                         round(wrap(math.atan2(target.y - memory.last_y, target.x - memory.last_x) - heading), 3),
                                         round(target.seen, 1), target.born, target.visited] if target else None,
                        "map_edges": local_edges,
                        "neighbors": [[o["id"], round(o["distance"], 1), round(o["angle"], 3),
                                       round(policy.statuses.get(o["id"], {}).get("energy", 0), 1)]
                                      for o in observations if o["type"] == "Agent"][:4],
                        "visible_foods": [[round(o["distance"], 2), round(o["angle"], 3)]
                                          for o in observations if o["type"] == "Fruit"][:5],
                        "predators": [[round(o["distance"], 2), round(o["angle"], 3), round(o["rel_dir"], 3)]
                                      for o in observations if o["type"] == "Predator"][:3],
                        "edges": [[[round(float(v), 2) for v in p] for p in edge] for edge in edges[:3]],
                        "real_foods": [[f.fruit_id, round(float(f.x - agent.x), 2), round(float(f.y - agent.y), 2),
                                        round(f.energy, 1), round(f.radius, 2)] for f in foods],
                    }
                    traces[agent.agent_id].append(frame)
            tick += 1
        summary = {"seed": args.seed, "time": sim.env.time, "score": sim.env.score, "case_found": False,
                   "orientation_checks_passed": args.orientations}
        print(json.dumps(summary), file=output)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
