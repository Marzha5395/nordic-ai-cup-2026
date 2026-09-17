from collections import Counter
from dataclasses import dataclass, field
import math

from src.utils.controllers.predator_avoidance import (
    crosses,
    escape_move,
    maneuver_move,
    predict_predator,
    segment_distance,
    touches_edge,
    wrap,
)
from src.utils.controllers.world_memory import Landmark, WorldMemory, neighbor_heading
from src.utils.controllers.navigation import Navigator


TAU = 2 * math.pi
BIOME_SPEED = {"forest": 1.0, "grassland": 1.0, "desert": 0.8, "swamp": 0.5, "river": 0.3}
DEFAULT_POPULATION = 16
DEFAULT_TUNING = {
    "maneuvers": True,
    "tracking": True,
    "ripen": 20,
    "reserve": 180,
    "goal_steering": True,
    "food_planning": True,
    "patient": True,
    "renewal": True,
    "late_renewal": True,
    "cooperation": True,
    "selection": True,
    "half_life": 900,
    "vision_weight": 2,
    "forage_speed": 10,
    "visibility": True,
}


@dataclass
class AgentMemory:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    travel_heading: float = 0.0
    last_distance: float = 0.0
    last_age: float = -1.0
    move_heading: float = 0.0
    target: Landmark | None = None
    target_distance: float = 0.0
    target_since: float = 0.0
    edges: dict = field(default_factory=dict)
    fruits: list = field(default_factory=list)
    trees: list = field(default_factory=list)
    landmarks: list = field(default_factory=list)
    predators: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    last_scan: float = -1000.0
    scan_left: float = 0.0
    last_birth: float = -1000.0
    stuck: int = 0
    last_x: float = 0.0
    last_y: float = 0.0
    navigator: Navigator | None = None
    explored: dict = field(default_factory=dict)
    biomes: dict = field(default_factory=dict)
    explore_goal: tuple | None = None
    explore_since: float = 0.0
    last_energy: float = 0.0
    last_cost: float = 0.0
    senescent: bool = False
    retired: bool = False
    prune_visible: bool = False

    def detectable(self, point, status):
        dx, dy = point.x - self.x, point.y - self.y
        distance = math.hypot(dx, dy)
        if distance < status["hearing_radius"] - 2:
            return True
        if not self.prune_visible or distance >= status["vision_range"] - 3:
            return False
        if abs(wrap(math.atan2(dy, dx) - self.heading)) >= status["vision_angle"] / 2 - 0.01:
            return False
        return not any(crosses(self.x, self.y, point.x, point.y, edge)
                       for group in self.edges.values() for edge in group)

    def observe(self, status, now):
        previous_age = self.last_age
        self.last_age = status["age"]
        c, s = math.cos(self.heading), math.sin(self.heading)
        observations = status["observations"]
        local_edges = {}
        static = []
        objects = {"Fruit": [], "Tree": [], "Agent": [], "Predator": []}
        for obs in observations:
            kind = obs.get("type")
            if kind == "Edge":
                (ax, ay), (bx, by) = obs["coords"]
                edge = (c * ax - s * ay, s * ax + c * ay, c * bx - s * by, s * bx + c * by)
                key = (round(edge[2] - edge[0], 4), round(edge[3] - edge[1], 4))
                local_edges[(key, edge)] = None
            elif kind in objects:
                angle = self.heading + obs["angle"]
                x, y = obs["distance"] * math.cos(angle), obs["distance"] * math.sin(angle)
                objects[kind].append((x, y, obs))
                if kind in ("Fruit", "Tree"):
                    static.append((kind, x, y))
        if self.last_distance > 1e-6:
            shifts = []
            limit = max(25.0, self.last_distance * 2 + 2)
            for key, edge in local_edges:
                for known in self.edges.get(key, []):
                    dx, dy = known[0] - edge[0] - self.x, known[1] - edge[1] - self.y
                    if math.hypot(dx, dy) < limit:
                        shifts.append((dx, dy, 3))
            for kind, x, y in static:
                for tag, px, py in self.landmarks:
                    if tag == kind:
                        dx, dy = px - x - self.x, py - y - self.y
                        if math.hypot(dx, dy) < limit:
                            shifts.append((dx, dy, 2 if kind == "Tree" else 1))
            if shifts:
                votes = Counter()
                for dx, dy, weight in shifts:
                    votes[(round(dx * 10), round(dy * 10))] += weight

                def support(point):
                    ix, iy = round(point[0] * 10), round(point[1] * 10)
                    count = sum(votes[(ix + dx, iy + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                    return count, -math.hypot(point[0], point[1])

                dx, dy, _ = max(shifts, key=support)
                self.x += dx
                self.y += dy
        moved = math.hypot(self.x - self.last_x, self.y - self.last_y)
        self.stuck = self.stuck + 1 if self.last_distance > 1 and moved < self.last_distance * 0.3 else 0
        for key, edge in local_edges:
            world = (edge[0] + self.x, edge[1] + self.y, edge[2] + self.x, edge[3] + self.y)
            known = self.edges.setdefault(key, [])
            if not any(math.hypot(e[0] - world[0], e[1] - world[1]) < 3 for e in known):
                known.append(world)
        self.landmarks = [(kind, x + self.x, y + self.y) for kind, x, y in static]
        for kind, memory, expiry in (("Fruit", self.fruits, 45.0), ("Tree", self.trees, 75.0)):
            visible = objects[kind]
            seen = set()
            for x, y, obs in visible:
                wx, wy = self.x + x, self.y + y
                match = min(memory, key=lambda p: math.hypot(p.x - wx, p.y - wy), default=None)
                if match is None or math.hypot(match.x - wx, match.y - wy) > 4:
                    born = now if kind == "Fruit" and previous_age >= 0 and math.hypot(
                        wx - self.last_x, wy - self.last_y
                    ) < status["hearing_radius"] - 3 else None
                    match = Landmark(wx, wy, now, now, born=born)
                    memory.append(match)
                match.x, match.y, match.seen = wx, wy, now
                match.shared = False
                seen.add(id(match))
            memory[:] = [
                p
                for p in memory
                if now - p.seen < expiry
                and (id(p) in seen or not self.detectable(p, status))
                and (kind != "Fruit" or math.hypot(p.x - self.x, p.y - self.y) >= 9.9)
            ]
        if not self.trail or now - self.trail[-1][2] >= 2:
            self.trail.append((self.x, self.y, now))
            self.trail[:] = self.trail[-60:]
        return objects


class SurvivalPolicy:
    def __init__(
        self, population=DEFAULT_POPULATION, share_maps=True, predictive_escape=True, camping=True, world_memory=False,
        tuning=None,
    ):
        self.tuning = dict(DEFAULT_TUNING if tuning is None else tuning)
        self.statuses = {}
        self.harvest_rate = 0.0
        self.population = population
        self.share_maps = share_maps
        self.predictive_escape = predictive_escape
        self.camping = camping
        self.world = WorldMemory() if world_memory else None
        self.agents = {}
        self.last_time = None
        self.cached_actions = []
        self.cached_state = None

    def reset(self):
        self.agents.clear()
        self.statuses.clear()
        self.harvest_rate = 0.0
        self.last_time = None
        self.cached_actions = []
        self.cached_state = None
        if self.world is not None:
            self.world = WorldMemory()

    def decide(self, statuses, sim_time):
        if self.last_time == sim_time and statuses == self.cached_state:
            return [dict(action) for action in self.cached_actions]
        if self.last_time is not None and (sim_time <= self.last_time or not statuses):
            self.reset()
        if self.tuning.get("resource_budget"):
            gain = 0.0
            for status in statuses:
                memory = self.agents.get(status["agent_id"])
                if memory is not None and status["age"] > memory.last_age:
                    aging = status["age"] * 0.01 if memory.senescent else 0.0
                    gain += max(0.0, status["energy"] - memory.last_energy + memory.last_cost + aging)
            dt = sim_time - self.last_time if self.last_time is not None else 0.1
            self.harvest_rate += (gain / dt - self.harvest_rate) * (1 - math.exp(-dt / 30))
        self.statuses = {s["agent_id"]: s for s in statuses}
        alive = set(self.statuses)
        self.agents = {key: value for key, value in self.agents.items() if key in alive}
        newborns = alive - self.agents.keys()
        sensed = {}
        previous_actions = {a["agent_id"]: a for a in self.cached_actions}
        stale = set()
        for status in statuses:
            agent_id = status["agent_id"]
            memory = self.agents.setdefault(agent_id, AgentMemory(prune_visible=self.tuning.get("visibility", False)))
            if memory.last_age == status["age"] and agent_id in previous_actions:
                stale.add(agent_id)
            else:
                sensed[agent_id] = memory.observe(status, sim_time)
        if self.share_maps:
            for agent_id in sorted(newborns):
                self._inherit_map(self.agents[agent_id], sensed[agent_id]["Agent"], sim_time)
        if self.world is not None:
            self.world.update(self.agents, sensed, {s["agent_id"]: s for s in statuses}, sim_time)
        actions = []
        population = len(statuses)
        priority = (lambda s: (-self._fitness(s), s["agent_id"])) if self.tuning.get("selection") else (lambda s: s["agent_id"])
        for status in sorted(statuses, key=priority):
            agent_id = status["agent_id"]
            memory = self.agents[agent_id]
            if agent_id in stale:
                action = dict(previous_actions[agent_id])
                action.update(
                    turn_angle=0.0,
                    spawn_agent=False,
                    move_direction=wrap(memory.move_heading - memory.heading),
                )
                if status["energy"] < status["max_energy"] / 5:
                    action["move_distance"] = min(action["move_distance"], status["speed"])
                memory.last_energy = status["energy"]
                memory.last_cost = (min(action["move_distance"], status["speed"]) * 0.05
                                    + max(0.0, action["move_distance"] - status["speed"]) * 0.5 + 0.1)
                memory.last_x, memory.last_y = memory.x, memory.y
                memory.last_distance = action["move_distance"] * BIOME_SPEED.get(status["biome"], 1.0)
                memory.x += memory.last_distance * math.cos(memory.move_heading)
                memory.y += memory.last_distance * math.sin(memory.move_heading)
            else:
                action = self._action(status, memory, sensed[agent_id], sim_time, population)
            actions.append(action)
            population += int(action["spawn_agent"])
        self.last_time = sim_time
        self.cached_state = statuses
        self.cached_actions = actions
        return [dict(action) for action in actions]

    def _inherit_map(self, memory, neighbors, now):
        candidates = [(px, py, obs) for px, py, obs in neighbors if obs.get("id") in self.agents]
        if not candidates:
            return
        px, py, obs = max(candidates, key=lambda p: len(self.agents[p[2]["id"]].edges))
        parent = self.agents[obs["id"]]
        rotation = neighbor_heading(memory, px, py, obs) - parent.heading
        c, s = math.cos(rotation), math.sin(rotation)

        def transform(x, y):
            dx, dy = x - parent.x, y - parent.y
            return memory.x + px + c * dx - s * dy, memory.y + py + s * dx + c * dy

        for group in parent.edges.values():
            for edge in group:
                ax, ay = transform(edge[0], edge[1])
                bx, by = transform(edge[2], edge[3])
                key = (round(bx - ax, 4), round(by - ay, 4))
                known = memory.edges.setdefault(key, [])
                if not any(math.hypot(e[0] - ax, e[1] - ay) < 3 for e in known):
                    known.append((ax, ay, bx, by))
        for source, destination in ((parent.fruits, memory.fruits), (parent.trees, memory.trees)):
            for point in source:
                wx, wy = transform(point.x, point.y)
                if math.hypot(wx - memory.x, wy - memory.y) > 15 and not any(
                    math.hypot(p.x - wx, p.y - wy) < 4 for p in destination
                ):
                    destination.append(Landmark(wx, wy, point.seen, point.first, born=point.born))

    def _fitness(self, status):
        speed = min(status["speed"], status["sprint_speed"])
        return (speed / 5 + status["hearing_radius"] / 50
                + self.tuning.get("vision_weight", 1) * status["vision_range"] / 200
                + status["vision_angle"] * 1.5 / math.pi + min(status["sprint_speed"], 25) / 50
                + (self.tuning.get("escape_fitness", 0) if speed > 15.1 else 0))

    def _owns_food(self, status, memory, fruit, neighbors):
        def priority(agent, distance):
            traits = self.agents[agent["agent_id"]]
            movement = max(0.0, distance - 10) / BIOME_SPEED.get(agent["biome"], 1.0)
            speed = max(0.1, min(agent["speed"], agent["sprint_speed"], self.tuning.get("forage_speed", 10)))
            cost = movement * 0.05 + movement / speed * (0.1 + (agent["age"] * 0.01 if traits.senescent else 0))
            score = cost * 10 + agent["energy"] * self.tuning.get("ration_weight", 0.5) + 200 * traits.senescent
            return score, cost

        dx, dy = fruit.x - memory.x, fruit.y - memory.y
        own_score, _ = priority(status, math.hypot(dx, dy))
        for ax, ay, obs in neighbors:
            other_id = obs.get("id")
            if other_id not in self.statuses or self.agents[other_id].retired:
                continue
            other = self.statuses[other_id]
            distance = math.hypot(dx - ax, dy - ay)
            other_heading = neighbor_heading(memory, ax, ay, obs)
            angle = wrap(math.atan2(dy - ay, dx - ax) - other_heading)
            if not any(o["type"] == "Fruit" and abs(o["distance"] - distance) < 4
                       and abs(wrap(o["angle"] - angle)) * distance < 4 for o in other["observations"]):
                continue
            score, cost = priority(other, distance)
            if other["energy"] > cost + 2 and (score, other_id) < (own_score, status["agent_id"]):
                return False
        return True

    def _explore(self, status, memory, neighbors, now):
        x, y = memory.x, memory.y
        ix, iy = round(x / 80), round(y / 80)
        memory.biomes[(ix, iy)] = status["biome"]
        radius = status["vision_range"]
        steps = math.ceil(radius / 80)
        for dx in range(-steps, steps + 1):
            for dy in range(-steps, steps + 1):
                px, py = (ix + dx) * 80 - x, (iy + dy) * 80 - y
                distance = math.hypot(px, py)
                if distance < status["hearing_radius"] or (
                    distance < radius and abs(wrap(math.atan2(py, px) - memory.heading)) < status["vision_angle"] / 2
                ):
                    memory.explored[(ix + dx, iy + dy)] = now
        if memory.explore_goal is not None and now - memory.explore_since < 8 and memory.stuck < 3:
            gx, gy = memory.explore_goal
            if math.hypot(gx - x, gy - y) > 40:
                return gx, gy, 10.0
        boundaries = [edge for group in memory.edges.values() for edge in group
                      if math.hypot(edge[2] - edge[0], edge[3] - edge[1]) > 1000]
        best, goal = float("inf"), None
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                gx, gy = (ix + dx) * 80, (iy + dy) * 80
                distance = math.hypot(gx - x, gy - y)
                if not 100 < distance < 420 or any(crosses(x, y, gx, gy, edge) for edge in boundaries):
                    continue
                score = distance * 0.08
                score += 100 * math.exp(-max(0, now - memory.explored.get((ix + dx, iy + dy), -1000)) / 35)
                score += 8 * (1 - math.cos(math.atan2(gy - y, gx - x) - memory.travel_heading))
                score += {"river": 55, "desert": 25, "swamp": 5}.get(memory.biomes.get((ix + dx, iy + dy)), 0)
                score += sum(max(0, 150 - math.hypot(gx - x - ax, gy - y - ay)) * 0.3 for ax, ay, _ in neighbors)
                if score < best:
                    best, goal = score, (gx, gy)
        memory.explore_goal = goal
        memory.explore_since = now
        return (*goal, 10.0) if goal is not None else None

    def _action(self, status, memory, objects, now, population):
        x, y, heading = memory.x, memory.y, memory.heading
        energy, age = status["energy"], status["age"]
        if age > 60 and memory.last_energy - energy - memory.last_cost > age * 0.005:
            memory.senescent = True
        if population == 1:
            memory.retired = False
        healthy = not memory.senescent if self.tuning.get("patient") else age < 65
        modifier = BIOME_SPEED.get(status["biome"], 1.0)
        walk = min(status["speed"], status["sprint_speed"])
        sprint = status["sprint_speed"] if energy > status["max_energy"] / 5 + 8 else walk
        look_bias = min(self.tuning.get("look_bias", 0.0), status["vision_angle"] * 0.35)
        edges = [
            edge
            for group in memory.edges.values()
            for edge in group
            if segment_distance(x, y, edge) < status["vision_range"] + 40
        ]
        predators = [(px, py, obs) for px, py, obs in objects["Predator"] if obs["distance"] < 180]
        threats = []
        tracks = []
        for px, py, obs in predators:
            previous = min(
                memory.predators, key=lambda p: math.hypot(p[0] - x - px, p[1] - y - py), default=None
            )
            active = (
                previous is None
                or now - previous[2] > 0.15
                or math.hypot(previous[0] - x - px, previous[1] - y - py) > 0.5
            )
            predator_heading = wrap(neighbor_heading(memory, px, py, obs))
            pred_modifier = modifier
            if self.tuning.get("tracking") and previous is not None and now - previous[2] < 0.15 and active:
                displacement = math.hypot(previous[0] - x - px, previous[1] - y - py)
                pred_modifier = max(0.3, min(1.0, displacement / (15 if obs["distance"] < 90 else 11)))
            tracks.append((x + px, y + py, now))
            if active:
                px, py, predator_heading = predict_predator(px, py, predator_heading, 0, 0, heading, pred_modifier)
            threats.append((px, py, predator_heading, active, pred_modifier))
        memory.predators = tracks
        nearest_predator = min((obs["distance"] for _, _, obs in predators), default=1000.0)
        danger = nearest_predator < 95
        if self.predictive_escape:
            danger = any(math.hypot(px, py) < (100 if active else 35) for px, py, _, active, _ in threats)
        neighbors = objects["Agent"]
        competitors = [
            (ax, ay, obs) for ax, ay, obs in neighbors
            if obs.get("id") not in self.agents or not self.agents[obs["id"]].retired
        ] if self.tuning.get("cooperation") else neighbors
        target = None
        yield_food = False
        best_cost = float("inf")
        nearby_fruits = sorted(
            (fruit for fruit in memory.fruits if now - fruit.visited >= 12),
            key=lambda fruit: math.hypot(fruit.x - x, fruit.y - y),
        )[:12]
        for fruit in nearby_fruits:
            dx, dy = fruit.x - x, fruit.y - y
            distance = math.hypot(dx, dy)
            if distance < 9.9 or now - fruit.visited < 12:
                continue
            if self.tuning.get("rationing") and not self._owns_food(status, memory, fruit, neighbors):
                yield_food |= distance < 120
                continue
            cost = distance + 4 * (now - fruit.seen)
            if fruit is memory.target:
                cost -= self.tuning.get("commitment", 0)
            if self.tuning.get("food_planning") and fruit.born is not None:
                travel_time = distance / max(1, walk * modifier) * 0.1
                wait_time = max(0, self.tuning.get("ripen", 20) - (now - fruit.born) - travel_time)
                if energy > 40 + (1 + (age * 0.1 if memory.senescent else 0)) * wait_time:
                    cost += wait_time * 20 * modifier
            cost += sum(90 for ax, ay, _ in competitors if math.hypot(dx - ax, dy - ay) + 15 < distance)
            cost += sum(180 for px, py, _ in predators if math.hypot(dx - px, dy - py) < 80)
            cost += sum(80 for edge in edges if crosses(x, y, fruit.x, fruit.y, edge))
            if cost < best_cost:
                best_cost, target = cost, fruit
        waiting = False
        if target is not None:
            ripen = self.tuning.get("ripen", 0)
            fruit_age = now - target.born if target.born is not None else 20.0
            wait_time = max(0.0, ripen - fruit_age)
            drain = 1 + (age * 0.1 if memory.senescent else 0)
            waiting = wait_time > 0 and energy > 40 + drain * wait_time
            reserve = self.tuning.get("reserve", 0)
            waiting |= bool(reserve and energy > min(reserve + 80, status["max_energy"] - 65))
            remaining = math.hypot(target.x - x, target.y - y)
            if target is not memory.target or remaining < memory.target_distance - 5 or (waiting and remaining < 25):
                memory.target, memory.target_distance, memory.target_since = target, remaining, now
            elif now - memory.target_since > (8 if self.tuning.get("navigation") else 3):
                target.visited = now
                memory.target = target = None
        distance = walk
        desired = memory.travel_heading
        rest = False
        goal = None
        if target is not None:
            dx, dy = target.x - x, target.y - y
            desired = math.atan2(dy, dx)
            stop = 18.0 if waiting else self.tuning.get("pickup_stop", 5.0)
            distance = min(walk, max(0.0, (math.hypot(dx, dy) - stop) / modifier))
            rest = distance == 0
            goal = (target.x, target.y, stop)
        else:
            tree_target = None
            best_cost = float("inf")
            for tree in memory.trees:
                if (self.tuning.get("exploration") or self.tuning.get("tree_patience")) and now - tree.visited < 18:
                    continue
                dx, dy = tree.x - x, tree.y - y
                dist = math.hypot(dx, dy)
                cost = dist + max(0.0, 18 - (now - tree.visited)) * 16
                if self.tuning.get("tree_patience") and tree is memory.target:
                    cost -= 25
                cost += sum(130 for ax, ay, _ in competitors if math.hypot(dx - ax, dy - ay) < 65)
                cost += sum(200 for px, py, _ in predators if math.hypot(dx - px, dy - py) < 110)
                if cost < best_cost:
                    best_cost, tree_target = cost, tree
            if tree_target is not None and self.tuning.get("tree_patience"):
                remaining = math.hypot(tree_target.x - x, tree_target.y - y)
                if (tree_target is not memory.target or remaining < memory.target_distance - 5
                        or remaining < min(25.0, status["hearing_radius"] * 0.6)):
                    memory.target, memory.target_distance, memory.target_since = tree_target, remaining, now
                elif now - memory.target_since > self.tuning["tree_patience"]:
                    tree_target.visited = now
                    tree_target = memory.target = None
            if tree_target is not None and best_cost < 450:
                dx, dy = tree_target.x - x, tree_target.y - y
                desired = math.atan2(dy, dx)
                camp_radius = min(25.0, status["hearing_radius"] * 0.6)
                goal = (tree_target.x, tree_target.y, camp_radius * 0.8)
                if self.camping and math.hypot(dx, dy) < camp_radius and healthy:
                    rest = energy < 75 or not any(
                        math.hypot(dx - ax, dy - ay) < 80 for ax, ay, _ in competitors
                    )
                if math.hypot(dx, dy) < camp_radius and not rest:
                    tree_target.visited = now
                    rest = energy > 120 and healthy and not competitors
            if energy > 250 and age < 55 and not neighbors:
                rest = True
            if rest:
                distance = 0.0
            if now - memory.last_scan > (5.0 if rest else self.tuning.get("scan_period", 3.0)):
                memory.scan_left = TAU
                memory.last_scan = now
        if self.tuning.get("exploration"):
            explore_goal = self._explore(status, memory, neighbors, now)
            if goal is None and explore_goal is not None:
                goal = explore_goal
                desired = math.atan2(goal[1] - y, goal[0] - x)
        reserve = self.tuning.get("reserve", 0)
        if target is None and reserve and energy > reserve and not memory.senescent:
            distance, rest = 0.0, True
        stationary_reserve = self.tuning.get("stationary_reserve", 0)
        if stationary_reserve and energy > stationary_reserve and not memory.senescent:
            distance, rest, goal = 0.0, True, None
            memory.target_since = now
        if target is None and yield_food and energy > 50:
            memory.target, goal, distance, rest = None, None, 0.0, True
        if memory.retired and population > 1:
            target, goal, distance, rest = None, None, 0.0, True
        if self.tuning.get("navigation") and goal is not None and distance > 0 and not danger:
            if memory.navigator is None:
                memory.navigator = Navigator(spacing=self.tuning.get("grid_spacing", 15), margin=self.tuning.get("grid_margin"))
            memory.navigator.update(edge for group in memory.edges.values() for edge in group)
            waypoint = memory.navigator.waypoint(x, y, goal[0], goal[1], now, stop=goal[2])
            if waypoint is not None:
                dx, dy = waypoint[0] - x, waypoint[1] - y
                desired = math.atan2(dy, dx)
                distance = min(walk, math.hypot(dx, dy) / modifier)
        if rest and now - memory.last_scan > 5:
            memory.scan_left = TAU
            memory.last_scan = now
        if not danger:
            distance = min(distance, self.tuning.get("forage_speed", walk))
        if danger:
            vx, vy = 0.0, 0.0
            for px, py, obs in predators:
                weight = 1 / max(15.0, obs["distance"]) ** 2
                vx -= px * weight
                vy -= py * weight
            desired = math.atan2(vy, vx)
            distance = sprint if nearest_predator < 65 else walk
            rest = False
        if memory.stuck > 3:
            desired += math.pi / 2
        if distance > 0:
            steering_edges = [
                edge for edge in edges if segment_distance(x, y, edge) < distance * modifier + 34
            ]
            best = -float("inf")
            choice = desired
            for offset in (0.0, -0.35, 0.35, -0.7, 0.7, -1.05, 1.05, -1.4, 1.4, -1.9, 1.9, math.pi):
                angle = desired + offset
                dx, dy = math.cos(angle), math.sin(angle)
                travel = distance * modifier
                nx, ny = x + dx * travel, y + dy * travel
                score = 20 * math.cos(offset)
                for edge in steering_edges:
                    d = segment_distance(nx, ny, edge)
                    collision = touches_edge(nx, ny, edge) if self.tuning.get("precise_walls") else d < 7
                    if collision or crosses(x, y, nx, ny, edge):
                        score -= 500 + (7 - min(7, d)) * 20
                    future = segment_distance(nx + dx * 18, ny + dy * 18, edge)
                    score -= max(0, 16 - future) * (0.2 if self.tuning.get("goal_steering") and goal is not None else 2)
                for px, py, obs in predators:
                    d = math.hypot(px - dx * travel, py - dy * travel)
                    score -= 3000 / max(5, d - 15)
                    if d < 32:
                        score -= 1000
                if target is None and not danger and not (self.tuning.get("goal_steering") and goal is not None):
                    score -= sum(
                        max(0.0, 35 - math.hypot(nx + dx * 35 - tx, ny + dy * 35 - ty)) * 0.25
                        for tx, ty, when in memory.trail
                        if now - when > 4
                    )
                    score -= sum(
                        max(0.0, 40 - math.hypot(ax - dx * 30, ay - dy * 30)) * 0.5 for ax, ay, _ in neighbors
                    )
                if score > best:
                    best, choice = score, angle
            desired = choice
        if danger and self.predictive_escape:
            local_edges = [
                (ax - x, ay - y, bx - x, by - y)
                for ax, ay, bx, by in edges
                if segment_distance(x, y, (ax, ay, bx, by)) < sprint * modifier * 2 + 20
            ]
            if self.tuning.get("maneuvers"):
                distance, desired = maneuver_move(threats, desired, walk, sprint, modifier, local_edges, energy, status["max_energy"], self.tuning.get("margin", 19), look_bias)
            else:
                distance, desired = escape_move(threats, desired, walk, sprint, modifier, local_edges)
        watch = look_bias and any(active and math.hypot(px, py) < 180 for px, py, _, active, _ in threats)
        if memory.scan_left > 0 and not danger and not watch:
            turn = min(max(0.25, status["vision_angle"] * 0.8), memory.scan_left)
            memory.scan_left -= turn
        elif danger or watch:
            nearest = min(predators, key=lambda p: p[2]["distance"])
            turn = wrap(math.atan2(nearest[1], nearest[0]) - heading + look_bias)
        else:
            turn = wrap(desired - heading) if distance > 0 else 0.0
        target_population = max(3, int(self.population - now / self.tuning.get("decay", 150)))
        if self.tuning.get("half_life"):
            target_population = max(3, math.ceil(self.population * 0.5 ** (now / self.tuning["half_life"])))
        if self.tuning.get("resource_budget") and now > 200:
            target_population = min(target_population, max(3, math.ceil(self.harvest_rate / self.tuning["resource_budget"])))
        spawn = energy > self.tuning.get("birth_energy", 280) and population < target_population
        spawn |= age > 60 and energy > 155 and population < target_population + 2
        spawn |= population < 3 and energy > 150
        spawn |= age > 100 and energy > 125 and population < target_population + 3
        if self.tuning.get("renewal"):
            productive = population - sum(m.retired for m in self.agents.values())
            if self.tuning.get("resource_budget") and memory.senescent and productive > target_population:
                memory.retired = True
            birth_energy = self.tuning.get("birth_energy", 280)
            unfit = False
            if self.tuning.get("selection"):
                average = sum(self._fitness(s) for s in self.statuses.values()) / len(self.statuses)
                birth_energy = max(180, min(400, birth_energy + 100 * (average - self._fitness(status))))
                gap = self.tuning.get("selection_gap")
                unfit = gap is not None and self._fitness(status) + gap < average and productive > 3
            spawn = energy > birth_energy and productive < target_population
            late = self.tuning.get("late_renewal")
            spawn |= (memory.senescent or age > (115 if late else 90)) and energy > self.tuning.get("old_birth_energy", 115) and productive <= target_population
            spawn |= not late and age > 65 and energy > 180 and productive < target_population
            spawn = spawn and (not memory.retired or (late and productive < target_population))
            if unfit:
                spawn = False
                memory.retired |= memory.senescent
        control_cost = (min(distance, status["speed"]) * 0.05
                        + max(0.0, distance - status["speed"]) * 0.5 + min(math.pi, abs(turn)) / TAU)
        birth_interval = self.tuning.get("old_birth_interval", 4) if memory.senescent else 4
        spawn = bool(spawn and now - memory.last_birth > birth_interval and nearest_predator > 60
                     and energy > 100 + control_cost)
        if spawn:
            memory.last_birth = now
            if self.tuning.get("renewal") and (age > (115 if self.tuning.get("late_renewal") else 65) or memory.senescent):
                memory.retired = True
        memory.last_energy = energy
        memory.last_cost = control_cost + 100 * spawn + 0.1
        memory.last_x, memory.last_y = x, y
        memory.last_distance = distance * modifier
        memory.move_heading = desired
        memory.x += memory.last_distance * math.cos(desired)
        memory.y += memory.last_distance * math.sin(desired)
        memory.heading = wrap(heading + turn)
        if distance > 0 and not danger and target is None:
            memory.travel_heading = desired
        return {
            "agent_id": status["agent_id"],
            "move_distance": float(distance),
            "move_direction": wrap(desired - heading),
            "turn_angle": float(turn),
            "spawn_agent": spawn,
        }
