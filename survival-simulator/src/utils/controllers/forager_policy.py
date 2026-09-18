"""Hivemind forager: a shared world map with coordinated foraging and exploration.

Agents localize in the world frame from the boundary walls (or from a localized
neighbor), then share one map of walls, fruit, trees and recently observed
ground. Each tick the colony assigns fruit and camp sites without duplicate
claims and sends the remaining agents to explore stale ground.
"""
from collections import Counter
from dataclasses import dataclass, field
import heapq
import logging
import math

from src.utils.controllers.predator_avoidance import (
    crosses,
    maneuver_move,
    predict_predator,
    segment_distance,
    touches_edge,
    wrap,
)
from src.utils.controllers.world_memory import neighbor_heading


logger = logging.getLogger(__name__)
TAU = 2 * math.pi
WIDTH, HEIGHT, WALL = 1600.0, 1200.0, 30.0
BIOME_SPEED = {"forest": 1.0, "grassland": 1.0, "desert": 0.8, "swamp": 0.5, "river": 0.3}
TREE_PRIOR = {"forest": 1.0, "grassland": 0.6, "swamp": 0.8, "desert": 0.15, "river": 0.0}
BOUNDARIES = (
    (0.0, WALL, WIDTH, WALL),
    (0.0, HEIGHT - WALL, WIDTH, HEIGHT - WALL),
    (WALL, 0.0, WALL, HEIGHT),
    (WIDTH - WALL, 0.0, WIDTH - WALL, HEIGHT),
)
CELL = 40.0
COLUMNS, ROWS = int(WIDTH // CELL), int(HEIGHT // CELL)
REGION = 2  # exploration regions are REGION x REGION cells
FRUIT_RIPE = 20.0
FRUIT_ROT = 50.0
NEVER = -1e9

DEFAULT_SETTINGS = {
    "population": 20,
    "min_population": 4,
    "max_population": 24,
    "half_life": 900.0,
    "birth_energy": 260.0,
    "old_birth_margin": 8,
    "early_birth_age": 45.0,
    "early_birth_energy": 180.0,
    "early_birth_margin": 2,
    "time_cost": 4.0,
    "commitment": 20.0,
    "hungry_override": 0.0,
    "birth_food_check": False,
    "wait_cost": None,
    "old_rest": False,
    "fruit_range": 400.0,
    "explore_spin": 0.6,
    "camp_scan": 1.05,
    "camp_scan_period": 0.5,
    "travel_spin": 0.5,
    "watch_range": 180.0,
    "flight_range": 170.0,
    "flight_angle": 0.6,
    "unseen_range": 100.0,
    "approach_only": False,
    "approach_angle": 1.0,
    "near_range": 45.0,
    "killer_memory": 0.0,
    "target_danger": 40.0,
    "explore_danger": 75.0,
    "steer_predators": True,
    "killer_range": 150.0,
    "stale_cap": 90.0,
    "tree_range": 700.0,
    "food_attraction": 1.0,
    "hunger_urgency": 1.0,
    "camp_threshold": 5.0,
    "selection": 0.0,
    "selection_gap": 0.3,
    "elite_select": False,
    "elite_bonus": 60.0,
}


def edge_key(edge):
    return round(edge[2] - edge[0], 4), round(edge[3] - edge[1], 4)


class EdgeMap:
    """Wall segments with lookups by exact direction (for localization) and by area."""

    SIZE = 100.0

    def __init__(self, listener=None):
        self.by_key = {}
        self.grid = {}
        self.listener = listener

    def add(self, edge):
        known = self.by_key.setdefault(edge_key(edge), [])
        if any(math.hypot(e[0] - edge[0], e[1] - edge[1]) < 3 for e in known):
            return False
        known.append(edge)
        x0, x1 = sorted((edge[0], edge[2]))
        y0, y1 = sorted((edge[1], edge[3]))
        for cx in range(int(x0 // self.SIZE), int(x1 // self.SIZE) + 1):
            for cy in range(int(y0 // self.SIZE), int(y1 // self.SIZE) + 1):
                self.grid.setdefault((cx, cy), []).append(edge)
        if self.listener is not None:
            self.listener(edge)
        return True

    def lookup(self, key):
        return self.by_key.get(key, ())

    def box(self, x0, y0, x1, y1):
        found = {}
        for cx in range(int(x0 // self.SIZE), int(x1 // self.SIZE) + 1):
            for cy in range(int(y0 // self.SIZE), int(y1 // self.SIZE) + 1):
                for edge in self.grid.get((cx, cy), ()):
                    found[id(edge)] = edge
        return list(found.values())

    def near(self, x, y, radius):
        return self.box(x - radius, y - radius, x + radius, y + radius)

    def all(self):
        return [edge for group in self.by_key.values() for edge in group]


class Item:
    """A remembered fruit or tree in world coordinates."""

    __slots__ = ("x", "y", "first", "seen", "born_lo", "born_hi", "fruits", "last_fruit", "blocked")

    def __init__(self, x, y, now, born_lo, born_hi):
        self.x, self.y = x, y
        self.first = self.seen = now
        self.born_lo, self.born_hi = born_lo, born_hi
        self.fruits = 0
        self.last_fruit = NEVER
        self.blocked = {}


class Registry:
    """Spatial hash of items; items never move, so matching is by position."""

    SIZE = 100.0

    def __init__(self):
        self.cells = {}

    def __iter__(self):
        for items in self.cells.values():
            yield from items

    def __len__(self):
        return sum(len(items) for items in self.cells.values())

    def match(self, x, y, tolerance=4.0):
        cx, cy = int(x // self.SIZE), int(y // self.SIZE)
        best, best_distance = None, tolerance
        for ix in (cx - 1, cx, cx + 1):
            for iy in (cy - 1, cy, cy + 1):
                for item in self.cells.get((ix, iy), ()):
                    distance = math.hypot(item.x - x, item.y - y)
                    if distance < best_distance:
                        best, best_distance = item, distance
        return best

    def add(self, item):
        self.cells.setdefault((int(item.x // self.SIZE), int(item.y // self.SIZE)), []).append(item)

    def remove(self, item):
        items = self.cells.get((int(item.x // self.SIZE), int(item.y // self.SIZE)))
        if items is not None and item in items:
            items.remove(item)

    def near(self, x, y, radius):
        found = []
        if radius > 250:
            for items in self.cells.values():
                for item in items:
                    if abs(item.x - x) <= radius and abs(item.y - y) <= radius and math.hypot(item.x - x, item.y - y) <= radius:
                        found.append(item)
            return found
        for cx in range(int((x - radius) // self.SIZE), int((x + radius) // self.SIZE) + 1):
            for cy in range(int((y - radius) // self.SIZE), int((y + radius) // self.SIZE) + 1):
                for item in self.cells.get((cx, cy), ()):
                    if math.hypot(item.x - x, item.y - y) <= radius:
                        found.append(item)
        return found

    def prune(self, keep):
        for key in list(self.cells):
            self.cells[key] = [item for item in self.cells[key] if keep(item)]


class Planner:
    """Grid A* over the shared wall map; cells near known walls are blocked."""

    STEP = 10.0
    MARGIN = 7.5
    MOVES = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))

    def __init__(self):
        self.columns = int(WIDTH // self.STEP) + 1
        self.rows = int(HEIGHT // self.STEP) + 1
        self.blocked = bytearray(self.columns * self.rows)

    def add(self, edge):
        ax, ay, bx, by = edge
        margin = self.MARGIN
        x0 = max(0, int((min(ax, bx) - margin) // self.STEP))
        x1 = min(self.columns - 1, int((max(ax, bx) + margin) // self.STEP) + 1)
        y0 = max(0, int((min(ay, by) - margin) // self.STEP))
        y1 = min(self.rows - 1, int((max(ay, by) + margin) // self.STEP) + 1)
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                if touches_edge(cx * self.STEP, cy * self.STEP, edge, margin):
                    self.blocked[cy * self.columns + cx] = 1

    def free(self, cx, cy):
        return 0 <= cx < self.columns and 0 <= cy < self.rows and not self.blocked[cy * self.columns + cx]

    def search(self, x, y, gx, gy, reach, limit=5000):
        """Cell-center waypoints from near (x, y) to within `reach` of (gx, gy), or None."""
        step = self.STEP
        sx, sy = round(x / step), round(y / step)
        starts = [(sx + dx, sy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if self.free(sx + dx, sy + dy)]
        if not starts:
            return None
        ex, ey = gx / step, gy / step
        reach_cells = reach / step
        columns = self.columns
        blocked = self.blocked
        frontier = []
        costs = {}
        parents = {}
        for cell in starts:
            cost = math.hypot(cell[0] * step - x, cell[1] * step - y) / step
            costs[cell] = cost
            heapq.heappush(frontier, (cost + math.hypot(cell[0] - ex, cell[1] - ey), cost, cell))
        reached = None
        expanded = 0
        while frontier and expanded < limit:
            _, cost, cell = heapq.heappop(frontier)
            if cost > costs[cell]:
                continue
            expanded += 1
            cx, cy = cell
            if math.hypot(cx - ex, cy - ey) <= reach_cells:
                reached = cell
                break
            for dx, dy, length in self.MOVES:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < columns and 0 <= ny < self.rows) or blocked[ny * columns + nx]:
                    continue
                if dx and dy and (blocked[cy * columns + nx] or blocked[ny * columns + cx]):
                    continue
                new_cost = cost + length
                if new_cost < costs.get((nx, ny), 1e18):
                    costs[(nx, ny)] = new_cost
                    parents[(nx, ny)] = cell
                    heuristic = max(0.0, math.hypot(nx - ex, ny - ey) - reach_cells)
                    heapq.heappush(frontier, (new_cost + heuristic, new_cost, (nx, ny)))
        if reached is None:
            return None
        path = []
        while reached is not None:
            path.append((reached[0] * step, reached[1] * step))
            reached = parents.get(reached)
        path.reverse()
        return path


@dataclass
class Agent:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    localized: bool = False
    edges: EdgeMap = field(default_factory=EdgeMap)
    landmarks: list = field(default_factory=list)
    last_age: float = -1.0
    last_distance: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0
    move_heading: float = 0.0
    stuck: int = 0
    last_energy: float = 0.0
    last_cost: float = 0.0
    senescent: bool = False
    last_birth: float = -1000.0
    predators: list = field(default_factory=list)
    target: object = None
    target_kind: str = ""
    target_since: float = 0.0
    target_best: float = 0.0
    goal: tuple | None = None
    path: list | None = None
    path_goal: tuple | None = None
    path_time: float = -1000.0
    bad_goals: dict = field(default_factory=dict)
    goal_since: float = -1000.0
    last_scan: float = -1000.0
    scan_left: float = 0.0
    wander: float = 0.0
    fresh: bool = False
    agent_id: int = -1
    mode: str = ""
    objects: dict = field(default_factory=dict)
    seen_edges: list = field(default_factory=list)

    def parse(self, status):
        c, s = math.cos(self.heading), math.sin(self.heading)
        edges = {}
        raw = set()
        objects = {"Fruit": [], "Tree": [], "Agent": [], "Predator": []}
        for obs in status["observations"]:
            kind = obs.get("type")
            if kind == "Edge":
                (ax, ay), (bx, by) = obs["coords"]
                if (ax, ay, bx, by) in raw:
                    continue
                raw.add((ax, ay, bx, by))
                edge = (c * ax - s * ay, s * ax + c * ay, c * bx - s * by, s * bx + c * by)
                edges[(edge_key(edge), edge)] = None
            elif kind in objects:
                angle = self.heading + obs["angle"]
                objects[kind].append((obs["distance"] * math.cos(angle), obs["distance"] * math.sin(angle), obs))
        return list(edges), objects

    def correct(self, local_edges, objects):
        """Correct odometry by matching static features against the map (after moving only)."""
        static = [(kind, x, y) for kind in ("Fruit", "Tree") for x, y, _ in objects[kind]]
        if self.last_distance > 1e-6:
            shifts = []
            limit = max(25.0, self.last_distance * 2 + 2)
            for key, edge in local_edges:
                for known in self.edges.lookup(key):
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
        for _, edge in local_edges:
            self.edges.add((edge[0] + self.x, edge[1] + self.y, edge[2] + self.x, edge[3] + self.y))
        self.landmarks = [(kind, x + self.x, y + self.y) for kind, x, y in static]

    def boundary_frame(self, local_edges):
        """World (rotation, x, y) of this agent from a visible boundary wall, if any."""
        for _, (ax, ay, bx, by) in local_edges:
            length = math.hypot(bx - ax, by - ay)
            horizontal = abs(length - WIDTH) < 0.05
            if not horizontal and abs(length - HEIGHT) < 0.05:
                pass
            elif not horizontal:
                continue
            rotation = (0.0 if horizontal else math.pi / 2) - math.atan2(by - ay, bx - ax)
            c, s = math.cos(rotation), math.sin(rotation)
            sx, sy = c * ax - s * ay, s * ax + c * ay
            if horizontal:
                return rotation, -sx, (WALL if sy < 0 else HEIGHT - WALL) - sy
            return rotation, (WALL if sx < 0 else WIDTH - WALL) - sx, -sy
        return None

    def rebase(self, rotation, wx, wy, edges):
        """Move this agent's private frame into the world frame and adopt the shared wall map."""
        c, s = math.cos(rotation), math.sin(rotation)
        ox, oy = self.x, self.y

        def transform(x, y):
            dx, dy = x - ox, y - oy
            return wx + c * dx - s * dy, wy + s * dx + c * dy

        for ax, ay, bx, by in self.edges.all():
            edges.add((*transform(ax, ay), *transform(bx, by)))
        self.landmarks = [(kind, *transform(x, y)) for kind, x, y in self.landmarks]
        self.predators = [(*transform(x, y), when) for x, y, when in self.predators]
        self.last_x, self.last_y = transform(self.last_x, self.last_y)
        self.x, self.y = wx, wy
        self.heading = wrap(self.heading + rotation)
        self.move_heading = wrap(self.move_heading + rotation)
        self.wander = wrap(self.wander + rotation)
        self.edges = edges
        self.localized = True


class ForagerPolicy:
    def __init__(self, settings=None):
        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            unknown = set(settings) - set(DEFAULT_SETTINGS)
            if unknown:
                raise ValueError(f"unknown settings: {sorted(unknown)}")
            self.settings.update(settings)
        self.reset()

    def reset(self):
        self.agents = {}
        self.statuses = {}
        self.planner = Planner()
        self.edges = EdgeMap(self.planner.add)
        for edge in BOUNDARIES:
            self.edges.add(edge)
        self.fruits = Registry()
        self.trees = Registry()
        self.coverage = [NEVER] * (COLUMNS * ROWS)
        self.biomes = {}
        self.sightings = {}
        self.predators_now = []
        self.killers = []
        self.previous_predators = []
        self.last_seen_agents = {}
        self.danger = {}
        self.explore_base = None
        self.last_time = None
        self.cached_state = None
        self.cached_actions = []

    # ------------------------------------------------------------------ tick
    def decide(self, statuses, sim_time):
        try:
            return self._decide(statuses, sim_time)
        except Exception:  # a failed response would end the run; forget state and hold still instead
            logger.exception("decision failed at t=%s; resetting controller state", sim_time)
            self.reset()
            return [{"agent_id": status["agent_id"], "move_distance": 0.0, "move_direction": 0.0,
                     "turn_angle": 0.0, "spawn_agent": False} for status in statuses]

    def _decide(self, statuses, sim_time):
        if self.last_time == sim_time and statuses == self.cached_state:
            return [dict(action) for action in self.cached_actions]
        if self.last_time is not None and (sim_time <= self.last_time or not statuses):
            self.reset()
        now = sim_time
        self.statuses = {s["agent_id"]: s for s in statuses}
        self.agents = {key: agent for key, agent in self.agents.items() if key in self.statuses}
        previous = {a["agent_id"]: a for a in self.cached_actions}
        for status in statuses:
            agent_id = status["agent_id"]
            agent = self.agents.get(agent_id)
            if agent is None:
                agent = self.agents[agent_id] = Agent(agent_id=agent_id)
            agent.fresh = not (agent.last_age == status["age"] and agent_id in previous)
            if not agent.fresh:
                continue
            if agent.last_age >= 0 and status["age"] > 59 and not agent.senescent:
                if agent.last_energy - status["energy"] - agent.last_cost > status["age"] * 0.005:
                    agent.senescent = True
            agent.last_age = status["age"]
            local_edges, agent.objects = agent.parse(status)
            agent.seen_edges = [edge for _, edge in local_edges]
            agent.correct(local_edges, agent.objects)
            frame = agent.boundary_frame(local_edges)
            if frame is not None and not agent.localized:
                agent.rebase(*frame, self.edges)
        self._inherit_frames()
        self._update_world(now)
        self._track_killers(now)
        actions = self._plan(statuses, previous, now)
        self.last_time = sim_time
        self.cached_state = statuses
        self.cached_actions = actions
        return [dict(action) for action in actions]

    def _inherit_frames(self):
        for _ in range(2):
            for agent_id, agent in self.agents.items():
                if agent.localized or not agent.fresh:
                    continue
                for px, py, obs in agent.objects["Agent"]:
                    neighbor = self.agents.get(obs.get("id"))
                    if neighbor is None or not neighbor.localized:
                        continue
                    rotation = neighbor.heading - neighbor_heading(agent, px, py, obs)
                    c, s = math.cos(rotation), math.sin(rotation)
                    wx = neighbor.x - (c * px - s * py)
                    wy = neighbor.y - (s * px + c * py)
                    agent.rebase(rotation, wx, wy, self.edges)
                    break

    # ------------------------------------------------------------- world map
    def _visible(self, agent, status, x, y, margin=0.0):
        """Whether a point would certainly be sensed by this agent now."""
        dx, dy = x - agent.x, y - agent.y
        distance = math.hypot(dx, dy)
        if distance < status["hearing_radius"] - 2 - margin:
            return True
        if distance >= status["vision_range"] - 3 - margin or distance < 1e-6:
            return False
        slack = math.asin(min(1.0, margin / distance)) if margin else 0.0
        if abs(wrap(math.atan2(dy, dx) - agent.heading)) >= status["vision_angle"] / 2 - 0.01 - slack:
            return False
        for edge in agent.seen_edges:
            ex = (edge[0] + agent.x, edge[1] + agent.y, edge[2] + agent.x, edge[3] + agent.y)
            if crosses(agent.x, agent.y, x, y, ex) or segment_distance(x, y, ex) < 4 + margin:
                return False
        return True

    def _born(self, x, y, now, lifetime):
        cx, cy = int(x // CELL), int(y // CELL)
        if not (0 <= cx < COLUMNS and 0 <= cy < ROWS):
            return now - lifetime, now
        last = self.coverage[cy * COLUMNS + cx]
        return max(last, now - lifetime), now

    def _update_world(self, now):
        observers = [(self.agents[i], self.statuses[i]) for i in self.agents
                     if self.agents[i].localized and self.agents[i].fresh]
        seen = set()
        previous = self.predators_now
        self.predators_now = []
        for agent, status in observers:
            for px, py, obs in agent.objects["Predator"]:
                wx, wy = agent.x + px, agent.y + py
                if any(abs(p[0] - wx) < 3 and abs(p[1] - wy) < 3 for p in self.predators_now):
                    continue
                last = min(previous, key=lambda p: math.hypot(p[0] - wx, p[1] - wy), default=None)
                shift = math.hypot(last[0] - wx, last[1] - wy) if last is not None else 1e9
                active = shift > 0.5
                modifier = 1.0
                if shift < 25:
                    modifier = max(0.3, min(1.0, shift / (15 if obs["distance"] < 90 else 11)))
                self.predators_now.append((wx, wy, wrap(neighbor_heading(agent, px, py, obs)), active, modifier))
        for agent, status in observers:
            ix, iy = int(agent.x // CELL), int(agent.y // CELL)
            self.biomes[(ix, iy)] = status["biome"]
            for kind, registry, lifetime in (("Fruit", self.fruits, FRUIT_ROT), ("Tree", self.trees, 100.0)):
                for px, py, _ in agent.objects[kind]:
                    wx, wy = agent.x + px, agent.y + py
                    item = registry.match(wx, wy)
                    if item is None:
                        item = Item(wx, wy, now, *self._born(wx, wy, now, lifetime))
                        registry.add(item)
                        if kind == "Fruit" and item.born_hi - item.born_lo < 3:
                            for tree in self.trees.near(wx, wy, 65):
                                tree.fruits += 1
                                tree.last_fruit = now
                                tree.born_hi = min(tree.born_hi, now - 20)
                                tree.born_lo = min(tree.born_lo, tree.born_hi)
                    item.seen = now
                    seen.add(id(item))
            for px, py, obs in agent.objects["Predator"]:
                wx, wy = agent.x + px, agent.y + py
                self.sightings[(int(wx // CELL), int(wy // CELL))] = (wx, wy, now)
        for agent, status in observers:
            radius = max(status["hearing_radius"], status["vision_range"])
            for registry in (self.fruits, self.trees):
                for item in registry.near(agent.x, agent.y, radius):
                    if id(item) not in seen and self._visible(agent, status, item.x, item.y):
                        registry.remove(item)
        self.fruits.prune(lambda f: now - f.born_lo < FRUIT_ROT + 1 and now - f.seen < 60)
        self.trees.prune(lambda t: now - t.born_lo < 110 and now - t.seen < 120)
        self.sightings = {key: value for key, value in self.sightings.items() if now - value[2] < 2.5}
        self.danger = {}
        for cx, cy in self.sightings:
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    if abs(dx) + abs(dy) < 4:
                        self.danger[(cx + dx, cy + dy)] = 1
        self.explore_base = None
        for agent, status in observers:
            self._cover(agent, status, now)

    def _track_killers(self, now):
        """Predators that just ate an agent tend to kill again soon; remember and follow them."""
        s = self.settings
        predators = self.predators_now
        for agent_id, (x, y, energy) in self.last_seen_agents.items():
            if agent_id in self.statuses or not s["killer_memory"]:
                continue
            # The killer is usually still in view; otherwise fall back to last tick's sightings (the victim's own).
            for candidates, radius, seen in ((predators, 45.0, now), (self.previous_predators, 70.0, now - 0.1)):
                nearest = min(candidates, key=lambda p: math.hypot(p[0] - x, p[1] - y), default=None)
                if nearest is None:
                    continue
                gap = math.hypot(nearest[0] - x, nearest[1] - y)
                if gap < radius and (energy > 1.5 or gap < radius - 20):
                    self.killers.append([nearest[0], nearest[1], seen, now + s["killer_memory"]])
                    break
        self.previous_predators = predators
        tracked = []
        for killer in self.killers:
            reach = 1.6 * (now - killer[2]) * 10 * 15 + 25
            nearest = min(predators, key=lambda p: math.hypot(p[0] - killer[0], p[1] - killer[1]), default=None)
            if nearest is not None and math.hypot(nearest[0] - killer[0], nearest[1] - killer[1]) < reach:
                killer[0], killer[1], killer[2] = nearest[0], nearest[1], now
            if now < killer[3] and now - killer[2] < 15 and not any(
                    math.hypot(k[0] - killer[0], k[1] - killer[1]) < 10 for k in tracked):
                tracked.append(killer)
        self.killers = tracked
        marked = [(k[0], k[1]) for k in tracked if k[2] == now]
        self.predators_now = [p[:5] + (any(abs(p[0] - kx) < 1 and abs(p[1] - ky) < 1 for kx, ky in marked),)
                              for p in predators]
        for kx, ky in marked:
            cx, cy = int(kx // CELL), int(ky // CELL)
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx * dx + dy * dy <= 26:
                        self.danger[(cx + dx, cy + dy)] = 1
        self.last_seen_agents = {agent_id: (agent.x, agent.y, self.statuses[agent_id]["energy"])
                                 for agent_id, agent in self.agents.items() if agent.localized}

    def _cover(self, agent, status, now):
        half = CELL * 0.7072
        hearing = status["hearing_radius"] - 2 - half
        vision = status["vision_range"] - 3 - half
        cone = status["vision_angle"] / 2 - 0.01
        if hearing > 0:
            self._cover_box(agent, agent.x - hearing, agent.y - hearing, agent.x + hearing, agent.y + hearing,
                            lambda px, py, dx, dy, d: d < hearing, now)
        if vision <= half or cone <= 0:
            return
        points = [(agent.x, agent.y)]
        for angle in (agent.heading - cone, agent.heading + cone, agent.heading):
            points.append((agent.x + vision * math.cos(angle), agent.y + vision * math.sin(angle)))
        for axis in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            if abs(wrap(axis - agent.heading)) < cone:
                points.append((agent.x + vision * math.cos(axis), agent.y + vision * math.sin(axis)))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        edges = [(e[0] + agent.x, e[1] + agent.y, e[2] + agent.x, e[3] + agent.y) for e in agent.seen_edges]
        heading = agent.heading

        def visible(px, py, dx, dy, distance):
            if distance >= vision or distance < half:
                return False
            if abs(wrap(math.atan2(dy, dx) - heading)) >= cone - math.asin(half / distance):
                return False
            for edge in edges:
                if crosses(agent.x, agent.y, px, py, edge) or segment_distance(px, py, edge) < half:
                    return False
            return True

        self._cover_box(agent, min(xs), min(ys), max(xs), max(ys), visible, now)

    def _cover_box(self, agent, x0, y0, x1, y1, test, now):
        coverage = self.coverage
        cx0, cx1 = max(0, int(x0 // CELL)), min(COLUMNS - 1, int(x1 // CELL))
        cy0, cy1 = max(0, int(y0 // CELL)), min(ROWS - 1, int(y1 // CELL))
        for cy in range(cy0, cy1 + 1):
            py = (cy + 0.5) * CELL
            for cx in range(cx0, cx1 + 1):
                index = cy * COLUMNS + cx
                if coverage[index] == now:
                    continue
                px = (cx + 0.5) * CELL
                dx, dy = px - agent.x, py - agent.y
                if test(px, py, dx, dy, math.hypot(dx, dy)):
                    coverage[index] = now

    # -------------------------------------------------------------- planning
    def _target_population(self, now):
        s = self.settings
        target = math.ceil(s["population"] * 0.5 ** (now / s["half_life"]))
        return max(s["min_population"], min(s["max_population"], target))

    @staticmethod
    def _fruit_value(fruit, arrival):
        """Energy expected when eating at `arrival`, the wait until it is likely ripe, and survival odds."""
        youngest = arrival - fruit.born_hi
        oldest = arrival - fruit.born_lo
        if oldest > FRUIT_ROT:
            alive = max(0.0, (FRUIT_ROT - youngest) / max(1.0, oldest - youngest))
        else:
            alive = 1.0
        middle = (youngest + oldest) / 2
        return min(60.0, 20 + 2 * max(0.0, middle)), max(0.0, FRUIT_RIPE - middle), alive

    def _plan(self, statuses, previous, now):
        population = len(statuses)
        target_population = self._target_population(now)
        self.mean_fitness = sum(self._fitness(st) for st in statuses) / max(1, population)
        self.best_walk = max((min(st["speed"], st["sprint_speed"]) for st in statuses), default=0.0)
        s = self.settings
        order = sorted(
            (status for status in statuses if self.agents[status["agent_id"]].fresh),
            key=lambda st: (self.agents[st["agent_id"]].target is None, st["energy"], st["agent_id"]),
        )
        # Current targets stay claimed by their holders; only the holder may give one up.
        claimed = {id(agent.target): agent for agent in self.agents.values() if agent.target is not None}
        for status in order:
            agent = self.agents[status["agent_id"]]
            if agent.localized:
                self._choose_fruit(agent, status, now, claimed)
        for status in order:
            agent = self.agents[status["agent_id"]]
            if agent.localized and agent.target_kind != "fruit":
                self._choose_tree(agent, status, now, claimed)
        goals = [a.goal for a in self.agents.values() if a.goal is not None]
        actions = []
        for status in statuses:
            agent = self.agents[status["agent_id"]]
            if not agent.fresh:
                action = dict(previous[status["agent_id"]])
                action.update(turn_angle=0.0, spawn_agent=False,
                              move_direction=wrap(agent.move_heading - agent.heading))
                if status["energy"] < status["max_energy"] / 5:
                    action["move_distance"] = min(action["move_distance"], status["speed"])
                self._advance(agent, status, action, now, stale=True)
                actions.append(action)
                continue
            try:
                action = self._act(agent, status, now, population, target_population, goals)
            except Exception:  # never let one agent's planning failure drop the whole response
                logger.exception("planning failed for agent %s; holding still", status["agent_id"])
                action = self._hold(agent, status, now)
            population += int(action["spawn_agent"])
            actions.append(action)
        return actions

    def _travel(self, agent, x, y):
        """Effective walking distance to (x, y): known slow terrain and walls make it longer."""
        distance = math.hypot(x - agent.x, y - agent.y)
        samples = int(distance // CELL) + 1
        slowness = 0.0
        for i in range(samples):
            t = (i + 0.5) / samples
            px, py = agent.x + (x - agent.x) * t, agent.y + (y - agent.y) * t
            slowness += 1.0 / BIOME_SPEED.get(self.biomes.get((int(px // CELL), int(py // CELL))), 1.0)
        effective = distance * slowness / samples
        if not self._clear(agent.x, agent.y, x, y, margin=0.0):
            effective *= 1.3
        return effective

    def _choose(self, agent, status, now, claimed, registry, kind, radius, utility_of, threshold):
        """Pick the unclaimed item with the best utility; exact travel costs only for the top few."""
        s = self.settings
        energy = status["energy"]
        walk = max(0.5, min(status["speed"], status["sprint_speed"]))
        time_cost = s["time_cost"] * (1 + s["hunger_urgency"] * max(0.0, 60 - energy) / 30)
        rough = []
        for item in registry.near(agent.x, agent.y, radius):
            holder = claimed.get(id(item), agent)
            if holder is not agent:
                # A hungry agent may take food from a much better-fed one.
                other = self.statuses.get(holder.agent_id)
                if not (energy < s["hungry_override"] and other is not None
                        and other["energy"] > energy + 100):
                    continue
            if item.blocked.get(id(agent), -100) > now:
                continue
            distance = math.hypot(item.x - agent.x, item.y - agent.y)
            bonus = s["commitment"] if item is agent.target else 0.0
            rough.append((utility_of(item, distance, walk, time_cost) + bonus, id(item), item))
        rough.sort(reverse=True)
        best, best_utility = None, threshold
        for estimate, _, item in rough[:4]:
            if estimate <= best_utility:
                break
            distance = self._travel(agent, item.x, item.y)
            if 0.05 * distance + 0.1 * distance / walk + 3 > energy:
                continue
            utility = utility_of(item, distance, walk, time_cost)
            if item is agent.target:
                utility += s["commitment"]
            utility -= s["target_danger"] * self._danger_at(item.x, item.y)
            if utility > best_utility:
                best, best_utility = item, utility
        if best is not None or agent.target_kind == kind:
            if agent.target is not None and claimed.get(id(agent.target)) is agent and agent.target is not best:
                del claimed[id(agent.target)]
            if best is not None:
                holder = claimed.get(id(best))
                if holder is not None and holder is not agent:
                    holder.target, holder.target_kind = None, ""
            self._set_target(agent, best, kind if best is not None else "", now)
            if best is not None:
                claimed[id(best)] = agent
        return best

    def _choose_fruit(self, agent, status, now, claimed):
        energy, capacity = status["energy"], status["max_energy"] - status["energy"]
        hungry = energy < 45

        def utility_of(fruit, distance, walk, time_cost):
            travel = distance / walk * 0.1
            value, wait, alive = self._fruit_value(fruit, now + travel)
            if hungry:
                wait = 0.0
            elif wait > 0:
                value = 60.0
                alive = self._fruit_value(fruit, now + travel + wait)[2]
            wait_cost = time_cost if self.settings["wait_cost"] is None else self.settings["wait_cost"]
            return min(value, capacity + 5) * alive - distance * 0.05 - time_cost * travel - wait_cost * wait

        # Aging agents burn energy fast: they leave fruit to the young unless they need it for a birth.
        threshold = 1e9 if agent.senescent and energy > 115 else 4.0
        return self._choose(agent, status, now, claimed, self.fruits, "fruit", self.settings["fruit_range"],
                            utility_of, threshold)

    def _choose_tree(self, agent, status, now, claimed):
        def utility_of(tree, distance, walk, time_cost):
            travel = distance / walk * 0.1
            return self._tree_output(tree, now + travel) - distance * 0.05 - time_cost * travel

        return self._choose(agent, status, now, claimed, self.trees, "tree", self.settings["tree_range"],
                            utility_of, self.settings["camp_threshold"])

    def _clear(self, x, y, gx, gy, margin=5.5):
        edges = self.edges.box(min(x, gx) - margin, min(y, gy) - margin, max(x, gx) + margin, max(y, gy) + margin)
        segment = (x, y, gx, gy)
        for edge in edges:
            if crosses(x, y, gx, gy, edge):
                return False
            if margin and (touches_edge(gx, gy, edge, margin) or segment_distance(edge[0], edge[1], segment) < margin
                           or segment_distance(edge[2], edge[3], segment) < margin):
                return False
        return True

    def _route(self, agent, gx, gy, reach, now):
        """Next point to walk towards on the way to (gx, gy); None when no known route exists."""
        x, y = agent.x, agent.y
        if self._clear(x, y, gx, gy):
            agent.path = None
            return gx, gy
        if (agent.path is None or agent.path_goal is None or agent.stuck > 3
                or math.hypot(gx - agent.path_goal[0], gy - agent.path_goal[1]) > 10 or now - agent.path_time > 3):
            agent.path = self.planner.search(x, y, gx, gy, reach)
            agent.path_goal, agent.path_time = (gx, gy), now
            if agent.path is None:
                return None
        path = agent.path
        while len(path) > 1 and math.hypot(path[0][0] - x, path[0][1] - y) < 8:
            path.pop(0)
        choice = path[0]
        for waypoint in path[1:16]:
            if math.hypot(waypoint[0] - x, waypoint[1] - y) > 150:
                break
            if self._clear(x, y, waypoint[0], waypoint[1]):
                choice = waypoint
        if len(path) == 1 and math.hypot(choice[0] - x, choice[1] - y) < 8:
            return gx, gy
        return choice

    @staticmethod
    def _tree_output(tree, arrival):
        """Expected fruit energy produced after `arrival` over the next minute."""
        youngest = arrival - tree.born_hi
        oldest = arrival - tree.born_lo
        if oldest - youngest < 25:
            age = (youngest + oldest) / 2
            start, end = max(age, 20.0), min(age + 60.0, 58.0)
            window = max(0.0, end - start)
            return 6.0 * window * 0.8
        if tree.last_fruit > NEVER and arrival - tree.last_fruit < 25:
            return 6.0 * 20 * 0.6
        return 6.0 * 20 * 0.35

    @staticmethod
    def _set_target(agent, target, kind, now):
        if target is not agent.target:
            agent.target_since = now
            agent.target_best = float("inf")
        agent.target, agent.target_kind = target, kind

    def _danger_at(self, x, y):
        return self.danger.get((int(x // CELL), int(y // CELL)), 0)

    # ----------------------------------------------------------------- acting
    def _act(self, agent, status, now, population, target_population, goals):
        s = self.settings
        x, y, heading = agent.x, agent.y, agent.heading
        energy, age = status["energy"], status["age"]
        modifier = BIOME_SPEED.get(status["biome"], 1.0)
        walk = max(0.0, min(status["speed"], status["sprint_speed"]))
        sprint = status["sprint_speed"] if energy > status["max_energy"] / 5 + 8 else walk
        reach = max(walk, status["sprint_speed"]) * 2 + 40
        edges = self.edges.near(x, y, reach) if agent.localized else agent.edges.near(x, y, reach)

        # --- predators: (dx, dy, distance) relative to this agent
        if agent.localized:
            reach = max(180.0, s["killer_range"])
            sensed = [(p[0] - x, p[1] - y, p[2], p[3], p[4], p[5]) for p in self.predators_now
                      if abs(p[0] - x) < reach and abs(p[1] - y) < reach]
        else:
            sensed = []
            for px, py, obs in agent.objects["Predator"]:
                last = min(agent.predators, key=lambda p: math.hypot(p[0] - x - px, p[1] - y - py), default=None)
                active = last is None or now - last[2] > 0.15 or math.hypot(last[0] - x - px, last[1] - y - py) > 0.5
                pred_modifier = modifier
                if last is not None and now - last[2] < 0.15 and active:
                    shift = math.hypot(last[0] - x - px, last[1] - y - py)
                    pred_modifier = max(0.3, min(1.0, shift / (15 if obs["distance"] < 90 else 11)))
                sensed.append((px, py, wrap(neighbor_heading(agent, px, py, obs)), active, pred_modifier, False))
            agent.predators = [(x + px, y + py, now) for px, py, *_ in sensed]
        predators, threats, triggers = [], [], []
        approaching = False
        for px, py, pred_heading, active, pred_modifier, killer in sensed:
            gap = math.hypot(px, py)
            if killer and active and gap < s["killer_range"]:
                approaching = True
            if gap >= (max(180.0, s["killer_range"]) if killer else 180.0):
                continue
            predators.append((px, py, gap))
            if (active and gap < s["flight_range"]
                    and abs(wrap(math.atan2(-py, -px) - pred_heading)) < s["flight_angle"]
                    and (not agent.localized or self._clear(x + px, y + py, x, y, margin=0.0))):
                approaching = True
            # Awake predators heading our way are threats from further out than ones moving away.
            toward = abs(wrap(math.atan2(-py, -px) - pred_heading)) < s["approach_angle"]
            if not active:
                trigger = 35.0
            elif toward or not s["approach_only"]:
                trigger = s["unseen_range"]
            else:
                trigger = s["near_range"]
            if active:
                px, py, pred_heading = predict_predator(px, py, pred_heading, 0, 0, heading, pred_modifier)
            threats.append((px, py, pred_heading, active, pred_modifier))
            triggers.append(trigger)
        nearest_predator = min((gap for _, _, gap in predators), default=1000.0)
        danger = approaching or any(math.hypot(threat[0], threat[1]) < trigger
                                    for threat, trigger in zip(threats, triggers))

        # --- movement goal
        distance, desired, mode = 0.0, agent.move_heading, "rest"
        stop = 0.0
        goal = None
        if not agent.localized:
            fruits = [(px, py) for px, py, _ in agent.objects["Fruit"]]
            if fruits:
                px, py = min(fruits, key=lambda p: math.hypot(*p))
                desired, distance, mode = math.atan2(py, px), min(walk, math.hypot(px, py) / max(modifier, 0.1)), "fruit"
            elif now > 1.0:
                desired, distance, mode = agent.wander, walk, "explore"
            else:
                mode = "scan"
        elif agent.target_kind in ("fruit", "tree"):
            item = agent.target
            remaining = math.hypot(item.x - x, item.y - y)
            if agent.target_kind == "fruit":
                wait = self._fruit_value(item, now)[1]
                hold = wait > 0.5 and energy >= 45
                stop = 18.0 if hold else 0.0
                arrived = hold and remaining <= stop + 1
                mode = "wait" if arrived else "fruit"
            else:
                arrived = remaining < 3
                mode = "camp" if arrived else "tree"
            goal = (item.x, item.y)
            if not arrived:
                point = self._route(agent, item.x, item.y, max(stop, 12.0), now)
                if point is None:
                    item.blocked[id(agent)] = now + 15
                    agent.target, agent.target_kind, mode = None, "", "rest"
                else:
                    desired = math.atan2(point[1] - y, point[0] - x)
                    gap = remaining - stop if point == goal else math.hypot(point[0] - x, point[1] - y)
                    distance = min(walk, max(0.0, gap) / max(modifier, 0.1))
            if agent.target is not None:
                self._track_progress(agent, remaining, now, arrived, 10 if agent.path else 4)
        elif agent.senescent and s["old_rest"]:
            mode = "rest"
        else:
            goal = self._explore(agent, status, now, goals)
            if goal is not None:
                point = self._route(agent, goal[0], goal[1], 40.0, now)
                if point is None:
                    agent.bad_goals[goal] = now + 30
                    agent.goal = None
                else:
                    desired = math.atan2(point[1] - y, point[0] - x)
                    distance = walk
                    mode = "explore"

        if danger:
            vx = vy = 0.0
            for px, py, gap in predators:
                weight = 1 / max(15.0, gap) ** 2
                vx -= px * weight
                vy -= py * weight
            desired = math.atan2(vy, vx)
            distance = sprint if nearest_predator < 65 else walk
            mode = "escape"
        if agent.stuck > 3 and mode != "escape":
            desired += math.pi / 2 if agent.stuck % 16 < 8 else -math.pi / 2
        if distance > 0:
            avoid = []
            if agent.localized and not danger:
                for fruit in self.fruits.near(x, y, distance * modifier + 15):
                    if fruit is not agent.target and self._fruit_value(fruit, now)[1] > 2:
                        avoid.append((fruit.x, fruit.y))
            steer_predators = predators if danger or s["steer_predators"] else []
            desired = self._steer(x, y, desired, distance * modifier, edges, steer_predators, avoid)
        if danger:
            local_edges = [(ax - x, ay - y, bx - x, by - y) for ax, ay, bx, by in edges
                           if segment_distance(x, y, (ax, ay, bx, by)) < sprint * modifier * 2 + 20]
            distance, desired = maneuver_move(threats, desired, walk, sprint, modifier, local_edges,
                                              energy, status["max_energy"])

        # --- looking
        watch = [p for p in predators if p[2] < s["watch_range"]]
        if (danger and predators) or (watch and mode != "explore"):
            nearest = min(watch or predators, key=lambda p: p[2])
            turn = wrap(math.atan2(nearest[1], nearest[0]) - heading)
        elif mode in ("explore", "scan"):
            turn = s["explore_spin"]
        elif mode in ("camp", "wait", "rest"):
            if agent.scan_left <= 0 and now - agent.last_scan >= s["camp_scan_period"]:
                agent.scan_left = s["camp_scan"]
                agent.last_scan = now
            turn = min(agent.scan_left, s["camp_scan"])
            agent.scan_left -= turn
        else:
            turn = s["travel_spin"] if distance > 0 else 0.0

        # --- reproduction
        spawn = False
        control = min(distance, status["speed"]) * 0.05 + max(0.0, distance - status["speed"]) * 0.5
        control += min(math.pi, abs(turn)) / TAU
        # Selection: fitter agents (faster, sharper senses) breed at lower energy, weaker ones at higher.
        edge = s["selection"] * 100 * (self._fitness(status) - self.mean_fitness)
        elite = True
        if s["elite_select"]:
            # Fast walkers escape predators and cover ground for free, so the colony breeds from them.
            elite = walk >= self.best_walk * 0.97
            edge += s["elite_bonus"] if elite else 0.0
        if nearest_predator > 90 and not danger and energy > 100 + control + 1:
            if not elite and population >= s["min_population"]:
                pass
            elif agent.senescent:
                spawn = population < target_population + s["old_birth_margin"] and now - agent.last_birth > 0.5
                if s["selection"] and edge < -s["selection_gap"] * 100 * s["selection"] and population >= 4:
                    spawn = False
            else:
                spawn = (energy > max(150.0, min(450.0, s["birth_energy"] - edge)) and population < target_population
                         and now - agent.last_birth > 3)
                spawn = spawn or (age > s["early_birth_age"]
                                  and energy > max(130.0, min(450.0, s["early_birth_energy"] - edge))
                                  and population < target_population + s["early_birth_margin"]
                                  and now - agent.last_birth > 3)
            if spawn and s["birth_food_check"] and agent.localized and not self._food_near(agent, now):
                spawn = False
            spawn = spawn or (population < 2 and energy > 120)
        if spawn:
            agent.last_birth = now
        agent.mode = mode
        self._advance(agent, status, None, now, distance=distance, desired=desired, turn=turn,
                      spawn=spawn, control=control)
        return {
            "agent_id": status["agent_id"],
            "move_distance": float(distance),
            "move_direction": wrap(desired - heading),
            "turn_angle": float(turn),
            "spawn_agent": bool(spawn),
        }

    def _food_near(self, agent, now, radius=250.0):
        """Whether a newborn here would find unclaimed food: a fruit or a productive tree."""
        claimed = {id(other.target) for other in self.agents.values() if other.target is not None}
        for fruit in self.fruits.near(agent.x, agent.y, radius):
            if id(fruit) not in claimed and self._fruit_value(fruit, now + 10)[2] > 0.5:
                return True
        return any(id(tree) not in claimed and self._tree_output(tree, now + 10) > 40
                   for tree in self.trees.near(agent.x, agent.y, radius))

    @staticmethod
    def _fitness(status):
        walk = min(status["speed"], status["sprint_speed"])
        return walk / 10 + 0.3 * status["vision_range"] / 200 + 0.2 * status["hearing_radius"] / 50

    def _hold(self, agent, status, now):
        agent.mode = "rest"
        self._advance(agent, status, None, now, distance=0.0, desired=agent.move_heading)
        return {"agent_id": status["agent_id"], "move_distance": 0.0, "move_direction": 0.0,
                "turn_angle": 0.0, "spawn_agent": False}

    def _track_progress(self, agent, remaining, now, arrived, patience):
        if arrived or remaining < agent.target_best - 5:
            agent.target_best = remaining
            agent.target_since = now
        elif now - agent.target_since > patience:
            agent.target.blocked[id(agent)] = now + 15
            agent.target, agent.target_kind = None, ""

    def _explore_scores(self, now):
        """Per-region exploration value, shared by every agent that replans this tick."""
        if self.explore_base is None:
            step = CELL * REGION
            base = {}
            for ry in range(int(HEIGHT // step)):
                for rx in range(int(WIDTH // step)):
                    gx, gy = (rx + 0.5) * step, (ry + 0.5) * step
                    if not (WALL + 10 < gx < WIDTH - WALL - 10 and WALL + 10 < gy < HEIGHT - WALL - 10):
                        continue
                    prior = TREE_PRIOR.get(self.biomes.get((int(gx // CELL), int(gy // CELL))), 0.7)
                    score = self._staleness(gx, gy, now) * (0.3 + prior)
                    score -= self.settings["explore_danger"] * self._danger_at(gx, gy)
                    base[(rx, ry)] = score
            # Known food that nobody has claimed draws idle agents towards it.
            weight = self.settings["food_attraction"]
            if weight:
                claimed = {id(agent.target) for agent in self.agents.values() if agent.target is not None}
                for registry, value_of in ((self.fruits, lambda f: self._fruit_value(f, now + 5)[0]
                                            * self._fruit_value(f, now + 5)[2]),
                                           (self.trees, lambda t: 0.5 * self._tree_output(t, now + 5))):
                    for item in registry:
                        key = (int(item.x // step), int(item.y // step))
                        if key in base and id(item) not in claimed:
                            base[key] += weight * value_of(item)
            self.explore_base = base
        return self.explore_base

    def _explore(self, agent, status, now, goals):
        if agent.goal is not None and now - agent.goal_since < 20 and agent.stuck < 6:
            gx, gy = agent.goal
            if math.hypot(gx - agent.x, gy - agent.y) > 50 and self._staleness(gx, gy, now) > 20:
                return agent.goal
        step = CELL * REGION
        scores = dict(self._explore_scores(now))
        points = [(a.x, a.y) for a in self.agents.values() if a is not agent and a.localized]
        points += [g for g in goals if g is not agent.goal]
        reach = int(250 // step) + 1
        for ox, oy in points:
            rx0, ry0 = int(ox // step), int(oy // step)
            for rx in range(rx0 - reach, rx0 + reach + 1):
                for ry in range(ry0 - reach, ry0 + reach + 1):
                    if (rx, ry) in scores:
                        gap = math.hypot((rx + 0.5) * step - ox, (ry + 0.5) * step - oy)
                        scores[(rx, ry)] -= max(0.0, 250 - gap) * 0.25
        best, best_score = None, -float("inf")
        for (rx, ry), score in scores.items():
            gx, gy = (rx + 0.5) * step, (ry + 0.5) * step
            distance = math.hypot(gx - agent.x, gy - agent.y)
            if distance < 60 or agent.bad_goals.get((gx, gy), -1) > now:
                continue
            score -= distance * 0.12
            if score > best_score:
                best, best_score = (gx, gy), score
        agent.goal, agent.goal_since = best, now
        return best

    def _staleness(self, x, y, now):
        cap = self.settings["stale_cap"]
        total = 0.0
        cx0, cy0 = int(x // CELL) - 1, int(y // CELL) - 1
        count = 0
        for cy in (cy0, cy0 + 1):
            for cx in (cx0, cx0 + 1):
                if 0 <= cx < COLUMNS and 0 <= cy < ROWS:
                    total += min(cap, now - self.coverage[cy * COLUMNS + cx])
                    count += 1
        return total / max(1, count)

    def _steer(self, x, y, desired, travel, edges, predators, avoid=()):
        nearby = [edge for edge in edges if segment_distance(x, y, edge) < travel + 34]
        if not nearby and not predators and not avoid:
            return desired
        best, choice = -float("inf"), desired
        for offset in (0.0, -0.35, 0.35, -0.7, 0.7, -1.05, 1.05, -1.4, 1.4, -1.9, 1.9, math.pi):
            angle = desired + offset
            dx, dy = math.cos(angle), math.sin(angle)
            nx, ny = x + dx * travel, y + dy * travel
            score = 20 * math.cos(offset)
            for edge in nearby:
                d = segment_distance(nx, ny, edge)
                if touches_edge(nx, ny, edge) or crosses(x, y, nx, ny, edge):
                    score -= 500 + (7 - min(7, d)) * 20
                future = segment_distance(nx + dx * 18, ny + dy * 18, edge)
                score -= max(0, 16 - future) * 0.3
            for px, py, _ in predators:
                d = math.hypot(px - dx * travel, py - dy * travel)
                score -= 3000 / max(5, d - 15)
                if d < 32:
                    score -= 1000
            for fx, fy in avoid:
                if math.hypot(fx - nx, fy - ny) < 14.5:
                    score -= 60
            if score > best:
                best, choice = score, angle
        return choice

    def _advance(self, agent, status, action, now, stale=False, distance=0.0, desired=0.0, turn=0.0,
                 spawn=False, control=0.0):
        """Dead-reckon the pose and forecast the energy after this action."""
        modifier = BIOME_SPEED.get(status["biome"], 1.0)
        if stale:
            distance = action["move_distance"]
            desired = agent.move_heading
            control = min(distance, status["speed"]) * 0.05 + max(0.0, distance - status["speed"]) * 0.5
        agent.last_energy = status["energy"]
        agent.last_cost = control + 100 * spawn + 0.1
        agent.last_x, agent.last_y = agent.x, agent.y
        agent.last_distance = distance * modifier
        agent.move_heading = desired
        agent.x += agent.last_distance * math.cos(desired)
        agent.y += agent.last_distance * math.sin(desired)
        agent.heading = wrap(agent.heading + turn)
        if not agent.localized and distance > 0 and not stale:
            agent.wander = desired
