from dataclasses import dataclass
import math


@dataclass
class Landmark:
    x: float
    y: float
    seen: float
    first: float
    visited: float = -1000.0
    shared: bool = False
    born: float | None = None


def neighbor_heading(memory, px, py, observation):
    bearing = memory.heading + observation["angle"] if "angle" in observation else math.atan2(py, px)
    opposite = math.pi if observation.get("distance", math.hypot(px, py)) > 0 else 0.0
    return bearing + opposite - observation["rel_dir"]


class WorldMemory:
    def __init__(self):
        self.frames = {}
        self.fruits = {}
        self.trees = {}

    @staticmethod
    def to_world(frame, x, y):
        angle, ox, oy = frame
        c, s = math.cos(angle), math.sin(angle)
        return ox + c * x - s * y, oy + s * x + c * y

    @staticmethod
    def to_local(frame, x, y):
        angle, ox, oy = frame
        c, s = math.cos(angle), math.sin(angle)
        dx, dy = x - ox, y - oy
        return c * dx + s * dy, -s * dx + c * dy

    def locate(self, agent_id, memory):
        if agent_id in self.frames:
            return
        for group in memory.edges.values():
            for ax, ay, bx, by in group:
                length = math.hypot(bx - ax, by - ay)
                horizontal = abs(length - 1600) < 0.01
                vertical = abs(length - 1200) < 0.01
                if not horizontal and not vertical:
                    continue
                angle = (0.0 if horizontal else math.pi / 2) - math.atan2(by - ay, bx - ax)
                c, s = math.cos(angle), math.sin(angle)
                ex, ey = c * ax - s * ay, s * ax + c * ay
                px, py = c * memory.x - s * memory.y, s * memory.x + c * memory.y
                if horizontal:
                    ox, oy = -ex, (30 if py > ey else 1170) - ey
                else:
                    ox, oy = (30 if px > ex else 1570) - ex, -ey
                self.frames[agent_id] = (angle, ox, oy)
                return

    def update(self, memories, sensed, statuses, now):
        self.frames = {key: frame for key, frame in self.frames.items() if key in memories}
        for agent_id, memory in memories.items():
            self.locate(agent_id, memory)
        for _ in range(2):
            for agent_id, objects in sensed.items():
                if agent_id in self.frames:
                    continue
                memory = memories[agent_id]
                for px, py, obs in objects["Agent"]:
                    neighbor_id = obs.get("id")
                    if neighbor_id not in self.frames:
                        continue
                    neighbor = memories[neighbor_id]
                    frame = self.frames[neighbor_id]
                    wx, wy = self.to_world(frame, neighbor.x, neighbor.y)
                    relative_heading = neighbor_heading(memory, px, py, obs)
                    angle = frame[0] + neighbor.heading - relative_heading
                    c, s = math.cos(angle), math.sin(angle)
                    nx, ny = memory.x + px, memory.y + py
                    self.frames[agent_id] = (angle, wx - c * nx + s * ny, wy - s * nx - c * ny)
                    break
        for registry, kind, expiry in ((self.fruits, "fruits", 45), (self.trees, "trees", 75)):
            for key in list(registry):
                if now - registry[key][2] > expiry:
                    del registry[key]
            for agent_id in sensed:
                if agent_id not in self.frames:
                    continue
                memory = memories[agent_id]
                frame = self.frames[agent_id]
                ax, ay = self.to_world(frame, memory.x, memory.y)
                seen = set()
                for point in getattr(memory, kind):
                    if point.seen != now:
                        continue
                    wx, wy = self.to_world(frame, point.x, point.y)
                    key = (round(wx / 3), round(wy / 3))
                    registry[key] = (wx, wy, now, point.first)
                    seen.add(key)
                radius = statuses[agent_id]["hearing_radius"] - 2
                for key in list(registry):
                    px, py, _, _ = registry[key]
                    lx, ly = self.to_local(frame, px, py)
                    if key not in seen and (math.hypot(px - ax, py - ay) < radius
                                           or (memory.prune_visible and memory.detectable(Landmark(lx, ly, now, now), statuses[agent_id]))):
                        del registry[key]
            assignments = {}
            if kind == "fruits":
                pairs = []
                for agent_id in sensed:
                    if agent_id not in self.frames:
                        continue
                    memory = memories[agent_id]
                    ax, ay = self.to_world(self.frames[agent_id], memory.x, memory.y)
                    status = statuses[agent_id]
                    for key, (px, py, seen, _) in registry.items():
                        distance = math.hypot(px - ax, py - ay)
                        if distance < 700 and now - seen < 25:
                            cost = distance + status["energy"] * 0.15 + max(0, status["age"] - 60) * 3
                            pairs.append((cost, agent_id, key))
                claimed = set()
                for _, agent_id, key in sorted(pairs):
                    if agent_id not in assignments and key not in claimed:
                        assignments[agent_id] = registry[key]
                        claimed.add(key)
            for agent_id in sensed:
                if agent_id not in self.frames:
                    continue
                memory = memories[agent_id]
                frame = self.frames[agent_id]
                destination = getattr(memory, kind)
                radius = statuses[agent_id]["hearing_radius"] + 2
                candidates = registry.values()
                if kind == "fruits":
                    assigned = assignments.get(agent_id)
                    candidates = [assigned] if assigned else []
                    local_target = self.to_local(frame, assigned[0], assigned[1]) if assigned else None
                    destination[:] = [
                        p
                        for p in destination
                        if not p.shared
                        or (
                            local_target is not None
                            and math.hypot(p.x - local_target[0], p.y - local_target[1]) < 5
                        )
                    ]
                for wx, wy, seen, first in candidates:
                    px, py = self.to_local(frame, wx, wy)
                    distance = math.hypot(px - memory.x, py - memory.y)
                    if (
                        radius < distance < 700
                        and now - seen < 25
                        and not any(math.hypot(point.x - px, point.y - py) < 5 for point in destination)
                    ):
                        destination.append(Landmark(px, py, seen, first, shared=kind == "fruits"))
