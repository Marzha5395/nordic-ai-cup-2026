import heapq
import math

from src.utils.controllers.predator_avoidance import crosses, segment_distance, touches_edge


class Navigator:
    def __init__(self, spacing=15.0, margin=None):
        self.spacing = spacing
        self.margin = margin
        self.edges = {}
        self.blocked = set()
        self.path = []
        self.goal = None
        self.last_plan = -100.0
        self.path_version = 0

    def update(self, edges):
        for edge in edges:
            key = tuple(round(value, 2) for value in edge)
            if key in self.edges:
                continue
            self.edges[key] = edge
            ax, ay, bx, by = edge
            count = max(1, math.ceil(math.hypot(bx - ax, by - ay) / (self.spacing * 0.5)))
            candidates = set()
            for i in range(count + 1):
                x = ax + (bx - ax) * i / count
                y = ay + (by - ay) * i / count
                ix, iy = self.cell(x, y)
                candidates.update((ix + dx, iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            self.blocked.update(cell for cell in candidates
                                if (touches_edge(*self.point(cell), edge, self.margin) if self.margin is not None
                                    else segment_distance(*self.point(cell), edge) < 11))

    def cell(self, x, y):
        return round(x / self.spacing), round(y / self.spacing)

    def point(self, cell):
        return cell[0] * self.spacing, cell[1] * self.spacing

    @staticmethod
    def clear(x, y, gx, gy, edges, margin=5.5):
        segment = (x, y, gx, gy)
        for edge in edges:
            if crosses(x, y, gx, gy, edge) or touches_edge(gx, gy, edge, margin):
                return False
            if min(segment_distance(gx, gy, edge), segment_distance(edge[0], edge[1], segment),
                   segment_distance(edge[2], edge[3], segment)) < margin:
                return False
        return True

    def waypoint(self, x, y, gx, gy, now, stop=8.0):
        distance = math.hypot(gx - x, gy - y)
        if distance <= stop:
            return x, y
        tx = gx - (gx - x) * stop / distance
        ty = gy - (gy - y) * stop / distance
        nearby = [edge for edge in self.edges.values() if segment_distance(x, y, edge) < distance + 30]
        if self.clear(x, y, tx, ty, nearby):
            self.path = []
            self.goal = (gx, gy)
            return tx, ty
        changed = self.goal is None or math.hypot(gx - self.goal[0], gy - self.goal[1]) > 10
        if (changed or (not self.path and now - self.last_plan > 1.0)
                or (self.path_version != len(self.edges) and now - self.last_plan > 0.5)):
            self.goal = (gx, gy)
            self.path = self._search(x, y, gx, gy, stop)
            self.last_plan = now
            self.path_version = len(self.edges)
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < 10:
            self.path.pop(0)
        if not self.path:
            return None
        close_edges = [edge for edge in nearby if segment_distance(x, y, edge) < 150]
        choice = self.path[0]
        for waypoint in self.path[1:12]:
            if math.hypot(waypoint[0] - x, waypoint[1] - y) > 130:
                break
            if self.clear(x, y, waypoint[0], waypoint[1], close_edges):
                choice = waypoint
        return choice

    def _search(self, x, y, gx, gy, stop=8.0):
        sx, sy = self.cell(x, y)
        start = min(((sx + dx, sy + dy) for dx in range(-2, 3) for dy in range(-2, 3)
                     if (sx + dx, sy + dy) not in self.blocked),
                    key=lambda cell: math.hypot(self.point(cell)[0] - x, self.point(cell)[1] - y), default=None)
        if start is None:
            return []
        ex, ey = self.cell(gx, gy)
        goal_edges = [edge for edge in self.edges.values() if segment_distance(gx, gy, edge) < 40]
        goals = set()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cell = ex + dx, ey + dy
                if cell in self.blocked:
                    continue
                px, py = self.point(cell)
                distance = math.hypot(px - gx, py - gy)
                if distance >= 26:
                    continue
                if distance <= stop:
                    goals.add(cell)
                else:
                    tx, ty = gx + (px - gx) * stop / distance, gy + (py - gy) * stop / distance
                    if self.clear(px, py, tx, ty, goal_edges):
                        goals.add(cell)
        if not goals:
            return []
        frontier = [(math.hypot(start[0] - ex, start[1] - ey), 0.0, start)]
        costs = {start: 0.0}
        parents = {}
        reached = None
        moves = ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                 (1, 1, 1.414214), (1, -1, 1.414214), (-1, 1, 1.414214), (-1, -1, 1.414214))
        while frontier and len(costs) < 4500:
            _, cost, cell = heapq.heappop(frontier)
            if cost > costs[cell]:
                continue
            if cell in goals:
                reached = cell
                break
            cx, cy = cell
            for dx, dy, step in moves:
                neighbor = cx + dx, cy + dy
                if neighbor in self.blocked or (dx and dy and
                   ((cx + dx, cy) in self.blocked or (cx, cy + dy) in self.blocked)):
                    continue
                new_cost = cost + step
                if new_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = new_cost
                    parents[neighbor] = cell
                    heuristic = max(0.0, math.hypot(neighbor[0] - ex, neighbor[1] - ey) - 26 / self.spacing)
                    heapq.heappush(frontier, (new_cost + heuristic, new_cost, neighbor))
        if reached is None:
            return []
        path = [self.point(reached)]
        while reached in parents:
            reached = parents[reached]
            path.append(self.point(reached))
        return list(reversed(path))
