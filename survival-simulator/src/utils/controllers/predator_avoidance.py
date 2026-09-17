import math


TAU = 2 * math.pi


def wrap(angle):
    return (angle + math.pi) % TAU - math.pi


def closest_point(x, y, edge):
    ax, ay, bx, by = edge
    dx, dy = bx - ax, by - ay
    scale = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / scale)) if scale else 0.0
    return ax + t * dx, ay + t * dy


def segment_distance(x, y, edge):
    px, py = closest_point(x, y, edge)
    return math.hypot(x - px, y - py)


def touches_edge(x, y, edge, margin=5.1):
    ax, ay, bx, by = edge
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(x - ax, y - ay) < margin
    along = ((x - ax) * dx + (y - ay) * dy) / length
    across = abs((x - ax) * dy - (y - ay) * dx) / length
    return -margin < along < length + margin and across < margin


def crosses(ax, ay, bx, by, edge):
    cx, cy, dx, dy = edge
    vx, vy, wx, wy = bx - ax, by - ay, dx - cx, dy - cy
    det = vx * wy - vy * wx
    if abs(det) < 1e-8:
        return False
    t = ((cx - ax) * wy - (cy - ay) * wx) / det
    u = ((cx - ax) * vy - (cy - ay) * vx) / det
    return 0.0 < t < 1.0 and 0.0 <= u <= 1.0


def predict_predator(px, py, heading, ax, ay, agent_heading, modifier=1.0):
    dx, dy = ax - px, ay - py
    distance = math.hypot(dx, dy)
    angle = wrap(math.atan2(dy, dx) - heading)
    looking = wrap(math.atan2(-dy, -dx) - agent_heading)
    if distance > 60 and (distance > 250 or abs(angle) > math.pi / 6):
        return px + 11 * modifier * math.cos(heading), py + 11 * modifier * math.sin(heading), heading
    if abs(looking) > math.pi / 2 or distance < 90:
        turn = max(-0.3, min(0.3, angle * 0.5)) if abs(angle) > 0.05 else 0.0
        movement = turn if abs(angle) > 0.05 else angle
        length = min(15.0, distance) * modifier
    else:
        movement = angle - (1 if looking > 0 else -1 if looking < 0 else 0) * math.pi / 4
        turn = math.atan2(
            distance * math.sin(angle) - 15 * math.sin(movement),
            distance * math.cos(angle) - 15 * math.cos(movement),
        )
        length = 15.0 * modifier
    direction = heading + movement
    return px + length * math.cos(direction), py + length * math.sin(direction), wrap(heading + turn)


def escape_move(threats, desired, walk, sprint, modifier, local_edges):
    best_score = -float("inf")
    best_move = (walk, desired)
    speeds = [walk] if sprint <= walk else [walk, min(sprint, max(walk, 15.5)), sprint]
    for speed in speeds:
        travel = speed * modifier
        energy_cost = min(speed, walk) * 0.05 + max(0.0, speed - walk) * 0.5
        for i in range(24):
            direction = desired + i * TAU / 24
            dx, dy = travel * math.cos(direction), travel * math.sin(direction)
            score = 2 * math.cos(direction - desired) - 8 * energy_cost
            if any(segment_distance(dx, dy, edge) < 7 or crosses(0, 0, dx, dy, edge) for edge in local_edges):
                continue
            for edge in local_edges:
                clearance = segment_distance(dx * 2, dy * 2, edge)
                score -= max(0.0, 12 - clearance) * 3
            for px, py, predator_heading, active, predator_modifier in threats:
                ax, ay = 0.0, 0.0
                for step in range(5):
                    ax, ay = ax + dx, ay + dy
                    before = math.hypot(px - ax, py - ay)
                    facing = math.atan2(py - ay, px - ax)
                    if active:
                        px, py, predator_heading = predict_predator(
                            px, py, predator_heading, ax, ay, facing, predator_modifier
                        )
                    clearance = min(before, math.hypot(px - ax, py - ay))
                    discount = 0.7**step
                    score -= discount * max(0.0, 65 - clearance) ** 2 * 0.06
                    if clearance < 22:
                        score -= discount * (500 + (22 - clearance) * 100)
                    if clearance < 15:
                        score -= discount * 10000
            if score > best_score:
                best_score, best_move = score, (speed, direction)
    return best_move


def maneuver_move(threats, desired, walk, sprint, modifier, local_edges, energy, max_energy, margin=19.0, look_bias=0.0):
    best_score = -float("inf")
    best_move = (walk, desired)
    speeds = [walk] if sprint <= walk else [walk, sprint]
    for speed in speeds:
        for i in range(20):
            direction = desired + i * TAU / 20
            for curvature in (0.0, -0.4, 0.4):
                ax, ay = 0.0, 0.0
                remaining = energy
                predicted = list(threats)
                score = 2 * math.cos(direction - desired)
                for step in range(8):
                    length = speed if remaining >= max_energy / 5 else min(speed, walk)
                    cost = min(length, walk) * 0.05 + max(0, length - walk) * 0.5 + 0.1
                    remaining -= cost
                    angle = direction + curvature * step
                    nx = ax + length * modifier * math.cos(angle)
                    ny = ay + length * modifier * math.sin(angle)
                    discount = 0.8 ** step
                    if any(segment_distance(nx, ny, edge) < 6 or crosses(ax, ay, nx, ny, edge) for edge in local_edges):
                        score -= 20000 * discount
                        break
                    score -= discount * cost * 5
                    updated = []
                    for px, py, heading, active, pred_modifier in predicted:
                        before = math.hypot(px - nx, py - ny)
                        facing = math.atan2(py - ny, px - nx)
                        if active:
                            px, py, heading = predict_predator(px, py, heading, nx, ny, facing + look_bias, pred_modifier)
                        clearance = min(before, math.hypot(px - nx, py - ny))
                        score -= discount * max(0, 55 - clearance) ** 2 * 0.025
                        score -= discount * max(0, margin - clearance) ** 2 * 25
                        if clearance < max(15.5, margin - 3.5):
                            score -= discount * 20000
                        updated.append((px, py, heading, active, pred_modifier))
                    ax, ay, predicted = nx, ny, updated
                    if score < best_score:
                        break
                if score > best_score:
                    best_score, best_move = score, (speed, direction)
    return best_move
