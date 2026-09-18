import math
import random
import unittest

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest, StepResponse
from src.utils.controllers.forager_policy import ForagerPolicy, Item, Planner, wrap


def status(agent_id=0, observations=None, **kwargs):
    result = {
        "agent_id": agent_id,
        "observations": observations or [],
        "energy": 150.0,
        "biome": "forest",
        "age": 0.1,
        "speed": 10.0,
        "sprint_speed": 20.0,
        "hearing_radius": 50.0,
        "vision_angle": math.pi / 3,
        "vision_range": 200.0,
        "max_energy": 500.0,
    }
    result.update(kwargs)
    return result


def edge_observation(x, y, direction, start, end):
    """An Edge observation exactly as Creature.observe encodes it."""
    c, s = math.cos(-direction), math.sin(-direction)
    coords = []
    for px, py in (start, end):
        dx, dy = px - x, py - y
        coords.append((dx * c - dy * s, dx * s + dy * c))
    return {"type": "Edge", "coords": tuple(coords)}


def localized_policy(x=500.0, y=200.0, direction=0.3, **kwargs):
    policy = ForagerPolicy()
    wall = edge_observation(x, y, direction, (0.0, 30.0), (1600.0, 30.0))
    action = policy.decide([status(observations=[wall], **kwargs)], 0.1)[0]
    return policy, action


class ForagerPolicyTests(unittest.TestCase):
    def test_localizes_from_each_boundary_wall(self):
        walls = [((0.0, 30.0), (1600.0, 30.0)), ((0.0, 1170.0), (1600.0, 1170.0)),
                 ((30.0, 0.0), (30.0, 1200.0)), ((1570.0, 0.0), (1570.0, 1200.0))]
        rng = random.Random(3)
        for start, end in walls:
            for _ in range(10):
                x, y, direction = rng.uniform(60, 1540), rng.uniform(60, 1140), rng.uniform(-math.pi, math.pi)
                policy = ForagerPolicy()
                action = policy.decide([status(observations=[edge_observation(x, y, direction, start, end)])], 0.1)[0]
                agent = policy.agents[0]
                self.assertTrue(agent.localized)
                self.assertAlmostEqual(agent.last_x, x, places=6)
                self.assertAlmostEqual(agent.last_y, y, places=6)
                self.assertAlmostEqual(wrap(agent.heading - action["turn_angle"] - direction), 0.0, places=6)

    def test_offspring_inherits_world_frame_even_when_overlapping(self):
        rng = random.Random(11)
        for i in range(40):
            parent_x, parent_y, parent_dir = 500.0, 200.0, rng.uniform(-math.pi, math.pi)
            policy, action = localized_policy(parent_x, parent_y, parent_dir)
            parent = policy.agents[0]
            # The parent does not move this tick; the child appears beside (or on top of) it.
            parent_dir = wrap(parent_dir + action["turn_angle"])
            parent.x, parent.y, parent.heading = parent_x, parent_y, parent_dir
            parent.last_distance = 0.0
            offset = 0.0 if i % 2 else rng.uniform(10, 30)
            angle = rng.uniform(-math.pi, math.pi)
            child_x, child_y = parent_x + offset * math.cos(angle), parent_y + offset * math.sin(angle)
            child_dir = rng.uniform(-math.pi, math.pi)
            bearing = math.atan2(parent_y - child_y, parent_x - child_x) if offset else 0.0
            back = math.atan2(child_y - parent_y, child_x - parent_x) if offset else 0.0
            seen = {"type": "Agent", "id": 0, "distance": offset,
                    "angle": wrap(bearing - child_dir), "rel_dir": wrap(back - parent_dir)}
            policy.agents[0].last_age = -5  # the parent's observation stays fresh
            policy.decide([status(0, age=0.2), status(1, age=0.1, observations=[seen])], 0.2)
            child = policy.agents[1]
            self.assertTrue(child.localized)
            self.assertAlmostEqual(child.last_x, child_x, places=6)
            self.assertAlmostEqual(child.last_y, child_y, places=6)

    def test_move_direction_is_relative_to_heading(self):
        policy, _ = localized_policy(direction=0.0)
        agent = policy.agents[0]
        heading = agent.heading
        fruit = {"type": "Fruit", "distance": 40.0, "angle": wrap(math.pi / 2 - heading)}
        action = policy.decide([status(age=0.2, energy=40, observations=[fruit])], 0.2)[0]
        self.assertAlmostEqual(wrap(heading + action["move_direction"]), math.pi / 2, places=3)
        self.assertGreater(action["move_distance"], 0)

    def test_hungry_agent_walks_onto_fruit(self):
        policy, _ = localized_policy(direction=0.0)
        heading = policy.agents[0].heading
        fruit = {"type": "Fruit", "distance": 8.0, "angle": -heading}
        action = policy.decide([status(age=0.2, energy=40, observations=[fruit])], 0.2)[0]
        self.assertAlmostEqual(action["move_distance"], 8.0, places=6)

    def test_retry_is_idempotent(self):
        policy = ForagerPolicy()
        agents = [status()]
        first = policy.decide(agents, 0.1)
        position = (policy.agents[0].x, policy.agents[0].y)
        self.assertEqual(first, policy.decide(agents, 0.1))
        self.assertEqual(position, (policy.agents[0].x, policy.agents[0].y))

    def test_simulation_reset(self):
        policy = ForagerPolicy()
        expected = ForagerPolicy().decide([status()], 0.1)
        policy.decide([status()], 100.0)
        self.assertEqual(expected, policy.decide([status()], 0.1))
        self.assertEqual([], policy.decide([], 0.0))
        self.assertEqual({}, policy.agents)

    def test_dead_agents_are_pruned(self):
        policy = ForagerPolicy()
        policy.decide([status(0), status(1)], 0.1)
        policy.decide([status(1, age=0.2)], 0.2)
        self.assertEqual(set(policy.agents), {1})

    def test_stale_observation_repeats_movement_without_turning_or_spawning(self):
        policy, first = localized_policy(energy=400)
        before = (policy.agents[0].x, policy.agents[0].y)
        action = policy.decide([status(age=0.1, energy=399)], 0.2)[0]
        self.assertEqual(action["turn_angle"], 0.0)
        self.assertFalse(action["spawn_agent"])
        self.assertEqual(action["move_distance"], first["move_distance"])
        moved = math.hypot(policy.agents[0].x - before[0], policy.agents[0].y - before[1])
        self.assertAlmostEqual(moved, first["move_distance"])

    def test_skipped_observation_does_not_imply_senescence(self):
        policy = ForagerPolicy()
        policy.decide([status(age=80, energy=150)], 100)
        cost = policy.agents[0].last_cost
        policy.decide([status(age=80, energy=150 - cost + 0.1)], 100.1)
        cost = policy.agents[0].last_cost
        policy.decide([status(age=80.1, energy=150 - 0.1 - cost - 0.05)], 100.2)
        self.assertFalse(policy.agents[0].senescent)

    def test_detects_senescence_from_extra_drain(self):
        policy = ForagerPolicy()
        policy.decide([status(age=80, energy=150)], 100)
        cost = policy.agents[0].last_cost
        policy.decide([status(age=80.1, energy=150 - cost - 0.801)], 100.1)
        self.assertTrue(policy.agents[0].senescent)

    def test_reproduction_respects_energy_budget(self):
        rich = ForagerPolicy().decide([status(energy=450)], 0.1)[0]
        self.assertTrue(rich["spawn_agent"])
        poor = ForagerPolicy().decide([status(energy=100.5, age=100)], 0.1)[0]
        self.assertFalse(poor["spawn_agent"])
        crowded = ForagerPolicy({"min_population": 4, "population": 4, "early_birth_margin": 0})
        self.assertFalse(any(a["spawn_agent"] for a in crowded.decide([status(i, energy=450) for i in range(4)], 0.1)))

    def test_planner_routes_around_wall(self):
        planner = Planner()
        planner.add((300.0, 100.0, 300.0, 400.0))
        path = planner.search(200.0, 250.0, 400.0, 250.0, 12.0)
        self.assertIsNotNone(path)
        self.assertTrue(any(abs(y - 250) > 150 for _, y in path))
        self.assertLessEqual(math.hypot(path[-1][0] - 400, path[-1][1] - 250), 12.0 + 1e-9)

    def test_ripeness_estimate(self):
        fresh = Item(0, 0, 10.0, 10.0, 10.0)
        value, wait, alive = ForagerPolicy._fruit_value(fresh, 15.0)
        self.assertAlmostEqual(value, 30.0)
        self.assertAlmostEqual(wait, 15.0)
        self.assertEqual(alive, 1.0)
        self.assertEqual(ForagerPolicy._fruit_value(fresh, 61.0)[2], 0.0)

    def test_mutated_traits_produce_finite_legal_actions(self):
        rng = random.Random(7)
        policy = ForagerPolicy()
        for tick in range(1, 100):
            agent = status(
                age=tick / 10,
                energy=rng.uniform(1, 500),
                speed=rng.uniform(0.1, 20),
                sprint_speed=rng.uniform(0.1, 40),
                hearing_radius=rng.uniform(1, 100),
                vision_angle=rng.uniform(0.05, math.pi / 2),
                vision_range=rng.uniform(1, 400),
                max_energy=rng.uniform(75, 1000),
                observations=[{"type": "Predator", "distance": rng.uniform(5, 150),
                               "angle": rng.uniform(-3, 3), "rel_dir": rng.uniform(-3, 3)}],
            )
            action = policy.decide([agent], tick / 10)[0]
            ActionRequest(**action)
            self.assertTrue(all(math.isfinite(action[k]) for k in ("move_distance", "move_direction", "turn_angle")))
            self.assertGreaterEqual(action["move_distance"], 0)
            self.assertLessEqual(action["move_distance"], agent["sprint_speed"] + 1e-9)

    def test_world_localization_in_simulator(self):
        sim = SimulationCore(seed=5)
        policy = ForagerPolicy()
        actions = []
        worst = 0.0
        localized = 0
        while sim.env.time < 60 and sim.env.agents:
            state = sim.step(actions)
            decisions = policy.decide(state["observations"], state["sim_time"])
            actions = [(d["agent_id"], ActionRequest(**d)) for d in decisions]
            for agent in sim.env.agents:
                memory = policy.agents[agent.agent_id]
                if memory.localized:
                    localized += 1
                    worst = max(worst, math.hypot(memory.last_x - agent.x, memory.last_y - agent.y))
        self.assertGreater(localized, 0)
        self.assertLess(worst, 15.0)

    def test_tracked_killer_beyond_normal_range_does_not_crash(self):
        policy, _ = localized_policy(energy=300)
        agent = policy.agents[0]
        policy.settings.update(killer_memory=60, killer_range=250)
        policy.killers = [[agent.x + 200, agent.y, 0.1, 60.0]]
        heading = agent.heading
        predator = {"type": "Predator", "distance": 200.0, "angle": -heading, "rel_dir": math.pi}
        action = policy.decide([status(age=0.2, energy=300, observations=[predator])], 0.2)[0]
        ActionRequest(**action)
        self.assertEqual(policy.agents[0].mode, "escape")

    def test_internal_error_returns_safe_actions(self):
        policy = ForagerPolicy()
        policy._plan = None  # force a failure inside the decision
        actions = policy.decide([status(0), status(1)], 0.1)
        self.assertEqual([a["agent_id"] for a in actions], [0, 1])
        self.assertTrue(all(a["move_distance"] == 0 and not a["spawn_agent"] for a in actions))

    def test_unknown_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            ForagerPolicy({"no_such_setting": 1})

    def test_endpoint_contract(self):
        from agent_server import predict

        payload = StepResponse(game_status="ok", score=0, sim_time=0.1, n_agents=1, agent_status=[status()])
        result = predict(payload)
        self.assertEqual(set(result), {"actions"})
        self.assertEqual(ActionRequest(**result["actions"][0]).agent_id, 0)


if __name__ == "__main__":
    unittest.main()
