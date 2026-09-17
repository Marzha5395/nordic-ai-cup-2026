import math
import random
import unittest

from src.utils.DTOs import ActionRequest, StepResponse
from src.utils.controllers.survival_policy import (
    AgentMemory,
    Landmark,
    SurvivalPolicy,
    crosses,
    segment_distance,
)


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


class SurvivalPolicyTests(unittest.TestCase):
    def test_food_direction_is_relative(self):
        action = SurvivalPolicy().decide(
            [status(observations=[{"type": "Fruit", "distance": 40.0, "angle": 1.2}])], 0.1
        )[0]
        self.assertAlmostEqual(action["move_direction"], 1.2)
        self.assertAlmostEqual(action["turn_angle"], 1.2)
        self.assertEqual(action["move_distance"], 10.0)

    def test_stop_within_fruit_pickup_radius(self):
        action = SurvivalPolicy().decide(
            [status(observations=[{"type": "Fruit", "distance": 12.0, "angle": 0.0}])], 0.1
        )[0]
        self.assertAlmostEqual(action["move_distance"], 7.0)

    def test_economical_pickup_stops_just_inside_the_touch_radius(self):
        action = SurvivalPolicy(tuning={"pickup_stop": 9.5}).decide([
            status(observations=[{"type": "Fruit", "distance": 12, "angle": 0}]),
        ], 0.1)[0]
        self.assertAlmostEqual(action["move_distance"], 2.5)

    def test_biome_movement_modifier(self):
        action = SurvivalPolicy().decide(
            [status(biome="swamp", observations=[{"type": "Fruit", "distance": 12.0, "angle": 0.0}])], 0.1
        )[0]
        self.assertEqual(action["move_distance"], 10.0)

    def test_predator_escape(self):
        action = SurvivalPolicy().decide(
            [
                status(
                    energy=200,
                    observations=[{"type": "Predator", "distance": 35.0, "angle": 0.0, "rel_dir": 0.0}],
                )
            ],
            0.1,
        )[0]
        self.assertLess(math.cos(action["move_direction"]), -0.9)
        self.assertEqual(action["move_distance"], 20.0)
        self.assertFalse(action["spawn_agent"])

    def test_no_sprinting_below_energy_threshold(self):
        action = SurvivalPolicy().decide(
            [
                status(
                    energy=90,
                    observations=[{"type": "Predator", "distance": 35.0, "angle": 0.0, "rel_dir": 0.0}],
                )
            ],
            0.1,
        )[0]
        self.assertEqual(action["move_distance"], 10.0)

    def test_avoid_wall_between_agent_and_fruit(self):
        edge = (8.0, -50.0, 8.0, 50.0)
        action = SurvivalPolicy().decide(
            [
                status(
                    observations=[
                        {"type": "Fruit", "distance": 40.0, "angle": 0.0},
                        {"type": "Edge", "coords": [[8.0, -50.0], [8.0, 50.0]]},
                    ]
                )
            ],
            0.1,
        )[0]
        x = action["move_distance"] * math.cos(action["move_direction"])
        y = action["move_distance"] * math.sin(action["move_direction"])
        self.assertFalse(crosses(0, 0, x, y, edge))
        self.assertGreaterEqual(segment_distance(x, y, edge), 7)

    def test_precise_wall_clearance_uses_passable_corridors(self):
        action = SurvivalPolicy(tuning={"precise_walls": True}).decide([
            status(observations=[
                {"type": "Fruit", "distance": 40, "angle": math.pi / 2},
                {"type": "Edge", "coords": [[-6, -100], [-6, 100]]},
                {"type": "Edge", "coords": [[6, -100], [6, 100]]},
            ]),
        ], 0.1)[0]
        self.assertAlmostEqual(action["move_direction"], math.pi / 2)
        self.assertEqual(action["move_distance"], 10)

    def test_reproduction_budget(self):
        agents = [status(i, energy=350) for i in range(12)]
        actions = SurvivalPolicy(population=12).decide(agents, 0.1)
        self.assertFalse(any(a["spawn_agent"] for a in actions))
        action = SurvivalPolicy().decide([status(energy=350)], 0.1)[0]
        self.assertTrue(action["spawn_agent"])
        action = SurvivalPolicy().decide([status(energy=105)], 0.1)[0]
        self.assertFalse(action["spawn_agent"])

    def test_reproduction_cooldown(self):
        policy = SurvivalPolicy()
        self.assertTrue(policy.decide([status(energy=350)], 0.1)[0]["spawn_agent"])
        self.assertFalse(policy.decide([status(energy=350, age=0.2)], 0.2)[0]["spawn_agent"])

    def test_stale_observations_do_not_corrupt_odometry(self):
        policy = SurvivalPolicy()
        fruit = {"type": "Fruit", "distance": 40.0, "angle": 0.0}
        policy.decide([status(age=1.0, observations=[fruit])], 1.0)
        action = policy.decide([status(age=1.0, energy=149.0, observations=[fruit])], 1.1)[0]
        self.assertAlmostEqual(policy.agents[0].x, 20.0)
        self.assertEqual(action["turn_angle"], 0.0)
        self.assertFalse(action["spawn_agent"])

    def test_skipped_observation_does_not_falsely_imply_senescence(self):
        policy = SurvivalPolicy()
        policy.decide([status(age=80, energy=150, observations=[
            {"type": "Fruit", "distance": 100, "angle": 0},
        ])], 100)
        policy.decide([status(age=80, energy=149.5, observations=[
            {"type": "Fruit", "distance": 100, "angle": 0},
        ])], 100.1)
        policy.decide([status(age=80.1, energy=148.9, observations=[
            {"type": "Fruit", "distance": 80, "angle": 0},
        ])], 100.2)
        self.assertFalse(policy.agents[0].senescent)

    def test_unreachable_food_is_temporarily_skipped(self):
        policy = SurvivalPolicy()
        for tick in range(35):
            heading = policy.agents[0].heading if policy.agents else 0.0
            policy.decide(
                [
                    status(
                        age=tick / 10, observations=[{"type": "Fruit", "distance": 40.0, "angle": -heading}]
                    )
                ],
                tick / 10,
            )
        self.assertGreaterEqual(policy.agents[0].fruits[0].visited, 3.0)

    def test_offspring_inherits_rotated_map(self):
        policy = SurvivalPolicy()
        parent = AgentMemory(x=100, y=200)
        parent.edges[(0, 100)] = [(130, 150, 130, 250)]
        parent.fruits = [Landmark(200, 200, 1, 1)]
        child = AgentMemory()
        policy.agents = {0: parent, 1: child}
        policy._inherit_map(child, [(0, 20, {"id": 0, "rel_dir": 0})], 1)
        self.assertAlmostEqual(child.fruits[0].x, 0)
        self.assertAlmostEqual(child.fruits[0].y, -80)
        edge = next(iter(child.edges.values()))[0]
        for actual, expected in zip(edge, (-50, -10, 50, -10)):
            self.assertAlmostEqual(actual, expected)

    def test_overlapping_offspring_inherits_correct_map_orientation(self):
        policy = SurvivalPolicy()
        parent = AgentMemory(x=100, y=200, heading=0.4, fruits=[Landmark(200, 200, 1, 1)])
        child = AgentMemory()
        policy.agents = {0: parent, 1: child}
        policy._inherit_map(child, [(0.0, 0.0, {"id": 0, "distance": 0.0, "angle": -2.3, "rel_dir": -1.5})], 1)
        self.assertAlmostEqual(child.fruits[0].x, 100 * math.cos(-1.2))
        self.assertAlmostEqual(child.fruits[0].y, 100 * math.sin(-1.2))

    def test_map_inheritance_across_random_frames_and_overlap(self):
        from src.utils.controllers.predator_avoidance import wrap

        rng = random.Random(81)
        for i in range(100):
            parent_frame, child_frame = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
            heading = rng.uniform(-math.pi, math.pi)
            dx, dy = (0.0, 0.0) if i % 2 else (rng.uniform(-40, 40), rng.uniform(-40, 40))
            px = dx * math.cos(child_frame) + dy * math.sin(child_frame)
            py = -dx * math.sin(child_frame) + dy * math.cos(child_frame)
            bearing_back = math.atan2(-dy, -dx) if dx or dy else 0.0
            obs = {"id": 0, "distance": math.hypot(dx, dy),
                   "angle": wrap(math.atan2(dy, dx) - child_frame),
                   "rel_dir": wrap(bearing_back - parent_frame - heading)}
            parent = AgentMemory(x=10, y=20, heading=heading, fruits=[Landmark(210, 35, 1, 1)])
            child = AgentMemory()
            policy = SurvivalPolicy()
            policy.agents = {0: parent, 1: child}
            policy._inherit_map(child, [(px, py, obs)], 1)
            rotation = parent_frame - child_frame
            self.assertAlmostEqual(child.fruits[0].x, px + 200 * math.cos(rotation) - 15 * math.sin(rotation))
            self.assertAlmostEqual(child.fruits[0].y, py + 200 * math.sin(rotation) + 15 * math.cos(rotation))

    def test_overlapping_agents_share_correct_world_frame(self):
        from src.utils.controllers.world_memory import WorldMemory

        world = WorldMemory()
        world.frames[0] = (1.1, 100, 200)
        parent = AgentMemory(heading=0.4)
        child = AgentMemory()
        sensed = {0: {"Agent": []}, 1: {"Agent": [(0.0, 0.0, {
            "id": 0, "distance": 0.0, "angle": -2.3, "rel_dir": -1.5,
        })]}}
        world.update({0: parent, 1: child}, sensed, {0: status(0), 1: status(1)}, 1)
        x, y = world.to_world(world.frames[1], 100, 0)
        self.assertAlmostEqual(x, 100 + 100 * math.cos(2.3))
        self.assertAlmostEqual(y, 200 + 100 * math.sin(2.3))

    def test_camp_only_within_reliable_hearing_range(self):
        action = SurvivalPolicy().decide(
            [
                status(
                    energy=100,
                    hearing_radius=20,
                    observations=[{"type": "Tree", "distance": 40.0, "angle": 0.0}],
                )
            ],
            1.0,
        )[0]
        self.assertGreater(action["move_distance"], 0.0)

    def test_rest_near_productive_tree(self):
        action = SurvivalPolicy().decide(
            [status(energy=100, observations=[{"type": "Tree", "distance": 20.0, "angle": 0.0}])], 1.0
        )[0]
        self.assertEqual(action["move_distance"], 0.0)
        self.assertGreater(action["turn_angle"], 0.0)

    def test_world_location_from_boundary(self):
        from src.utils.controllers.world_memory import WorldMemory

        frame = (1.2, 700.0, 500.0)
        for edge in ((0, 30, 1600, 30), (0, 1170, 1600, 1170), (30, 0, 30, 1200), (1570, 0, 1570, 1200)):
            memory = AgentMemory()
            ax, ay = WorldMemory.to_local(frame, edge[0], edge[1])
            bx, by = WorldMemory.to_local(frame, edge[2], edge[3])
            memory.edges[(bx - ax, by - ay)] = [(ax, ay, bx, by)]
            world = WorldMemory()
            world.locate(0, memory)
            x, y = world.to_world(world.frames[0], 0, 0)
            self.assertAlmostEqual(x, 700.0)
            self.assertAlmostEqual(y, 500.0)
            px, py = world.to_world(world.frames[0], 10, 20)
            ex, ey = world.to_world(frame, 10, 20)
            self.assertAlmostEqual(px, ex)
            self.assertAlmostEqual(py, ey)

    def test_hivemind_shares_observed_food(self):
        from src.utils.controllers.world_memory import WorldMemory

        world = WorldMemory()
        world.frames[0] = (0, 100, 200)
        parent = AgentMemory(fruits=[Landmark(100, 0, 1, 1)])
        child = AgentMemory()
        sensed = {0: {"Agent": []}, 1: {"Agent": [(0, 20, {"id": 0, "rel_dir": 0})]}}
        world.update({0: parent, 1: child}, sensed, {0: status(0), 1: status(1)}, 1)
        self.assertAlmostEqual(child.fruits[0].x, 0)
        self.assertAlmostEqual(child.fruits[0].y, -80)
        self.assertAlmostEqual(world.to_world(world.frames[1], 0, 0)[0], 120)

    def test_predator_prediction_matches_simulator(self):
        from src.elements.predator import Predator
        from src.utils.controllers.predator_avoidance import predict_predator, wrap

        rng = random.Random(42)
        for _ in range(100):
            predator = Predator(10, 20, rng=rng)
            distance = rng.uniform(20, 200)
            angle = rng.uniform(-math.pi / 6, math.pi / 6)
            looking = rng.uniform(-math.pi, math.pi)
            ax = predator.x + distance * math.cos(predator.direction + angle)
            ay = predator.y + distance * math.sin(predator.direction + angle)
            agent_heading = math.atan2(predator.y - ay, predator.x - ax) - looking
            modifier = rng.choice([0.3, 0.5, 0.8, 1.0])
            signals = predator.step(
                [{"type": "Agent", "distance": distance, "angle": angle, "rel_dir": looking}]
            )
            expected_x = predator.x + signals["move"] * modifier * math.cos(
                predator.direction + signals["direction"]
            )
            expected_y = predator.y + signals["move"] * modifier * math.sin(
                predator.direction + signals["direction"]
            )
            px, py, heading = predict_predator(
                predator.x, predator.y, predator.direction, ax, ay, agent_heading, modifier
            )
            self.assertAlmostEqual(px, expected_x)
            self.assertAlmostEqual(py, expected_y)
            self.assertAlmostEqual(wrap(heading - predator.direction - signals.get("turn", 0)), 0)

    def test_collision_localization(self):
        memory = AgentMemory()
        memory.observe(status(observations=[{"type": "Edge", "coords": [[20.0, -50.0], [20.0, 50.0]]}]), 0.1)
        memory.x, memory.last_distance = 10.0, 10.0
        memory.observe(status(observations=[{"type": "Edge", "coords": [[20.0, -60.0], [20.0, 40.0]]}]), 0.2)
        self.assertAlmostEqual(memory.x, 0.0)
        self.assertAlmostEqual(memory.y, 10.0)
        self.assertEqual(sum(map(len, memory.edges.values())), 1)

    def test_stationary_agent_does_not_move_when_fruit_changes(self):
        memory = AgentMemory()
        memory.observe(status(observations=[{"type": "Fruit", "distance": 20, "angle": 0}]), 0.1)
        memory.observe(status(observations=[{"type": "Fruit", "distance": 25, "angle": 0}]), 0.2)
        self.assertAlmostEqual(memory.x, 0)
        self.assertAlmostEqual(memory.y, 0)
        self.assertAlmostEqual(memory.fruits[0].x, 25)

    def test_localization_ties_prefer_consistent_odometry(self):
        memory = AgentMemory()
        memory.observe(status(observations=[
            {"type": "Fruit", "distance": 20, "angle": 0},
            {"type": "Fruit", "distance": 40, "angle": math.pi / 2},
        ]), 0.1)
        memory.x, memory.last_distance = 10, 10
        memory.observe(status(observations=[
            {"type": "Fruit", "distance": 25, "angle": 0},
            {"type": "Fruit", "distance": math.hypot(10, 40), "angle": math.atan2(40, -10)},
        ]), 0.2)
        self.assertAlmostEqual(memory.x, 10)
        self.assertAlmostEqual(memory.y, 0)

    def test_landmarks_disambiguate_parallel_wall_matches(self):
        memory = AgentMemory(x=20, last_distance=20, landmarks=[("Fruit", 30, 40)])
        memory.edges[(100, 0)] = [(10, -20, 110, -20), (10, 10, 110, 10)]
        memory.observe(status(observations=[
            {"type": "Edge", "coords": [[10, -10], [110, -10]]},
            {"type": "Fruit", "distance": math.hypot(30, 20), "angle": math.atan2(20, 30)},
        ]), 1)
        self.assertAlmostEqual(memory.x, 0)
        self.assertAlmostEqual(memory.y, 20)

    def test_eaten_fruit_removed_from_memory(self):
        memory = AgentMemory()
        memory.observe(status(observations=[{"type": "Fruit", "distance": 30.0, "angle": 0.0}]), 0.1)
        self.assertEqual(len(memory.fruits), 1)
        memory.observe(status(), 0.2)
        self.assertEqual(memory.fruits, [])

    def test_missing_visible_fruit_is_removed_before_reaching_it(self):
        memory = AgentMemory(prune_visible=True)
        memory.observe(status(observations=[{"type": "Fruit", "distance": 100, "angle": 0}]), 0.1)
        memory.observe(status(), 0.2)
        self.assertEqual(memory.fruits, [])

    def test_occluded_fruit_is_not_deleted(self):
        memory = AgentMemory(prune_visible=True, fruits=[Landmark(100, 0, 0, 0)])
        memory.observe(status(observations=[{"type": "Edge", "coords": [[50, -40], [50, 40]]}]), 0.1)
        self.assertEqual(len(memory.fruits), 1)

    def test_fruit_outside_vision_cone_is_not_deleted(self):
        memory = AgentMemory(prune_visible=True, fruits=[Landmark(0, 100, 0, 0)])
        memory.observe(status(), 0.1)
        self.assertEqual(len(memory.fruits), 1)

    def test_retry_is_idempotent(self):
        policy = SurvivalPolicy()
        agents = [status()]
        first = policy.decide(agents, 0.1)
        position = (policy.agents[0].x, policy.agents[0].y)
        second = policy.decide(agents, 0.1)
        self.assertEqual(first, second)
        self.assertEqual(position, (policy.agents[0].x, policy.agents[0].y))
        first[0]["move_distance"] = 999
        self.assertNotEqual(first, policy.decide(agents, 0.1))

    def test_same_time_different_observations_start_fresh(self):
        policy = SurvivalPolicy()
        policy.decide([status(observations=[{"type": "Fruit", "distance": 40, "angle": 0}])], 0.1)
        action = policy.decide([status(observations=[{"type": "Fruit", "distance": 40, "angle": 1}])], 0.1)[0]
        self.assertAlmostEqual(action["move_direction"], 1.0)

    def test_terminal_request_resets_endpoint(self):
        from agent_server import policy, predict

        predict(StepResponse(game_status="ok", score=0, sim_time=0.1, n_agents=1, agent_status=[status()]))
        result = predict(
            StepResponse(
                game_status="game_over", score=3000, sim_time=3000, n_agents=1, agent_status=[status()]
            )
        )
        self.assertEqual(result, {"actions": []})
        self.assertEqual(policy.agents, {})
        self.assertEqual(policy.statuses, {})
        self.assertIsNone(policy.last_time)

    def test_simulation_reset(self):
        policy = SurvivalPolicy()
        agents = [status()]
        expected = SurvivalPolicy().decide(agents, 0.1)
        policy.decide(agents, 100.0)
        self.assertEqual(expected, policy.decide(agents, 0.1))
        self.assertEqual([], policy.decide([], 0.0))
        self.assertEqual({}, policy.agents)

    def test_dead_agents_are_pruned(self):
        policy = SurvivalPolicy()
        policy.decide([status(0), status(1)], 0.1)
        policy.decide([status(1)], 0.2)
        self.assertEqual(set(policy.agents), {1})

    def test_mutated_traits_produce_finite_legal_actions(self):
        rng = random.Random(7)
        policy = SurvivalPolicy()
        for tick in range(100):
            agent = status(
                age=tick / 10,
                energy=rng.uniform(1, 500),
                speed=rng.uniform(0.1, 20),
                sprint_speed=rng.uniform(0.1, 40),
                hearing_radius=rng.uniform(1, 100),
                vision_angle=rng.uniform(0.05, math.pi / 2),
                vision_range=rng.uniform(1, 400),
                max_energy=rng.uniform(75, 1000),
            )
            action = policy.decide([agent], tick / 10)[0]
            ActionRequest(**action)
            self.assertTrue(
                all(math.isfinite(action[key]) for key in ("move_distance", "move_direction", "turn_angle"))
            )
            self.assertGreaterEqual(action["move_distance"], 0)
            self.assertLessEqual(action["move_distance"], agent["sprint_speed"])

    def test_wait_for_observed_new_fruit_to_ripen(self):
        policy = SurvivalPolicy(tuning={"ripen": 20})
        memory = AgentMemory(fruits=[Landmark(18, 0, 1, 1, born=0)])
        policy.agents[0] = memory
        action = policy.decide(
            [status(energy=150, observations=[{"type": "Fruit", "distance": 18, "angle": 0}])], 1
        )[0]
        self.assertEqual(action["move_distance"], 0)
        self.assertLess(memory.fruits[0].visited, 0)

    def test_prefer_ripe_fruit_over_waiting_for_nearest(self):
        policy = SurvivalPolicy(tuning={"ripen": 20, "food_planning": True})
        policy.agents[0] = AgentMemory(fruits=[
            Landmark(20, 0, 21, 21, born=20), Landmark(0, 60, 21, 1, born=0),
        ])
        action = policy.decide([status(energy=150, observations=[
            {"type": "Fruit", "distance": 20, "angle": 0},
            {"type": "Fruit", "distance": 60, "angle": math.pi / 2},
        ])], 21)[0]
        self.assertAlmostEqual(action["move_direction"], math.pi / 2)

    def test_hungry_agent_does_not_wait_for_ripening(self):
        policy = SurvivalPolicy(tuning={"ripen": 20})
        policy.agents[0] = AgentMemory(fruits=[Landmark(18, 0, 1, 1, born=0)])
        action = policy.decide(
            [status(energy=15, observations=[{"type": "Fruit", "distance": 18, "angle": 0}])], 1
        )[0]
        self.assertGreater(action["move_distance"], 0)

    def test_conserve_reserves_without_food(self):
        action = SurvivalPolicy(tuning={"reserve": 220}).decide([status(energy=240, age=70)], 70)[0]
        self.assertEqual(action["move_distance"], 0)

    def test_detect_senescence_from_energy_cost(self):
        policy = SurvivalPolicy(tuning={"renewal": True})
        policy.decide([status(age=80, energy=250)], 100)
        memory = policy.agents[0]
        policy.decide([status(age=80.1, energy=250 - memory.last_cost - 0.8)], 100.1)
        self.assertTrue(memory.senescent)

    def test_last_retired_agent_can_continue_the_lineage(self):
        policy = SurvivalPolicy(tuning={"renewal": True})
        policy.agents[0] = AgentMemory(retired=True, senescent=True)
        action = policy.decide([status(age=100, energy=200)], 100)[0]
        self.assertTrue(action["spawn_agent"])

    def test_curved_evasion_survives_sprint_exhaustion(self):
        from src.utils.controllers.predator_avoidance import maneuver_move, predict_predator

        for separation in (55, 70, 90):
            px, py, heading, energy = separation, 0.0, math.pi, 140.0
            for _ in range(50):
                speed, angle = maneuver_move(
                    [(px, py, heading, True, 1.0)], math.atan2(-py, -px),
                    10, 20 if energy > 108 else 10, 1.0, [], energy, 500,
                )
                ax, ay = speed * math.cos(angle), speed * math.sin(angle)
                px, py, heading = predict_predator(px, py, heading, ax, ay, math.atan2(py - ay, px - ax))
                px, py = px - ax, py - ay
                energy -= min(speed, 10) * 0.05 + max(0, speed - 10) * 0.5 + 0.1
                self.assertGreater(math.hypot(px, py), 15)
            self.assertLess(energy, 100)

    def test_failed_navigation_is_not_replanned_every_tick(self):
        from unittest.mock import patch
        from src.utils.controllers.navigation import Navigator

        navigator = Navigator()
        navigator.update([(30, -40, 30, 40)])
        with patch.object(navigator, "_search", return_value=[]) as search:
            self.assertIsNone(navigator.waypoint(0, 0, 70, 0, 0.1))
            self.assertIsNone(navigator.waypoint(0, 0, 70, 0, 0.2))
            self.assertEqual(search.call_count, 1)
            navigator.waypoint(0, 0, 70, 0, 1.2)
            self.assertEqual(search.call_count, 2)

    def test_navigator_routes_around_a_wall(self):
        from src.utils.controllers.navigation import Navigator

        navigator = Navigator()
        edge = (30, -40, 30, 40)
        navigator.update([edge])
        x, y = 0.0, 0.0
        for tick in range(50):
            waypoint = navigator.waypoint(x, y, 70, 0, tick / 10, stop=5)
            self.assertIsNotNone(waypoint)
            distance = math.hypot(waypoint[0] - x, waypoint[1] - y)
            if distance < 1:
                break
            scale = min(1, 8 / distance)
            nx, ny = x + (waypoint[0] - x) * scale, y + (waypoint[1] - y) * scale
            self.assertFalse(crosses(x, y, nx, ny, edge))
            self.assertGreater(segment_distance(nx, ny, edge), 5)
            x, y = nx, ny
        self.assertLess(math.hypot(x - 70, y), 10)

    def test_fast_mutants_can_forage_economically(self):
        action = SurvivalPolicy(tuning={"forage_speed": 10}).decide([
            status(speed=20, observations=[{"type": "Fruit", "distance": 100, "angle": 0}]),
        ], 0.1)[0]
        self.assertEqual(action["move_distance"], 10)

    def test_foraging_limit_does_not_limit_escape(self):
        action = SurvivalPolicy(tuning={"forage_speed": 10, "maneuvers": True}).decide([
            status(speed=20, energy=80, observations=[
                {"type": "Predator", "distance": 35, "angle": 0, "rel_dir": 0},
            ]),
        ], 0.1)[0]
        self.assertEqual(action["move_distance"], 20)

    def test_healthy_older_agents_can_still_camp(self):
        action = SurvivalPolicy(tuning={"patient": True}).decide([
            status(age=80, energy=100, observations=[{"type": "Tree", "distance": 20, "angle": 0}]),
        ], 80)[0]
        self.assertEqual(action["move_distance"], 0)

    def test_watch_predators_with_a_nonzero_looking_offset(self):
        action = SurvivalPolicy(tuning={"look_bias": 0.3}).decide([
            status(observations=[
                {"type": "Fruit", "distance": 40, "angle": 1.2},
                {"type": "Predator", "distance": 150, "angle": 0, "rel_dir": 0},
            ]),
        ], 0.1)[0]
        self.assertAlmostEqual(action["turn_angle"], 0.3)

    def test_fitness_respects_actual_speed_cap(self):
        policy = SurvivalPolicy(tuning={"selection": True})
        self.assertLess(policy._fitness(status(speed=20, sprint_speed=5)), policy._fitness(status()))

    def test_parent_yields_food_to_hungry_offspring(self):
        policy = SurvivalPolicy(tuning={"rationing": True})
        agents = [
            status(0, energy=180, observations=[
                {"type": "Fruit", "distance": 30, "angle": 0},
                {"type": "Agent", "id": 1, "distance": 10, "angle": math.pi, "rel_dir": 0},
            ]),
            status(1, energy=65, observations=[
                {"type": "Fruit", "distance": 40, "angle": 0},
                {"type": "Agent", "id": 0, "distance": 10, "angle": 0, "rel_dir": -math.pi},
            ]),
        ]
        actions = {a["agent_id"]: a for a in policy.decide(agents, 0.1)}
        self.assertEqual(actions[0]["move_distance"], 0)
        self.assertGreater(actions[1]["move_distance"], 0)

    def test_target_commitment_prevents_small_cost_oscillations(self):
        policy = SurvivalPolicy(tuning={"commitment": 25})
        fruit = Landmark(40, 0, 0, 0)
        policy.agents[0] = AgentMemory(fruits=[fruit], target=fruit, target_distance=40)
        action = policy.decide([status(energy=100, observations=[
            {"type": "Fruit", "distance": 40, "angle": 0},
            {"type": "Fruit", "distance": 35, "angle": math.pi},
        ])], 0.1)[0]
        self.assertAlmostEqual(action["move_direction"], 0)

    def test_aging_agent_can_recycle_reserves_without_long_delay(self):
        policy = SurvivalPolicy(tuning={"renewal": True, "late_renewal": True,
                                        "old_birth_energy": 105, "old_birth_interval": 0.5})
        policy.agents[0] = AgentMemory(senescent=True, last_birth=99)
        action = policy.decide([status(age=80, energy=110)], 100)[0]
        self.assertTrue(action["spawn_agent"])

    def test_spawn_budget_includes_movement_and_turning(self):
        from unittest.mock import patch

        policy = SurvivalPolicy(tuning={"renewal": True, "old_birth_energy": 105})
        policy.agents[0] = AgentMemory(senescent=True)
        with patch("src.utils.controllers.survival_policy.escape_move", return_value=(40, 0)):
            action = policy.decide([status(age=80, energy=117, speed=1, sprint_speed=40, observations=[
                {"type": "Predator", "distance": 80, "angle": 0, "rel_dir": 0},
            ])], 100)[0]
        self.assertFalse(action["spawn_agent"])
        self.assertFalse(policy.agents[0].retired)

    def test_unreachable_tree_does_not_trap_a_hungry_agent(self):
        policy = SurvivalPolicy(tuning={"tree_patience": 3})
        tree = Landmark(100, 0, 0, 0)
        policy.agents[0] = AgentMemory(trees=[tree], target=tree, target_distance=100)
        policy.decide([status(energy=50, observations=[{"type": "Tree", "distance": 100, "angle": 0}])], 4)
        self.assertEqual(tree.visited, 4)

    def test_tree_patience_does_not_interrupt_camping(self):
        policy = SurvivalPolicy(tuning={"tree_patience": 3})
        tree = Landmark(20, 0, 0, 0)
        policy.agents[0] = AgentMemory(trees=[tree], target=tree, target_distance=20)
        action = policy.decide([status(energy=50, observations=[{"type": "Tree", "distance": 20, "angle": 0}])], 4)[0]
        self.assertEqual(action["move_distance"], 0)
        self.assertLess(tree.visited, 0)

    def test_food_budget_tracks_observed_energy_gains(self):
        policy = SurvivalPolicy(tuning={"resource_budget": 4})
        policy.decide([status(energy=100, age=1)], 1)
        energy = 160 - policy.agents[0].last_cost
        agents = [status(energy=energy, age=1.1)]
        policy.decide(agents, 1.1)
        self.assertGreater(policy.harvest_rate, 1.9)
        self.assertLess(policy.harvest_rate, 2.1)
        rate = policy.harvest_rate
        policy.decide(agents, 1.1)
        self.assertEqual(policy.harvest_rate, rate)
        policy.reset()
        self.assertEqual(policy.harvest_rate, 0)

    def test_food_budget_limits_reproduction_during_scarcity(self):
        policy = SurvivalPolicy(tuning={"renewal": True, "resource_budget": 6})
        agents = [status(i, energy=350) for i in range(4)]
        self.assertFalse(any(a["spawn_agent"] for a in policy.decide(agents, 500)))
        policy = SurvivalPolicy(tuning={"renewal": True, "resource_budget": 6})
        policy.harvest_rate = 100
        self.assertTrue(any(a["spawn_agent"] for a in policy.decide(agents, 500)))

    def test_navigation_grid_can_preserve_narrow_corridors(self):
        from src.utils.controllers.navigation import Navigator

        navigator = Navigator(spacing=10, margin=5.5)
        navigator.update([(-6, -100, -6, 100), (6, -100, 6, 100)])
        self.assertNotIn((0, 0), navigator.blocked)
        self.assertIn((1, 0), navigator.blocked)
        self.assertIn((-1, 0), navigator.blocked)
        self.assertIsNotNone(navigator.waypoint(0, 0, 0, 80, 0.1))

    def test_selection_preserves_stronger_lineages(self):
        policy = SurvivalPolicy(tuning={"renewal": True, "selection": True, "selection_gap": 0.5})
        policy.agents[0] = AgentMemory(senescent=True)
        weak = status(0, age=100, energy=200, vision_range=50)
        agents = [weak] + [status(i, vision_range=400) for i in range(1, 5)]
        actions = {a["agent_id"]: a for a in policy.decide(agents, 100)}
        self.assertFalse(actions[0]["spawn_agent"])
        self.assertTrue(policy.agents[0].retired)
        policy = SurvivalPolicy(tuning={"renewal": True, "selection": True, "selection_gap": 0.5})
        policy.agents[0] = AgentMemory(senescent=True)
        self.assertTrue(policy.decide([weak], 100)[0]["spawn_agent"])

    def test_navigation_goal_requires_a_clear_food_approach(self):
        from src.utils.controllers.navigation import Navigator

        navigator = Navigator(spacing=10, margin=5.5)
        edges = [(30, -40, 70, -40), (70, -40, 70, 40), (30, 40, 70, 40), (30, -40, 30, 40)]
        navigator.update(edges)
        x, y = 20.0, 35.0
        for tick in range(30):
            waypoint = navigator.waypoint(x, y, 40, 50, tick / 10, stop=9.5)
            self.assertIsNotNone(waypoint)
            distance = math.hypot(waypoint[0] - x, waypoint[1] - y)
            if distance < 0.01:
                break
            ratio = min(1, 5 / distance)
            x += (waypoint[0] - x) * ratio
            y += (waypoint[1] - y) * ratio
            self.assertFalse(25 < x < 75 and -45 < y < 45)
        self.assertLess(math.hypot(x - 40, y - 50), 10)

    def test_well_fed_agents_need_not_travel_to_wait_beside_food(self):
        action = SurvivalPolicy(tuning={"stationary_reserve": 300}).decide([
            status(energy=350, observations=[{"type": "Fruit", "distance": 100, "angle": 0}]),
        ], 0.1)[0]
        self.assertEqual(action["move_distance"], 0)
        hungry = SurvivalPolicy(tuning={"stationary_reserve": 300}).decide([
            status(energy=100, observations=[{"type": "Fruit", "distance": 100, "angle": 0}]),
        ], 0.1)[0]
        self.assertGreater(hungry["move_distance"], 0)

    def test_fitness_can_reward_walking_faster_than_predators(self):
        policy = SurvivalPolicy(tuning={"vision_weight": 2, "escape_fitness": 3})
        fast = status(speed=16, vision_range=200)
        slow = status(speed=10, vision_range=400)
        self.assertGreater(policy._fitness(fast), policy._fitness(slow))

    def test_endpoint_contract(self):
        from agent_server import predict

        payload = StepResponse(game_status="ok", score=0, sim_time=0.1, n_agents=1, agent_status=[status()])
        result = predict(payload)
        self.assertEqual(set(result), {"actions"})
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(ActionRequest(**result["actions"][0]).agent_id, 0)


if __name__ == "__main__":
    unittest.main()
