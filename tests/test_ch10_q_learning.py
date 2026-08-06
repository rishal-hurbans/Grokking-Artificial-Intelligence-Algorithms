import io
import re
import unittest
from contextlib import redirect_stdout

from test_helpers import load_module


class Chapter10QLearningTests(unittest.TestCase):
    def test_get_state_changes_with_agent_location(self):
        simulator_module = load_module(
            "auto_simulator",
            "ch10-reinforcement_learning/auto_simulator.py",
        )
        simulator = simulator_module.Simulator(
            simulator_module.DEFAULT_ROAD,
            simulator_module.DEFAULT_ROAD_SIZE_X,
            simulator_module.DEFAULT_ROAD_SIZE_Y,
            simulator_module.DEFAULT_START_X,
            simulator_module.DEFAULT_START_Y,
            simulator_module.DEFAULT_GOAL_X,
            simulator_module.DEFAULT_GOAL_Y,
        )

        origin_state = simulator.get_state()
        simulator.move(5, 5)
        moved_state = simulator.get_state()

        self.assertNotEqual(origin_state, moved_state)

    def test_average_car_penalty_report_uses_car_penalty_count(self):
        simulator_module = load_module(
            "auto_simulator",
            "ch10-reinforcement_learning/auto_simulator.py",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            simulator_module.print_q_learning_summary(
                number_of_episodes=3,
                total_epochs=12,
                total_penalties_person=30,
                total_penalties_car=6,
            )

        match = re.search(
            r"Average car penalties per episode:\s+([0-9.]+)",
            output.getvalue(),
        )
        self.assertIsNotNone(match)
        self.assertEqual(2.0, float(match.group(1)))


if __name__ == "__main__":
    unittest.main()
