import ast
import contextlib
import io
import tempfile
import unittest
from unittest import mock

from test_helpers import load_module, load_module_from_source, temporary_cwd


ACO_PATH = "ch06-swarm_intelligence-ants/carnival_aco.py"


def load_aco_without_demo():
    def skip_node(node):
        if node.lineno < 185:
            return False
        if isinstance(node, ast.Assign):
            return any(getattr(target, "id", None) == "aco" for target in node.targets)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
        return False

    return load_module_from_source(
        "carnival_aco_no_demo",
        ACO_PATH,
        skip_node,
    )


class AntColonyCorrectnessTests(unittest.TestCase):
    def test_carnival_aco_import_is_cwd_independent_and_does_not_solve(self):
        with tempfile.TemporaryDirectory() as directory:
            with temporary_cwd(directory):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    module = load_module(
                        "carnival_aco_import_check",
                        ACO_PATH,
                    )

        self.assertTrue(hasattr(module, "ACO"))
        self.assertNotIn("Best distance", output.getvalue())

    def test_random_fallback_selects_unvisited_attraction_id(self):
        aco = load_aco_without_demo()
        aco.ATTRACTION_COUNT = 5
        ant = aco.Ant()
        ant.visited_attractions = [0, 1, 2]

        with mock.patch.object(aco.random, "randint", return_value=1):
            selected = ant.visit_random_attraction()

        self.assertEqual(4, selected)

    def test_pheromone_update_only_reinforces_traversed_edges(self):
        aco = load_aco_without_demo()
        aco.ATTRACTION_COUNT = 4
        aco.attraction_distances = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
        colony = aco.ACO(number_of_ants_factor=0)
        colony.pheromone_trails = [[1.0 for _ in range(4)] for _ in range(4)]
        ant = aco.Ant()
        ant.visited_attractions = [0, 1, 2, 3]
        colony.ant_colony = [ant]

        colony.update_pheromones(evaporation_rate=0.5)

        self.assertGreater(colony.pheromone_trails[0][1], 0.5)
        self.assertGreater(colony.pheromone_trails[1][2], 0.5)
        self.assertGreater(colony.pheromone_trails[2][3], 0.5)
        self.assertGreater(colony.pheromone_trails[3][0], 0.5)
        self.assertEqual(0.5, colony.pheromone_trails[0][2])


if __name__ == "__main__":
    unittest.main()
