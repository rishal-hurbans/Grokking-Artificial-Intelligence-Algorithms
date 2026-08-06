import ast
import unittest
from unittest import mock

from test_helpers import ROOT, load_module_from_source


GA_PATH = "ch04-evolutionary_algorithms/knapsack_genetic_algorithm.py"


def load_ga_without_demo():
    return load_module_from_source(
        "knapsack_genetic_algorithm_no_demo",
        GA_PATH,
        lambda node: isinstance(node, ast.For) and node.lineno >= 219,
    )


class GeneticAlgorithmReproductionTests(unittest.TestCase):
    def test_ga_demo_execution_is_guarded(self):
        source = (ROOT / GA_PATH).read_text()

        self.assertIn("__main__", source)

    def test_crossover_receives_chromosome_strings_not_individual_records(self):
        ga = load_ga_without_demo()
        parents = [
            ["11110000", 10, 0],
            ["00001111", 8, 0],
        ]

        with mock.patch.object(
            ga,
            "one_point_crossover",
            return_value=["11111111", "00000000"],
        ) as crossover:
            children = ga.reproduce_children(parents)

        first_parent, second_parent = crossover.call_args.args[:2]
        self.assertIsInstance(first_parent, str)
        self.assertIsInstance(second_parent, str)
        self.assertEqual("11110000", first_parent)
        self.assertEqual("00001111", second_parent)
        self.assertEqual([["11111111", 0, 0], ["00000000", 0, 0]], children)

    def test_reproduction_accumulates_non_overlapping_parent_pairs(self):
        ga = load_ga_without_demo()
        parents = [
            ["aaaa", 10, 0],
            ["bbbb", 9, 0],
            ["cccc", 8, 0],
            ["dddd", 7, 0],
            ["eeee", 6, 0],
            ["ffff", 5, 0],
        ]

        with mock.patch.object(
            ga,
            "one_point_crossover",
            side_effect=[
                ["ab", "ba"],
                ["cd", "dc"],
                ["ef", "fe"],
            ],
        ) as crossover:
            children = ga.reproduce_children(parents)

        parent_pairs = [call.args[:2] for call in crossover.call_args_list]
        self.assertEqual([("aaaa", "bbbb"), ("cccc", "dddd"), ("eeee", "ffff")], parent_pairs)
        self.assertEqual(
            [
                ["ab", 0, 0],
                ["ba", 0, 0],
                ["cd", 0, 0],
                ["dc", 0, 0],
                ["ef", 0, 0],
                ["fe", 0, 0],
            ],
            children,
        )

    def test_mutation_preserves_chromosome_type_and_length(self):
        ga = load_ga_without_demo()
        children = [["101010", 0, 0]]

        with mock.patch.object(ga.random, "randint", return_value=2):
            mutated = ga.mutate_children(children, mutation_rate=5)

        self.assertIsInstance(mutated[0][0], str)
        self.assertEqual(6, len(mutated[0][0]))
        self.assertEqual([mutated[0][0], 0, 0], mutated[0])


if __name__ == "__main__":
    unittest.main()
