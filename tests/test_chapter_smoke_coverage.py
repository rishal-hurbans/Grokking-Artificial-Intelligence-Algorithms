import ast
import contextlib
import io
import math
import pathlib
import sys
import tempfile
import unittest
import warnings

from test_helpers import (
    ROOT,
    load_module,
    load_module_from_source,
)


class ChapterSmokeCoverageTests(unittest.TestCase):
    def test_all_chapters_have_readmes_and_summary_assets(self):
        for chapter in range(1, 13):
            matches = sorted(ROOT.glob(f"ch{chapter:02d}-*/README.md"))
            self.assertEqual(1, len(matches), f"Chapter {chapter} README missing")
            chapter_root = matches[0].parent
            summary = chapter_root / "readme_assets" / f"Ch{chapter}-Summary.png"
            self.assertTrue(summary.exists(), f"Chapter {chapter} summary asset missing")

    def test_chapter_2_bfs_and_dfs_find_the_maze_goal(self):
        sys.modules.pop("maze_puzzle", None)
        maze = load_module(
            "ch02_maze_puzzle",
            "ch02-search_fundamentals/uninformed_search/maze_puzzle.py",
        )
        sys.modules["maze_puzzle"] = maze
        bfs = load_module_from_source(
            "ch02_maze_bfs",
            "ch02-search_fundamentals/uninformed_search/maze_bfs.py",
            lambda node: getattr(node, "lineno", 0) >= 46,
        )
        dfs = load_module_from_source(
            "ch02_maze_dfs",
            "ch02-search_fundamentals/uninformed_search/maze_dfs.py",
            lambda node: getattr(node, "lineno", 0) >= 46,
        )

        puzzle = maze.MazePuzzle()
        start = maze.Point(2, 2)
        bfs_goal = bfs.run_bfs(puzzle, start, [])
        dfs_goal = dfs.run_dfs(puzzle, start)

        self.assertEqual(maze.MazePuzzle.GOAL, puzzle.get_current_point_value(bfs_goal))
        self.assertEqual(maze.MazePuzzle.GOAL, puzzle.get_current_point_value(dfs_goal))
        self.assertGreater(maze.get_path_length(bfs_goal), 0)
        self.assertGreater(maze.get_path_length(dfs_goal), 0)

    def test_chapter_3_astar_finds_the_maze_goal(self):
        sys.modules.pop("maze_puzzle", None)
        maze = load_module(
            "ch03_maze_puzzle",
            "ch03-intelligent_search/informed_search/maze_puzzle.py",
        )
        sys.modules["maze_puzzle"] = maze
        astar = load_module_from_source(
            "ch03_maze_astar",
            "ch03-intelligent_search/informed_search/maze_astar.py",
            lambda node: getattr(node, "lineno", 0) >= 50,
        )

        puzzle = maze.MazePuzzle()
        goal = astar.run_astar(puzzle, maze.Point(2, 2))

        self.assertEqual(maze.MazePuzzle.GOAL, puzzle.get_current_point_value(goal))
        self.assertGreater(maze.get_path_length(goal), 0)

    def test_chapter_7_pso_booth_function_has_known_minimum(self):
        pso = load_module_from_source(
            "drone_pso_no_demo",
            "ch07-swarm_intelligence-particles/drone_pso.py",
            lambda node: (
                isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == "swarm" for target in node.targets)
            ) or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and getattr(node, "lineno", 0) >= 196
            ),
        )

        self.assertEqual(0, pso.calculate_booth(1, 3))
        particle = pso.Particle(1, 3, pso.INERTIA, pso.COGNITIVE_CONSTANT, pso.SOCIAL_CONSTANT)
        self.assertEqual(0, particle.get_fitness())

    def test_chapter_8_data_preparation_encodes_and_splits_csv(self):
        data_module = load_module(
            "ml_data_preparation",
            "ch08-machine_learning/ml_data_preparation.py",
        )
        csv_rows = (
            "no,carat,cut,color,clarity,price\n"
            "1,0.2,Fair,D,SI1,300\n"
            "2,0.3,Good,E,VS1,400\n"
            "3,0.4,Fair,D,SI1,500\n"
            "4,0.5,Good,E,VS1,600\n"
        )
        encodings = {
            "cut": {"Fair": 1, "Good": 2},
            "color": {"D": 7, "E": 6},
            "clarity": {"SI1": 5, "VS1": 7},
        }

        with tempfile.TemporaryDirectory() as directory:
            csv_path = pathlib.Path(directory) / "diamonds.csv"
            csv_path.write_text(csv_rows)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                data = data_module.Data(str(csv_path), ["no"], "cut", encodings)

        self.assertEqual(4, len(data.X_train) + len(data.X_test))
        self.assertNotIn("cut", data.X_train.columns)
        self.assertNotIn("no", data.X_train.columns)
        self.assertTrue(set(data.y_train).union(data.y_test).issubset({1, 2}))

    def test_chapter_9_ann_scaling_and_forward_shape(self):
        ann = load_module(
            "car_ann",
            "ch09-artificial_neural_networks/car_ann.py",
        )
        dataset = ann.np.array([[0, 5], [10, 10]])
        scaled = ann.scale_dataset(dataset, 2, [0, 0], [10, 10])

        self.assertEqual([[0.0, 0.5], [1.0, 1.0]], scaled.tolist())

        labels = ann.np.array([[0], [1]])
        with contextlib.redirect_stdout(io.StringIO()):
            network = ann.NeuralNetwork(scaled, labels, hidden_node_count=2)
        network.forward_propagation()

        self.assertEqual(labels.shape, network.output.shape)

    def test_chapter_11_tokenization_helpers(self):
        needed_functions = {
            "character_tokenizer",
            "bigram_frequency_table",
            "merge_bigram",
            "tokens_to_id_stream",
            "pack_id_stream",
        }
        llm = load_module_from_source(
            "toy_llm_transformer_defs",
            "ch11-large_language_models/toy_llm_transformer.py",
            lambda node: not (
                (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "collections"
                )
                or (
                    isinstance(node, ast.FunctionDef)
                    and node.name in needed_functions
                )
            ),
        )

        tokens = llm.character_tokenizer("banana")
        merged = llm.merge_bigram(tokens, ("a", "n"))
        id_stream, vocab = llm.tokens_to_id_stream(merged)
        blocks = llm.pack_id_stream(id_stream, block_size=4, vocab=vocab)

        self.assertEqual(["b", "an", "an", "a"], merged)
        self.assertEqual(2, llm.bigram_frequency_table(tokens)[("a", "n")])
        self.assertEqual([[0, 1, 1, 2]], blocks)
        self.assertIn("<PAD>", vocab)

    def test_chapter_12_numpy_timestep_embedding_shape(self):
        unet = load_module_from_source(
            "toy_numpy_u_net_generate_defs",
            "ch12-generative_image_models/numpy_u_net/toy_numpy_u_net_generate.py",
            lambda node: not (
                (
                    isinstance(node, ast.Import)
                    and any(alias.name in {"math", "numpy"} for alias in node.names)
                )
                or (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "create_timestep_embedding"
                )
            ),
        )

        embedding = unet.create_timestep_embedding(3, 16)

        self.assertEqual((16,), embedding.shape)
        self.assertTrue(all(math.isfinite(value) for value in embedding))


if __name__ == "__main__":
    unittest.main()
