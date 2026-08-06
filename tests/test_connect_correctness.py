import unittest
from unittest import mock

from test_helpers import load_module


class ConnectCorrectnessTests(unittest.TestCase):
    def test_detects_ai_anti_diagonal_win(self):
        puzzle = load_module(
            "connect_puzzle",
            "ch03-intelligent_search/adversarial_search/connect_puzzle.py",
        )
        game = puzzle.Connect(board_size_x=4, board_size_y=4)
        game.board = ["___A", "__A_", "_A__", "A___"]

        self.assertTrue(game.has_a_row(puzzle.PLAYER_AI, 4))
        self.assertEqual("AI won", game.has_winner())

    def test_minimax_returns_immediate_root_move(self):
        ai = load_module(
            "connect_ai",
            "ch03-intelligent_search/adversarial_search/connect_ai.py",
        )

        with mock.patch.object(ai.random, "sample", return_value=[0, 1]):
            result = ai.minmax(FakeConnect(), 2, ai.MAX, 999)

        self.assertEqual(0, result.move)
        self.assertEqual(10, result.value)

    def test_alpha_beta_returns_immediate_root_move(self):
        ai = load_module(
            "connect_ai_alpha_beta_pruning",
            "ch03-intelligent_search/adversarial_search/"
            "connect_ai_alpha_beta_pruning.py",
        )

        with mock.patch.object(ai.random, "sample", return_value=[0, 1]):
            result = ai.minmax(
                FakeConnect(),
                2,
                ai.MAX,
                999,
                ai.INFINITY_NEGATIVE,
                ai.INFINITY_POSITIVE,
            )

        self.assertEqual(0, result.move)
        self.assertEqual(10, result.value)


class FakeConnect:
    board_size_x = 2
    board_size_y = 1

    def __init__(self, path=()):
        self.path = path

    def __deepcopy__(self, memo):
        return FakeConnect(self.path)

    def get_score_for_ai(self):
        if self.path == (0, 1):
            return 10
        if self.path == (1, 0):
            return -10
        return 0

    def is_board_full(self):
        return len(self.path) >= 2

    def play_move(self, slot):
        if self.path == () and slot in (0, 1):
            self.path = (slot,)
            return True
        if self.path == (0,) and slot == 1:
            self.path = (0, 1)
            return True
        if self.path == (1,) and slot == 0:
            self.path = (1, 0)
            return True
        return False


if __name__ == "__main__":
    unittest.main()
