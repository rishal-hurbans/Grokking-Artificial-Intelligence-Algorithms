import copy
import random

MIN = -1
MAX = 1


class Move:

    def __init__(self, move=0, value=0):
        self.move = move
        self.value = value


def choose_move(connect, depth):
    return minmax(connect, depth, 1, -1).move


def minmax(connect, depth, min_or_max, move):
    current_score = connect.get_score_for_ai()
    current_is_board_full = connect.is_board_full()
    if current_score != 0 or current_is_board_full or depth == 0:
        return Move(move, current_score)
    if min_or_max == MAX:
        best_max_score = -100
        best_max_move = -1
        moves = random.sample(range(0, 5), 5)
        for slot in moves:
            neighbor = copy.deepcopy(connect)
            move_outcome = neighbor.play_move(slot)
            if move_outcome:
                best = minmax(neighbor, depth - 1, -1, slot)
                if best.value > best_max_score:
                    best_max_score = best.value
                    best_max_move = slot
        return Move(best_max_move, best_max_score)
    elif min_or_max == MIN:
        best_min_score = 100
        best_min_move = -1
        moves = random.sample(range(0, 5), 5)
        for slot in moves:
            neighbor = copy.deepcopy(connect)
            move_outcome = neighbor.play_move(slot)
            if move_outcome:
                best = minmax(neighbor, depth - 1, 1, slot)
                if best.value < best_min_score:
                    best_min_score = best.value
                    best_min_move = slot
        return Move(best_min_move, best_min_score)
    return Move(0, 0)
