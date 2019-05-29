import maze_puzzle as mp


def run_a_star_search(maze_game, current_point):
    visited_points = []
    stack = [current_point]
    while stack:
        next_point = stack.pop()
        if not is_in_visited_points(next_point, visited_points):
            visited_points.append(next_point)
            if maze_game.get_current_point_value(next_point) == '*':
                return next_point
            else:
                neighbors = maze_game.get_neighbors(next_point)
                for neighbor in neighbors:
                    neighbor.set_parent(next_point)
                    neighbor.cost = set_cost(next_point, neighbor)
                stack.extend(neighbors)
                stack.sort(key=lambda x: x.cost, reverse=True)
    return "No goal found"


def set_cost(origin, target):
    distance_to_root = mp.get_path_length(target)
    cost = mp.get_move_cost(origin, target)
    return distance_to_root + cost


def is_in_visited_points(current_point, visited_points):
    for visited_point in visited_points:
        if current_point.x == visited_point.x and current_point.y == visited_point.y:
            return True
    return False


print("---A* Search---")
maze_game_main = mp.MazePuzzle()
outcome = run_a_star_search(maze_game_main, mp.Point(2, 2))
a_star_path = mp.get_path(outcome)
print('PATH LENGTH: ', mp.get_path_length(outcome))
maze_game_main.overlay_points_on_map(a_star_path)
print('PATH COST: ', mp.get_path_cost(outcome))
print('PATH POINTS:')
for point in a_star_path:
    print('Point: ', point.x, ',', point.y)
