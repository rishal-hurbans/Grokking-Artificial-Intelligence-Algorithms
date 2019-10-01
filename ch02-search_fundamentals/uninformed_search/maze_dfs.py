import maze_puzzle as mp


def run_dfs(maze_game, current_point):
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
                    stack.append(neighbor)
    return "No goal found"


def is_in_visited_points(current_point, visited_points):
    for visited_point in visited_points:
        if current_point.x == visited_point.x and current_point.y == visited_point.y:
            return True
    return False


maze_game_main = mp.MazePuzzle()
outcome = run_dfs(maze_game_main, mp.Point(2, 2))
outcome.print()
dfs_path = mp.get_path(outcome)
print('PATH LENGTH: ', len(dfs_path))
maze_game_main.overlay_points_on_map(dfs_path)
print('PATH COST: ', mp.get_path_cost(outcome))
for point in dfs_path:
    print('Point: ', point.x, ',', point.y)
