from collections import deque
from uninformed_search import maze_puzzle as mp


def run_bfs(maze_puzzle, current_point, visited_points):
    queue = deque()
    queue.append(current_point)
    visited_points.append(current_point)
    while queue:
        current_point = queue.popleft()
        neighbors = maze_puzzle.get_neighbors(current_point)
        for neighbor in neighbors:
            if not is_in_visited_points(neighbor, visited_points):
                neighbor.set_parent(current_point)
                queue.append(neighbor)
                visited_points.append(neighbor)
                if maze_puzzle.get_current_point_value(neighbor) == '*':
                    return neighbor
    return "No path"


def is_in_visited_points(current_point, visited_points):
    for visited_point in visited_points:
        if current_point.x == visited_point.x and current_point.y == visited_point.y:
            return True
    return False


print("---BFS---")
maze_game_main = mp.MazePuzzle()
outcome = run_bfs(maze_game_main, mp.Point(2, 2), [])
bfs_path = mp.get_path(outcome)
print('PATH LENGTH: ', len(bfs_path))
maze_game_main.overlay_points_on_map(bfs_path)
for point in bfs_path:
    print('Point: ', point.x, ',', point.y)
