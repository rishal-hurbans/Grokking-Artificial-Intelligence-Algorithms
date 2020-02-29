import maze_puzzle as mp


# Function to find a route using A* Search algorithm.

# The A* algorithm usually improves performance by estimating heuristics to minimize cost of the next node visited.
# Total cost is calculated using two metrics: the total distance from the start node to the current node, and the
# estimated cost of moving to a specific node by utilizing a heuristic. When attempting to minimize cost, a lower
# value will indicate a better performing solution.
def run_astar(maze_game, current_point):
    # Append the current node to the stack
    visited_points = []
    stack = [current_point]
    # Keep searching while there are nodes in the stack
    while stack:
        # Set the next node in the stack as the current node
        next_point = stack.pop()
        # If the current node hasn't already been exploited, search it
        if not is_in_visited_points(next_point, visited_points):
            visited_points.append(next_point)
            # Return the path to the current neighbor if it is the goal
            if maze_game.get_current_point_value(next_point) == '*':
                return next_point
            else:
                # Add all the current node's neighbors to the stack
                neighbors = maze_game.get_neighbors(next_point)
                for neighbor in neighbors:
                    neighbor.set_parent(next_point)
                    neighbor.cost = determine_cost(next_point, neighbor)
                stack.extend(neighbors)
                stack.sort(key=lambda x: x.cost, reverse=True)
    return "No path to the goal found"


# Determine cost based on the distance to root
def determine_cost(origin, target):
    distance_to_root = mp.get_path_length(target)
    cost = mp.get_move_cost(origin, target)
    return distance_to_root + cost


# Function to determine if the point has already been visited
def is_in_visited_points(current_point, visited_points):
    for visited_point in visited_points:
        if current_point.x == visited_point.x and current_point.y == visited_point.y:
            return True
    return False


print('---A* Search---')

# Function to determine if the point has already been visited
maze_game_main = mp.MazePuzzle()

# Run the greedy search algorithm with the initialized maze
outcome = run_astar(maze_game_main, mp.Point(2, 2))

# Get the path found by the greedy search algorithm
astar_path = mp.get_path(outcome)

# Print the results of the path found
print('PATH LENGTH: ', mp.get_path_length(outcome))
maze_game_main.overlay_points_on_map(astar_path)
print('PATH COST: ', mp.get_path_cost(outcome))
for point in astar_path:
    print('Point: ', point.x, ',', point.y)
