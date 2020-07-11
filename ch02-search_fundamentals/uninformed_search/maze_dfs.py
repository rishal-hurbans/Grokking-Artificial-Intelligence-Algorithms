import maze_puzzle as mp


# Function to find a route using the Depth-first Search algorithm.

# Depth-first search is an algorithm used to traverse a tree or generate nodes and paths in a tree. This algorithm
# starts at a specific node and explores paths of connected nodes of the first child and does this recursively until
# it reaches the furthest leaf node before backtracking and exploring other paths to leaf nodes via other child nodes
# that have been visited.

# Although the Depth-first search algorithm van be implemented with a recursive function. This implementation is
# achieved using a stack to better represent the order of operations as to which nodes get visited and processed.
# It is important to keep track of the visited points so that the same nodes do not get visited unnecessarily and
# create cyclic loops.
def run_dfs(maze_game, current_point):
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
                # Add the current node's neighbors to the stack
                neighbors = maze_game.get_neighbors(next_point)
                for neighbor in neighbors:
                    neighbor.set_parent(next_point)
                    stack.append(neighbor)
    return 'No path to the goal found.'


# Function to determine if the point has already been visited
def is_in_visited_points(current_point, visited_points):
    for visited_point in visited_points:
        if current_point.x == visited_point.x and current_point.y == visited_point.y:
            return True
    return False


print('---Depth-first Search---')

# Initialize a MazePuzzle
maze_game_main = mp.MazePuzzle()

# Run the depth first search algorithm with the initialized maze
starting_point = mp.Point(2, 2)
outcome = run_dfs(maze_game_main, starting_point)

# Get the path found by the depth first search algorithm
dfs_path = mp.get_path(outcome)

# Print the results of the path found
print('Path Length: ', len(dfs_path))
maze_game_main.overlay_points_on_map(dfs_path)
print('Path Cost: ', mp.get_path_cost(outcome))
for point in dfs_path:
    print('Point: ', point.x, ',', point.y)
