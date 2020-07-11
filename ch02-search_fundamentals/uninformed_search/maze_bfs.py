from collections import deque
import maze_puzzle as mp


# Function to find a route using the Breadth-first Search algorithm.
# Breadth-first search is an algorithm used to traverse or generate a tree. This algorithm starts at a specific node
# called the root and explores every node at that depth before exploring the next depth of nodes. It essentially visits
# all children of nodes at a specific depth before visiting the next depth of child until it finds a goal leaf node.

# The Breadth first search algorithm is best implemented using a first-in-first-out queue where the current depth of
# nodes are processed and their children are queued to be processed later. This order of processing is exactly what
# we require when implementing this algorithm.
def run_bfs(maze_puzzle, current_point, visited_points):
    queue = deque()
    # Append the current node to the queue
    queue.append(current_point)
    visited_points.append(current_point)
    # Keep searching while there are nodes in the queue
    while queue:
        # Set the next node in the queue as the current node
        current_point = queue.popleft()
        # Get the neighbors of the current node
        neighbors = maze_puzzle.get_neighbors(current_point)
        # Iterate through the neighbors of the current node
        for neighbor in neighbors:
            # Add the neighbor to the queue if it hasn't been visited
            if not is_in_visited_points(neighbor, visited_points):
                neighbor.set_parent(current_point)
                queue.append(neighbor)
                visited_points.append(neighbor)
                # Return the path to the current neighbor if it is the goal
                if maze_puzzle.get_current_point_value(neighbor) == '*':
                    return neighbor
    # In the case that no path to the goal was found
    return 'No path to the goal found.'


# Function to determine if the point has already been visited
def is_in_visited_points(current_point, visited_points):
    for visited_point in visited_points:
        if current_point.x == visited_point.x and current_point.y == visited_point.y:
            return True
    return False


print('---Breadth-first Search---')

# Initialize a MazePuzzle
maze_game_main = mp.MazePuzzle()

# Run the breadth first search algorithm with the initialized maze
starting_point = mp.Point(2, 2)
outcome = run_bfs(maze_game_main, starting_point, [])

# Get the path found by the breadth first search algorithm
bfs_path = mp.get_path(outcome)

# Print the results of the path found
print('Path Length: ', len(bfs_path))
maze_game_main.overlay_points_on_map(bfs_path)
print('Path Cost: ', mp.get_path_cost(outcome))
for point in bfs_path:
    print('Point: ', point.x, ',', point.y)
