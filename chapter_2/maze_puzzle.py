import copy


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 99999

    def set_parent(self, p):
        self.parent = p

    def set_cost(self, c):
        self.cost = c

    def print(self):
        print(self.x, ',', self.y)


NORTH = Point(0, 1)
SOUTH = Point(0, -1)
EAST = Point(1, 0)
WEST = Point(-1, 0)


class MazePuzzle:

    WALL = '#'
    EMPTY = '_'
    GOAL = '*'

    def __init__(self, maze_size_x=5, maze_size_y=5):
        self.maze_size_x = maze_size_x
        self.maze_size_y = maze_size_y
        self.maze = ['*0000',
                     '0###0',
                     '0#0#0',
                     '0#000',
                     '00000']

    def get_current_point_value(self, current_point):
        return self.maze[current_point.x][current_point.y]

    def get_neighbors(self, current_point):
        neighbors = []
        # potential_neighbors = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        potential_neighbors = [[SOUTH.x, SOUTH.y], [WEST.x, WEST.y], [NORTH.x, NORTH.y], [EAST.x, EAST.y]]
        for neighbor in potential_neighbors:
            target_point = Point(current_point.x + neighbor[0], current_point.y + neighbor[1])
            if 0 <= target_point.x < self.maze_size_x and 0 <= target_point.y < self.maze_size_y:
                if self.get_current_point_value(target_point) != '#':
                    neighbors.append(target_point)
        return neighbors

    def overlay_points_on_map(self, points):
        overlay_map = copy.deepcopy(self.maze)
        for point in points:
            new_row = overlay_map[point.x][:point.y] + '@' + overlay_map[point.x][point.y + 1:]
            overlay_map[point.x] = new_row

        result = ''
        for x in range(0, self.maze_size_x):
            for y in range(0, self.maze_size_y):
                result += overlay_map[x][y]
            result += '\n'
        print(result)

        return overlay_map

    def print_maze(self):
        result = ''
        for x in range(0, self.maze_size_x):
            for y in range(0, self.maze_size_y):
                result += self.maze[x][y]
            result += '\n'
        print(result)


def get_path(point):
    path = []
    current_point = point
    while current_point.parent is not None:
        path.append(current_point)
        current_point = current_point.parent
    return path


def get_path_length(point):
    path = []
    current_point = point
    total_length = 0
    while current_point.parent is not None:
        path.append(current_point)
        total_length += 1
        current_point = current_point.parent
    return total_length


def get_path_cost(point):
    path = []
    current_point = point
    total_cost = 0
    while current_point.parent is not None:
        path.append(current_point)
        total_cost += get_cost(get_direction(current_point.parent, current_point))
        current_point = current_point.parent
    return total_cost


def get_move_cost(origin, target):
    return get_cost(get_direction(origin, target))


def get_direction(origin, target):
    if target.x == origin.x and target.y == origin.y - 1:
        return 'N'
    elif target.x == origin.x and target.y == origin.y + 1:
        return 'S'
    elif target.x == origin.x + 1 and target.y == origin.y:
        return 'E'
    else:
        return 'W'


def get_cost(direction):
    if direction == 'N' or direction == 'S':
        return 5
    return 1
