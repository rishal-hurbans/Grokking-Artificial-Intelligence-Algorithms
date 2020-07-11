import csv
import random
import math

# Ant Colony Optimization (ACO)
# The Ant Colony Optimization algorithm is inspired by the behavior of ants moving between destinations, dropping
# pheromones and acting on pheromones that they come across. The emergent behavior is ants converging to paths of
# least resistance.

# Set the number of attractions in the data set
# Best total distance for 5 attractions: 19
# Best total distance for 48 attractions: 33523
ATTRACTION_COUNT = 48
# Initialize the 2D matrix for storing distances between attractions
attraction_distances = []
# Read attraction distance data set store it in matrix
with open('attractions-' + str(ATTRACTION_COUNT) + '.csv') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        attraction_distances.append(row)

# Set the probability of ants choosing a random attraction to visit (0.0 - 1.0)
RANDOM_ATTRACTION_FACTOR = 0.0
# Set the weight for pheromones on path for selection
ALPHA = 4
# Set the weight for heuristic of path for selection
BETA = 7


# The Ant class encompasses the idea of an ant in the ACO algorithm.
# Ants will move to different attractions and leave pheromones behind. Ants will also make a judgement about which
# attraction to visit next. And lastly, ants will have knowledge about their respective total distance travelled.
# - Memory: In the ACO algorithm, this is the list of attractions already visited.
# - Best fitness: The shortest total distance travelled across all attractions.
# - Action: Choose the next destination to visit and drop pheromones along the way.
class Ant:

    # The ant is initialized to a random attraction with no previously visited attractions
    def __init__(self):
        self.visited_attractions = []
        self.visited_attractions.append(random.randint(0, ATTRACTION_COUNT - 1))

    # Select an attraction using a random chance or ACO function
    def visit_attraction(self, pheromone_trails):
        if random.random() < RANDOM_ATTRACTION_FACTOR:
            self.visited_attractions.append(self.visit_random_attraction())
        else:
            self.visited_attractions.append(
                self.roulette_wheel_selection(self.visit_probabilistic_attraction(pheromone_trails)))

    # Select an attraction using a random chance
    def visit_random_attraction(self):
        all_attractions = set(range(0, ATTRACTION_COUNT))
        possible_attractions = all_attractions - set(self.visited_attractions)
        return random.randint(0, len(possible_attractions) - 1)

    # Calculate probabilities of visiting adjacent unvisited attractions
    def visit_probabilistic_attraction(self, pheromone_trails):
        current_attraction = self.visited_attractions[-1]
        all_attractions = set(range(0, ATTRACTION_COUNT))
        possible_attractions = all_attractions - set(self.visited_attractions)
        possible_indexes = []
        possible_probabilities = []
        total_probabilities = 0
        for attraction in possible_attractions:
            possible_indexes.append(attraction)
            pheromones_on_path = math.pow(pheromone_trails[current_attraction][attraction], ALPHA)
            heuristic_for_path = math.pow(1 / attraction_distances[current_attraction][attraction], BETA)
            probability = pheromones_on_path * heuristic_for_path
            possible_probabilities.append(probability)
            total_probabilities += probability
        possible_probabilities = [probability / total_probabilities for probability in possible_probabilities]
        return [possible_indexes, possible_probabilities, len(possible_attractions)]

    # Select an attraction using the probabilities of visiting adjacent unvisited attractions
    @staticmethod
    def roulette_wheel_selection(probabilities):
        slices = []
        total = 0
        possible_indexes = probabilities[0]
        possible_probabilities = probabilities[1]
        possible_attractions_count = probabilities[2]
        for i in range(0, possible_attractions_count):
            slices.append([possible_indexes[i], total, total + possible_probabilities[i]])
            total += possible_probabilities[i]
        spin = random.random()
        result = [s[0] for s in slices if s[1] < spin <= s[2]]
        return result[0]

    # Get the total distance travelled by this ant
    def get_distance_travelled(self):
        total_distance = 0
        for a in range(1, len(self.visited_attractions)):
            total_distance += attraction_distances[self.visited_attractions[a]][self.visited_attractions[a-1]]
        total_distance += attraction_distances[self.visited_attractions[0]][self.visited_attractions[len(self.visited_attractions) - 1]]
        return total_distance

    def print_info(self):
        print('Ant ', self.__hash__())
        print('Total attractions: ', len(self.visited_attractions))
        print('Total distance: ', self.get_distance_travelled())


# The ACO class encompasses the functions for the ACO algorithm consisting of many ants and attractions to visit
# The general lifecycle of an ant colony optimization algorithm is as follows:

# - Initialize the pheromone trails: This involves creating the concept of pheromone trails between attractions
# and initializing their intensity values.

# - Setup the population of ants: This involves creating a population of ants where each ant starts at a different
# attraction.

# - Choose the next visit for each ant: This involves choosing the next attraction to visit for each ant. This will
# happen until each ant has visited all attractions exactly once.

# - Update the pheromone trails: This involves updating the intensity of pheromone trails based on the ants’ movements
# on them as well as factoring in evaporation of pheromones.

# - Update the best solution: This involves updating the best solution given the total distance covered by each ant.

# - Determine stopping criteria: The process of ants visiting attractions repeats for a number of iterations. One
# iteration is every ant visiting all attractions exactly once. The stopping criteria determines the total number of
# iterations to run. More iterations will allow ants to make better decisions based on the pheromone trails.
class ACO:

    def __init__(self, number_of_ants_factor):
        self.number_of_ants_factor = number_of_ants_factor
        # Initialize the array for storing ants
        self.ant_colony = []
        # Initialize the 2D matrix for pheromone trails
        self.pheromone_trails = []
        # Initialize the best distance in swarm
        self.best_distance = math.inf
        self.best_ant = None

    # Initialize ants at random starting locations
    def setup_ants(self, number_of_ants_factor):
        number_of_ants = round(ATTRACTION_COUNT * number_of_ants_factor)
        self.ant_colony.clear()
        for i in range(0, number_of_ants):
            self.ant_colony.append(Ant())

    # Initialize pheromone trails between attractions
    def setup_pheromones(self):
        for r in range(0, len(attraction_distances)):
            pheromone_list = []
            for i in range(0, len(attraction_distances)):
                pheromone_list.append(1)
            self.pheromone_trails.append(pheromone_list)

    # Move all ants to a new attraction
    def move_ants(self, ant_population):
        for ant in ant_population:
            ant.visit_attraction(self.pheromone_trails)

    # Determine the best ant in the colony - after one tour of all attractions
    def get_best(self, ant_population):
        for ant in ant_population:
            distance_travelled = ant.get_distance_travelled()
            if distance_travelled < self.best_distance:
                self.best_distance = distance_travelled
                self.best_ant = ant
        return self.best_ant

    # Update pheromone trails based ant movements - after one tour of all attractions
    def update_pheromones(self, evaporation_rate):
        for x in range(0, ATTRACTION_COUNT):
            for y in range(0, ATTRACTION_COUNT):
                self.pheromone_trails[x][y] = self.pheromone_trails[x][y] * evaporation_rate
                for ant in self.ant_colony:
                    self.pheromone_trails[x][y] += 1 / ant.get_distance_travelled()

    # Tie everything together - this is the main loop
    def solve(self, total_iterations, evaporation_rate):
        self.setup_pheromones()
        for i in range(0, TOTAL_ITERATIONS):
            self.setup_ants(NUMBER_OF_ANTS_FACTOR)
            for r in range(0, ATTRACTION_COUNT - 1):
                self.move_ants(self.ant_colony)
            self.update_pheromones(evaporation_rate)
            self.best_ant = self.get_best(self.ant_colony)
            print(i, ' Best distance: ', self.best_ant.get_distance_travelled())


# Set the percentage of ants based on the total number of attractions
NUMBER_OF_ANTS_FACTOR = 0.5
# Set the number of tours ants must complete
TOTAL_ITERATIONS = 10000
# Set the rate of pheromone evaporation (0.0 - 1.0)
EVAPORATION_RATE = 0.4
aco = ACO(NUMBER_OF_ANTS_FACTOR)
aco.solve(TOTAL_ITERATIONS, EVAPORATION_RATE)
