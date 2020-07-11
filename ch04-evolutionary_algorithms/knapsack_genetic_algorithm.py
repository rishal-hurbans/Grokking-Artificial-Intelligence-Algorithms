import random

# The indexes for data array
# 0 = name, 1 = weight, 2 = value, 3 = fitness
KNAPSACK_ITEM_NAME_INDEX = 0
KNAPSACK_ITEM_WEIGHT_INDEX = 1
KNAPSACK_ITEM_VALUE_INDEX = 2

# Small knapsack dataset
# knapsack_items = [
#     ['Pearls', 3, 4],
#     ['Gold', 7, 7],
#     ['Crown', 4, 5],
#     ['Coin', 1, 1],
#     ['Axe', 5, 4],
#     ['Sword', 4, 3],
#     ['Ring', 2, 5],
#     ['Cup', 3, 1],
# ]

# Large knapsack dataset
knapsack_items = [
    ['Axe', 32252, 68674],
    ['Bronze coin', 225790, 471010],
    ['Crown', 468164, 944620],
    ['Diamond statue', 489494, 962094],
    ['Emerald belt', 35384, 78344],
    ['Fossil', 265590, 579152],
    ['Gold coin', 497911, 902698],
    ['Helmet', 800493, 1686515],
    ['Ink', 823576, 1688691],
    ['Jewel box', 552202, 1056157],
    ['Knife', 323618, 677562],
    ['Long sword', 382846, 833132],
    ['Mask', 44676, 99192],
    ['Necklace', 169738, 376418],
    ['Opal badge', 610876, 1253986],
    ['Pearls', 854190, 1853562],
    ['Quiver', 671123, 1320297],
    ['Ruby ring', 698180, 1301637],
    ['Silver bracelet', 446517, 859835],
    ['Timepiece', 909620, 1677534],
    ['Uniform', 904818, 1910501],
    ['Venom potion', 730061, 1528646],
    ['Wool scarf', 931932, 1827477],
    ['Cross bow', 952360, 2068204],
    ['Yesteryear book', 926023, 1746556],
    ['Zinc cup', 978724, 2100851, 0]
]

# The best knapsack score from the brute force approach
BEST_LARGE_KNAPSACK_SCORE = 13692887

# Genetic algorithms are used to evaluate large search spaces for a good solution. It is important to note that a
# genetic algorithm is not guaranteed to find the absolute best solution. It attempts to find the global best whilst
# avoiding local best solutions. The general lifecycle of a genetic algorithm is as follows:

# - Creation of a population: This involves creating a random population of potential solutions.

# - Measuring fitness of individuals in the population: This involves determining how good a specific solution is.
# This is accomplished by using a fitness function which scores solutions to determine how good they are.

# - Selecting parents based on their fitness: This involves selecting a number of pairs of parents that will reproduce
# offspring.

# - Reproducing individuals from parents: This involves creating offspring from their respective parents by mixing
# genetic information and applying slight mutations to the offspring.

# - Populating the next generation: This involves selecting individuals and offspring from the population that will
# survive to the next generation.

# The indexes for an individual's properties
INDIVIDUAL_CHROMOSOME_INDEX = 0
INDIVIDUAL_FITNESS_INDEX = 1
INDIVIDUAL_PROBABILITY_INDEX = 2


# Generate an initial population of random individuals
def generate_initial_population(population_size):
    population = []
    for individual in range(0, population_size):
        individual = ''.join([random.choice('01') for n in range(26)])
        population.append([individual, 0, 0])
    return population


# Calculate the fitness for each individual in the population given the maximum weight
def calculate_population_fitness(population, maximum_weight):
    best_fitness = 0
    for individual in population:
        individual_fitness = calculate_individual_fitness(individual[INDIVIDUAL_CHROMOSOME_INDEX], maximum_weight)
        individual[INDIVIDUAL_FITNESS_INDEX] = individual_fitness
        if individual_fitness > best_fitness:
            best_fitness = individual_fitness
        if individual_fitness == -1:
            population.remove(individual)
    return best_fitness


# Calculate the fitness for an individual
def calculate_individual_fitness(individual, maximum_weight):
    total_individual_weight = 0
    total_individual_value = 0
    for gene_index in range(len(individual)):
        gene_switch = individual[gene_index]
        if gene_switch == '1':
            total_individual_weight += knapsack_items[gene_index][KNAPSACK_ITEM_WEIGHT_INDEX]
            total_individual_value += knapsack_items[gene_index][KNAPSACK_ITEM_VALUE_INDEX]
    if total_individual_weight > maximum_weight:
        return -1
    return total_individual_value


# Set the probabilities for selection for each individual in the population
def set_probabilities(population):
    population_sum = sum(individual[INDIVIDUAL_FITNESS_INDEX] for individual in population)
    for individual in population:
        individual[INDIVIDUAL_PROBABILITY_INDEX] = individual[INDIVIDUAL_FITNESS_INDEX] / population_sum


# Roulette wheel selection to select individuals in a population
def roulette_wheel_selection(population, number_of_selections):
    set_probabilities(population)
    slices = []
    total = 0
    for r in range(0, len(population)):
        individual = population[r]
        slices.append([r, total, total + individual[INDIVIDUAL_PROBABILITY_INDEX]])
        total += individual[INDIVIDUAL_PROBABILITY_INDEX]
    chosen_ones = []
    for r in range(number_of_selections):
        spin = random.random()
        result = [s[0] for s in slices if s[1] < spin <= s[2]]
        chosen_ones.append(population[result[0]])
    return chosen_ones


# Reproduce children given two individuals using one point crossover
def one_point_crossover(parent_a, parent_b, xover_point):
    children = [parent_a[:xover_point] + parent_b[xover_point:],
                parent_b[:xover_point] + parent_a[xover_point:]]
    return children


# Reproduce children given two individuals using two point crossover
def two_point_crossover(parent_a, parent_b, xover_point_1, xover_point_2):
    children = [parent_a[:xover_point_1] + parent_b[xover_point_1:xover_point_2] + parent_a[xover_point_2:],
                parent_b[:xover_point_1] + parent_a[xover_point_1:xover_point_2] + parent_b[xover_point_2:]]
    return children


# Randomly mutate children
def mutate_children(children, mutation_rate):
    for child in children:
        random_index = random.randint(0, mutation_rate)
        if child[INDIVIDUAL_CHROMOSOME_INDEX][random_index] == '1':
            mutated_child = list(child[INDIVIDUAL_CHROMOSOME_INDEX])
            mutated_child[random_index] = '0'
            child[INDIVIDUAL_CHROMOSOME_INDEX] = mutated_child
        else:
            mutated_child = list(child[INDIVIDUAL_CHROMOSOME_INDEX])
            mutated_child[random_index] = '1'
            child[INDIVIDUAL_CHROMOSOME_INDEX] = mutated_child
    return children


# Reproduce children given selected individuals
def reproduce_children(chosen_selections):
    children = []
    for parent_index in range(len(chosen_selections)//2 - 1):
        children = one_point_crossover(chosen_selections[parent_index],
                                       chosen_selections[parent_index + 1],
                                       CROSSOVER_POSITION_1)
    return children


# Combine the existing population and newly reproduced children
def merge_population_and_children(population, children):
    return population + children


# Set the hyper parameters for the genetic algorithm
NUMBER_OF_GENERATIONS = 1000
INITIAL_POPULATION_SIZE = 1000
KNAPSACK_WEIGHT_CAPACITY = 6404180
CROSSOVER_POSITION_1 = 13
CROSSOVER_POSITION_2 = 22
MUTATION_RATE = 10
NUMBER_OF_ITERATIONS = 5


# Run the genetic algorithm
def run_ga():
    best_global_fitness = 0
    global_population = generate_initial_population(INITIAL_POPULATION_SIZE)
    for generation in range(NUMBER_OF_GENERATIONS):
        current_best_fitness = calculate_population_fitness(global_population, KNAPSACK_WEIGHT_CAPACITY)
        if current_best_fitness > best_global_fitness:
            best_global_fitness = current_best_fitness
        the_chosen = roulette_wheel_selection(global_population, 100)
        the_children = reproduce_children(the_chosen)
        the_children = mutate_children(the_children, MUTATION_RATE)
        global_population = merge_population_and_children(global_population, the_children)
        # print(global_population)

    print('Best fitness: ', best_global_fitness)
    print('Actual best: ', BEST_LARGE_KNAPSACK_SCORE)
    print('Accuracy: ', best_global_fitness / BEST_LARGE_KNAPSACK_SCORE * 100)
    print('Final population size: ', len(global_population))

    # calculate_population_fitness(global_population, KNAPSACK_WEIGHT_CAPACITY)
    # the_chosen = roulette_wheel_selection(global_population, 100)
    # the_children = reproduce_children(the_chosen)
    # the_children = mutate_children(the_children)
    # global_population = merge_population_and_children(global_population, the_children)
    # global_population = roulette_wheel_selection(global_population, 100)


# Run the genetic algorithm for a number of iterations
for i in range(0, NUMBER_OF_ITERATIONS):
    run_ga()

# print(calculate_individual_fitness('01100100010110001110001001', 6404180))
# print(calculate_individual_fitness('00110101000100011010001000', 6404180))
# print(calculate_individual_fitness('11100100110110000100101101', 6404180))
# print(calculate_individual_fitness('00001000010010101101001001', 6404180))
