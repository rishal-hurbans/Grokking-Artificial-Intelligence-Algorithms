import random

KNAPSACK_ITEM_NAME_INDEX = 0
KNAPSACK_ITEM_WEIGHT_INDEX = 1
KNAPSACK_ITEM_VALUE_INDEX = 2
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
INDIVIDUAL_CHROMOSOME_INDEX = 0
INDIVIDUAL_FITNESS_INDEX = 1
INDIVIDUAL_PROBABILITY_INDEX = 2


def generate_initial_population(population_size):
    population = []
    for individual in range(0, population_size):
        individual = ''.join([random.choice('01') for n in range(26)])
        population.append([individual, 0, 0])
    return population


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


def calculate_individual_fitness(individual, knapsack_max_weight):
    total_individual_weight = 0
    total_individual_value = 0
    for gene_index in range(len(individual)):
        bit = individual[gene_index]
        if bit == '1':
            total_individual_weight += knapsack_items[gene_index][KNAPSACK_ITEM_WEIGHT_INDEX]
            total_individual_value += knapsack_items[gene_index][KNAPSACK_ITEM_VALUE_INDEX]
    if total_individual_weight > knapsack_max_weight:
        return -1
    return total_individual_value


def set_probabilities(population):
    population_sum = sum(individual[INDIVIDUAL_FITNESS_INDEX] for individual in population)
    for individual in population:
        individual[INDIVIDUAL_PROBABILITY_INDEX] = individual[INDIVIDUAL_FITNESS_INDEX] / population_sum


def roulette_wheel_selection(population, number_of_selections):
    set_probabilities(population)
    chosen_ones = []
    for selection in range(number_of_selections):
        r = random.random()
        for individual in population:
            if r <= individual[INDIVIDUAL_PROBABILITY_INDEX]:
                chosen_ones.append(individual)
                break
    return chosen_ones


def one_point_crossover(parent_a, parent_b, xover_point):
    children = [parent_a[:xover_point] + parent_b[xover_point:],
                parent_b[:xover_point] + parent_a[xover_point:]]
    return mutate_children(children)


def two_point_crossover(parent_a, parent_b, xover_point_1, xover_point_2):
    children = [parent_a[:xover_point_1] + parent_b[xover_point_1:xover_point_2] + parent_a[xover_point_2:],
                parent_b[:xover_point_1] + parent_a[xover_point_1:xover_point_2] + parent_b[xover_point_2:]]
    return children


def mutate_children(children):
    for child in children:
        random_index = random.randint(0, 25)
        if child[INDIVIDUAL_CHROMOSOME_INDEX][random_index] == '1':
            mutated_child = list(child[INDIVIDUAL_CHROMOSOME_INDEX])
            mutated_child[random_index] = '0'
            child[INDIVIDUAL_CHROMOSOME_INDEX] = mutated_child
        else:
            mutated_child = list(child[INDIVIDUAL_CHROMOSOME_INDEX])
            mutated_child[random_index] = '1'
            child[INDIVIDUAL_CHROMOSOME_INDEX] = mutated_child
    return children


def reproduce_children(chosen_selections):
    children = []
    for parent_index in range(len(chosen_selections)//2 - 1):
        children = two_point_crossover(chosen_selections[parent_index],
                                       chosen_selections[parent_index + 1],
                                       CROSSOVER_POSITION_1,
                                       CROSSOVER_POSITION_2)
    return children


def merge_population_and_children(population, children):
    return population + children


NUMBER_OF_GENERATIONS = 1000
INITIAL_POPULATION_SIZE = 1000
KNAPSACK_WEIGHT_CAPACITY = 6404180
CROSSOVER_POSITION_1 = 5
CROSSOVER_POSITION_2 = 22


def run_ga():
    best_global_fitness = 0
    global_population = generate_initial_population(INITIAL_POPULATION_SIZE)
    for generation in range(NUMBER_OF_GENERATIONS):
        current_best_fitness = calculate_population_fitness(global_population, KNAPSACK_WEIGHT_CAPACITY)
        if current_best_fitness > best_global_fitness:
            best_global_fitness = current_best_fitness
        the_chosen = roulette_wheel_selection(global_population, 100)
        the_children = reproduce_children(the_chosen)
        the_children = mutate_children(the_children)
        global_population = merge_population_and_children(global_population, the_children)

    print('BEST FITNESS: \t', best_global_fitness)
    print('ACTUAL BEST: \t', 13692887)
    print('ACCURACY: \t\t', best_global_fitness / 13692887 * 100)
    print('FINAL POP: \t\t', len(global_population))

    # calculate_population_fitness(global_population, KNAPSACK_WEIGHT_CAPACITY)
    # the_chosen = roulette_wheel_selection(global_population, 100)
    # the_children = reproduce_children(the_chosen)
    # the_children = mutate_children(the_children)
    # global_population = merge_population_and_children(global_population, the_children)
    # global_population = roulette_wheel_selection(global_population, 100)


run_ga()
