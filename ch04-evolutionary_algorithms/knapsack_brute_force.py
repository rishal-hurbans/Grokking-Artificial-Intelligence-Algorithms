import itertools
from itertools import product
import time

KNAPSACK_WEIGHT_INDEX = 1
KNAPSACK_VALUE_INDEX = 2
# 0 = name, 1 = weight, 2 = value, 3 = fitness

knapsack_items = [
    ['Pearls', 3, 4],
    ['Gold', 7, 7],
    ['Crown', 4, 5],
    ['Coin', 1, 1],
    ['Axe', 5, 4],
    ['Sword', 4, 3],
    ['Ring', 2, 5],
    ['Cup', 3, 1],
]

# knapsack_items = [
#     ['Axe', 32252, 68674],
#     ['Bronze coin', 225790, 471010],
#     ['Crown', 468164, 944620],
#     ['Diamond statue', 489494, 962094],
#     ['Emerald belt', 35384, 78344],
#     ['Fossil', 265590, 579152],
#     ['Gold coin', 497911, 902698],
#     ['Helmet', 800493, 1686515],
#     ['Ink', 823576, 1688691],
#     ['Jewel box', 552202, 1056157],
#     ['Knife', 323618, 677562],
#     ['Long sword', 382846, 833132],
#     ['Mask', 44676, 99192],
#     ['Necklace', 169738, 376418],
#     ['Opal badge', 610876, 1253986],
#     ['Pearls', 854190, 1853562],
#     ['Quiver', 671123, 1320297],
#     ['Ruby ring', 698180, 1301637],
#     ['Silver bracelet', 446517, 859835],
#     ['Timepiece', 909620, 1677534],
#     ['Uniform', 904818, 1910501],
#     ['Venom potion', 730061, 1528646],
#     ['Wool scarf', 931932, 1827477],
#     ['Cross bow', 952360, 2068204],
#     ['Yesteryear book', 926023, 1746556],
#     ['Zinc cup', 978724, 2100851, 0]
# ]


def get_all_combinations(items):
    combinations = []
    for index in range(0, len(items)):
        combinations.append(items[index])
        els = [list(x) for x in itertools.combinations(items, index)]
        combinations.append(els)
    return combinations


def calculate_individual_fitness(individual, maximum_weight):
    total_weight = 0
    total_value = 0
    for gene in range(0, len(individual)):
        bit = individual[gene]
        if bit == 1:
            total_weight += knapsack_items[gene][KNAPSACK_WEIGHT_INDEX]
            total_value += knapsack_items[gene][KNAPSACK_VALUE_INDEX]
    if total_weight > maximum_weight:
        return 0
    return total_value


def run_brute_force():
    bit_string_size = 8
    best_score = 0
    best_individual = []
    knapsack_max_capacity = 10
    print('NUMBER OF COMBINATIONS: ', 2**bit_string_size)
    iteration = 0
    for i in product([0, 1], repeat=bit_string_size):
        current = calculate_individual_fitness(i, knapsack_max_capacity)
        if current > best_score:
            best_score = current
            best_individual = i
            print('ITERATION: ', iteration)
            print('BEST SCORE: ', best_score)
            print('BEST INDIVIDUAL: ', best_individual)
        iteration += 1

    print(best_individual)


start_time = time.time()
run_brute_force()
end_time = time.time()
total_time = end_time - start_time
print("TOTAL TIME:", total_time)
