import itertools
import csv
import math
import time

start = time.time()
attraction_count = 48
attraction_data_file = "attractions-" + str(attraction_count) + ".csv"
attraction_permutations = set(itertools.permutations(range(0, attraction_count)))
attraction_distances = []
with open(attraction_data_file) as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        attraction_distances.append(row)
print('DONE LOADING ATTRACTION DISTANCES')
best_total = math.inf
best_permutation = None
for attraction_permutation in attraction_permutations:
    last_attraction = attraction_permutation[0]
    total_distance = 0
    for attraction_index in range(1, len(attraction_permutation)):
        total_distance += attraction_distances[last_attraction][attraction_permutation[attraction_index]]
        last_attraction = attraction_permutation[attraction_index]
    if total_distance < best_total:
        best_total = total_distance
        best_permutation = attraction_permutation
    print('BEST PERMUTATION: ', best_permutation)
    print('BEST DISTANCE: ', best_total)

print('BEST PERMUTATION: ', best_permutation)
print('BEST DISTANCE: ', best_total)

end = time.time()
print('TIME TO COMPUTE: ', end - start)
