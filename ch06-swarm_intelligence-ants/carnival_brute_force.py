import itertools
import csv
import math
import time

start = time.time()

# Load the carnival attraction distances from a CSV file
attraction_count = 48
attraction_data_file = 'attractions-' + str(attraction_count) + '.csv'
attraction_distances = []
with open(attraction_data_file) as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        attraction_distances.append(row)
print('Done loading attraction distances')

# Initialize the best distance and best permutation
best_distance = math.inf
best_permutation = None
# Determine the distance score for every permutation to find the best
for attraction_permutation in itertools.permutations(range(0, attraction_count)):
    last_attraction = attraction_permutation[0]
    total_distance = 0
    for attraction_index in range(1, len(attraction_permutation)):
        total_distance += attraction_distances[last_attraction][attraction_permutation[attraction_index]]
        last_attraction = attraction_permutation[attraction_index]
    if total_distance < best_distance:
        best_distance = total_distance
        best_permutation = attraction_permutation
    print('Current best permutation: ', best_permutation)
    print('Current best distance: ', best_distance)

print('Best permutation: ', best_permutation)
print('Best distance: ', best_distance)

end = time.time()
print('Time to compute: ', end - start)
