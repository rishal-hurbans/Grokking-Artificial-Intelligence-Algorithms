import numpy as np
import matplotlib.pyplot as plt
import statistics

# Carat values for each diamond
carats = [0.3,
          0.41,
          0.75,
          0.91,
          1.2,
          1.31,
          1.5,
          1.74,
          1.96,
          2.21]

# Scale the carat values for each diamond to be similarly sized to the price
carats = [i * 1000 for i in carats]

# Price values for each diamond
price = [339,
         561,
         2760,
         2763,
         2809,
         3697,
         4022,
         4677,
         6147,
         6535]

# Calculate the mean for 'price' and 'carat'
mean_X = statistics.mean(carats)
print(mean_X)
mean_Y = statistics.mean(price)
print(mean_Y)

# Calculate the number of examples in the dataset
number_of_examples = len(carats)

# Print values for x (Carats)
print('x')
for i in range(number_of_examples):
    print('{0:.0f}'.format(carats[i]))

# Print values for x - mean of x
print('x - x mean')
for i in range(number_of_examples):
    print('{0:.0f}'.format(carats[i] - mean_X))

# Print values for y - mean of y
print('y - y mean')
for i in range(number_of_examples):
    print('{0:.0f}'.format(price[i] - mean_Y))

# Print values for x - (x mean)^2
print('x - (x mean)^2')
sum_x_squared = 0
for i in range(number_of_examples):
    ans = (carats[i] - mean_X) ** 2
    sum_x_squared += ans
    print('{0:.0f}'.format(ans))
print('SUM squared: ', sum_x_squared)

# Print values for x - x mean * y - y mean
print('x - x mean) * y - y mean')
sum_multiple = 0
for i in range(number_of_examples):
    ans = (carats[i] - mean_X) * (price[i] - mean_Y)
    sum_multiple += ans
    print('{0:.0f}'.format(ans))
print('SUM multi: ', sum_multiple)

b1 = sum_multiple / sum_x_squared
print('b1: ', b1)
b0 = mean_Y - (b1 * mean_X)
print('b0: ', b0)
min_x = np.min(carats)
max_x = np.max(carats)
x = np.linspace(min_x, max_x, 10)

# Express the regression line by y = mx + c
y = b0 + b1 * x

# Testing data
carats_test = [
    220,
    330,
    710,
    810,
    1080,
    1390,
    1500,
    1640,
    1850,
    1910
]

price_test = [
    342,
    403,
    2772,
    2789,
    2869,
    3914,
    4022,
    4849,
    5688,
    6632
]

price_test_mean = statistics.mean(price_test)
print('price test mean: ', price_test_mean)
price_test_n = len(price_test)
print('price test difference:')
for i in range(price_test_n):
    print(price_test[i] - price_test_mean)

print('price test difference squared:')
sum_of_price_test_difference = 0
for i in range(price_test_n):
    ans = (price_test[i] - price_test_mean) ** 2
    sum_of_price_test_difference += ans
    print(ans)
print('sum diff: ', sum_of_price_test_difference)

print('predicted values:')
for i in range(price_test_n):
    print('{0:.0f}'.format(b0 + carats_test[i] * b1))
print('predicted values - mean:')
for i in range(price_test_n):
    print('{0:.0f}'.format((b0 + carats_test[i] * b1) - price_test_mean))

print('predicted values - mean squared:')
sum_of_price_test_prediction_difference = 0
for i in range(price_test_n):
    ans = ((b0 + carats_test[i] * b1) - price_test_mean) ** 2
    sum_of_price_test_prediction_difference += ans
    print('{0:.0f}'.format(ans))
print('sum prediction: ', sum_of_price_test_prediction_difference)

# Calculate the R^2 score
ss_numerator = 0
ss_denominator = 0
for i in range(number_of_examples):
    y_predicted = b0 + b1 * carats_test[i]
    ss_numerator += ((price_test[i] - mean_Y) - y_predicted) ** 2
    ss_denominator += (price_test[i] - mean_Y) ** 2
r2 = ss_numerator / ss_denominator
print('R2: ', r2)

# Plot the data on a figure to better understand visually
fig = plt.figure()
plt.figure(num=None, figsize=(5, 5), dpi=300, facecolor='w', edgecolor='w')

# Plot the original training data in red
plt.scatter(carats, price, color='red', label='Scatter Plot')
# Plot the testing data in black
plt.scatter(carats_test, price_test, color='black', label='Scatter Plot')
# Plot lines to represent the mean for x and y in gray
plt.axvline(x=mean_X, color='gray')
plt.axhline(y=mean_Y, color='gray')
# Plot the regression line using the min and max for carats
rex_x = [300, 2210]
rex_y = [515.7, 6511.19]
plt.plot(rex_x, rex_y, color='green')
# Label the figure, save it, and show it
plt.xlabel('Carat')
plt.ylabel('Price')
plt.savefig('carat_vs_price_test_scatter.png')
plt.show()
