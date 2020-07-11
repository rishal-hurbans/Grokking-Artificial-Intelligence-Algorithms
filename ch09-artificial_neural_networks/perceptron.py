import numpy as np

# The Perceptron
# The neuron is the fundamental concept that makes up the brain and nervous systems.
# It accepts many inputs from other neurons, processes those inputs, and transfers the result to other “connected”
# neurons. Artificial neural networks are based on the fundamental concept of the Perceptron. The Perceptron is a
# logical representation of a single biological neuron.

# Features
# Smoking, Obesity, Exercise
dataset = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 1, 1]])
dataset_labels = np.array([[1, 0, 0, 1, 1, 0, 0, 1]])
dataset_labels = dataset_labels.reshape(8, 1)

np.random.seed(42)
weights = np.random.rand(3, 1)
bias = 1  # np.random.rand(1)
learning_rate = 0.05


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


for epoch in range(10000):
    # Multiply every input with it's respective weight and sum the outputs
    weight_and_sum_results = np.dot(dataset, weights) + bias
    # Apply the sigmoid activation function to all the input sums
    activation_results = sigmoid(weight_and_sum_results)
    # Determine error for each data row
    error = activation_results - dataset_labels
    # Find slope of the predicated results using derivatives
    predicted_results_derivative = sigmoid_derivative(activation_results)
    # Find amount to adjust weights by
    z_delta = error * predicted_results_derivative
    # Transpose array to work with consistent shaped matrices
    inputs = dataset.transpose()
    # Update weights using gradient descent
    weights -= learning_rate * np.dot(inputs, z_delta)
    # Update bias
    for num in z_delta:
        bias -= learning_rate * num


# Smoker, obese, no exercise
single_point = np.array([1, 0, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Smoker, not obese, no exercise')
print(result)

# Non smoker, obese, no exercise
single_point = np.array([0, 1, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Non smoker, obese, no exercise')
print(result)

# Non smoker, not obese, exercise
single_point = np.array([0, 0, 1])
result = sigmoid(np.dot(single_point, weights) + bias)
print('Non smoker, not obese, does exercise')
print(result)
