import numpy as np


# Artificial Neural Network
# Let’s walk through the phases and different operations involved in the back propagation algorithm.

# Phase A: Setup

# 1.	Define ANN architecture: This involves defining the input nodes, the output nodes, the number of hidden layers,
# the number of neurons in each hidden layer, the activation functions used, and more. We will dive into some of these
# details in the next section. For now, we will stick to the same ANN architecture that we have already used in the
# previous section.

# 2.	Initialize ANN weights: The weights in the ANN must be initialized to some value. There are various approaches
# to this, however, the key principle is that the weights will be constantly adjusted as the ANN learns from training
# examples but we need to start somewhere.

# Phase B: Forward propagation: This is the same process that we covered in the previous section. The same calculations
# are carried out; however, the predicted output will be compared with the actual class for each example in the training
# set to train the network.

# Phase C: Training
# 1.	Calculate cost: Following from forward propagation, the cost is the difference between the predicted output and
# the actual class for the examples in the training set. The cost is effectively determining how bad the ANN is at
# predicting the class of examples.

# 2.	Update weights in the ANN: The weights of the ANN are the only thing that can be adjusted by the network itself.
# The architecture and configurations that we defined in phase A doesn’t change during training the network. The weights
# are essentially encoding the “intelligence” of the network. Weights are adjusted to be larger or smaller which impacts
# the strength of the inputs.

# 3.	Stopping condition: Training cannot happen indefinitely. Similarly to many of the algorithms explored in this
# book, a sensible stopping condition needs to be determined. If we have a large dataset, we might decide that we will
# use 500 examples in our training dataset over 1000 iterations to train the ANN. This means that the 500 examples will
# be passed through the network 1000 times and adjust the weights in every iteration.

# Scale a dataset to values between 0 and 1 using min-max scaling
def scale_dataset(dataset, feature_count, feature_min, feature_max):
    scaled_data = []
    for data in dataset:
        example = []
        for i in range(0, feature_count):
            example.append(scale_data_feature(data[i], feature_min[i], feature_max[i]))
        scaled_data.append(example)
    return np.array(scaled_data)


# Scale features of a dataset to values between 0 and 1 using min-max scaling
def scale_data_feature(data, feature_min, feature_max):
    return (data - feature_min) / (feature_max - feature_min)


# The sigmoid function will be used as the activation function
# np.exp is a mathematical constant called Euler's number, approximately 2.71828
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# A class to encapsulate the functionality of the artificial neural network
class NeuralNetwork:
    def __init__(self, features, labels, hidden_node_count):
        # Map the features as inputs to the neural network
        self.input = features
        # Initialize weights between the input layer and hidden layer to random values
        self.weights_input = np.random.rand(self.input.shape[1], hidden_node_count)
        print(self.weights_input)
        # Initialize hidden node outputs to nothing
        self.hidden = None
        # Initialize weights between the hidden layer and output node
        self.weights_hidden = np.random.rand(hidden_node_count, 1)
        print(self.weights_hidden)
        # Map the actual expected outputs to the labels
        self.expected_output = labels
        # Initialize the output results to zeros
        self.output = np.zeros(self.expected_output.shape)

    def add_example(self, features, label):
        np.append(self.input, features)
        np.append(self.expected_output, label)

    # Process forward propagation by calculating the weighted sum and activation values
    def forward_propagation(self):
        hidden_weighted_sum = np.dot(self.input, self.weights_input)
        self.hidden = sigmoid(hidden_weighted_sum)
        output_weighted_sum = np.dot(self.hidden, self.weights_hidden)
        self.output = sigmoid(output_weighted_sum)

    # Process back propagation by calculating the cost and updating the weights
    def back_propagation(self):
        cost = self.expected_output - self.output
        print('ACTUAL: ')
        print(self.expected_output)
        print('PREDICTED: ')
        print(self.output)
        print('COSTS: ')
        print(cost)
        print('HIDDEN: ')
        print(self.hidden)
        weights_hidden_update = np.dot(self.hidden.T, (2 * cost * sigmoid_derivative(self.output)))
        print('WEIGHTS HIDDEN UPDATE:')
        print(weights_hidden_update)
        weights_input_update = np.dot(self.input.T, (np.dot(2 * cost * sigmoid_derivative(self.output), self.weights_hidden.T) * sigmoid_derivative(self.hidden)))
        print('WEIGHTS INPUT UPDATE:')
        print(weights_hidden_update)

        # update the weights with the derivative (slope) of the loss function
        self.weights_hidden += weights_hidden_update
        print('WEIGHTS HIDDEN:')
        print(weights_hidden_update)

        self.weights_input += weights_input_update
        print('WEIGHTS INPUT:')
        print(weights_hidden_update)


def run_neural_network(feature_data, label_data, feature_count, features_min, features_max, hidden_node_count, epochs):
    # Apply min-max scaling to the dataset
    scaled_feature_data = scale_dataset(feature_data, feature_count, features_min, features_max)
    # Initialize a neural network with the scaled data and hidden nodes
    nn = NeuralNetwork(scaled_feature_data, label_data, hidden_node_count)
    # Train the artificial neural network over many iterations with the same training data
    for epoch in range(epochs):
        nn.forward_propagation()
        nn.back_propagation()

    print('OUTPUTS: ')
    for r in nn.output:
        print(r)

    print('INPUT WEIGHTS: ')
    print(nn.weights_input)

    print('HIDDEN WEIGHTS: ')
    print(nn.weights_hidden)


if __name__ == '__main__':
    # Number of features in the dataset
    FEATURE_COUNT = 4
    # Minimum possible values for features (Speed, Terrain Quality, Degree of Vision, Driving Experience)
    FEATURE_MIN = [0, 0, 0, 0]
    # Maximum possible values for features (Speed, Terrain Quality, Degree of Vision, Driving Experience)
    FEATURE_MAX = [120, 10, 360, 400000]
    # Number of hidden nodes
    HIDDEN_NODE_COUNT = 5
    # Number of iterations to train the neural network
    EPOCHS = 1500

    # Training dataset (Speed, Terrain Quality, Degree of Vision, Driving Experience)
    car_collision_data = np.array([
        [65, 5,	180, 80000],
        [120, 1, 72, 110000],
        [8,	6,	288, 50000],
        [50, 2,	324, 1600],
        [25, 9,	36, 160000],
        [80, 3,	120, 6000],
        [40, 3,	360, 400000]
    ])

    # Labels for training dataset (0 = No collision occurred, 1 = Collision occurred)
    car_collision_data_labels = np.array([
        [0],
        [1],
        [0],
        [1],
        [0],
        [1],
        [0]])

    # Run neural network
    run_neural_network(car_collision_data,
                       car_collision_data_labels,
                       FEATURE_COUNT,
                       FEATURE_MIN,
                       FEATURE_MAX,
                       HIDDEN_NODE_COUNT,
                       EPOCHS)
