# Decision Trees
# Decision trees are structures that describe a series of decisions that are made to find a solution to a problem.
# If we’re deciding whether or not to wear shorts for the day, we might make a series of decisions to inform the
# outcome. Will it be cold during the day? If not, will we be out late in the evening when it does get cold?
# We might decide to wear shorts on a warm day, but not if we will be out when it gets cold.
# In building a decision tree, all possible questions will be tested to determine which one is the best question to
# ask at a specific point in the decision tree. To test a question, the concept of entropy is used. Entropy is the
# uncertainty of the dataset.

# The data used for learning
feature_names = ['carat', 'price', 'cut']
feature_examples = [[0.21, 327, 'Average'],
                    [0.39, 897, 'Perfect'],
                    [0.50, 1122, 'Perfect'],
                    [0.76, 907, 'Average'],
                    [0.87, 2757, 'Average'],
                    [0.98, 2865, 'Average'],
                    [1.13, 3045, 'Perfect'],
                    [1.34, 3914, 'Perfect'],
                    [1.67, 4849, 'Perfect'],
                    [1.81, 5688, 'Perfect']]


# The Question class defines a feature and value that it should satisfy
class Question:

    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def filter(self, example):
        value = example[self.feature]
        return value >= self.value

    def to_string(self):
        return 'Is ' + feature_names[self.feature] + ' >= ' + str(self.value) + '?'


# The ExamplesNode class defines a node in the tree that contains classified examples
class ExamplesNode:
    def __init__(self, examples):
        self.examples = find_unique_label_counts(examples)


# The DecisionNode class defines a node in the tree that contains a question, and two branches
class DecisionNode:
    def __init__(self, question, branch_true, branch_false):
        self.question = question
        self.branch_true = branch_true
        self.branch_false = branch_false


# Count the unique classes and their counts from a list of examples
def find_unique_label_counts(examples):
    class_count = {}
    for example in examples:
        label = example[-1]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count


# Split a list of examples based on a question being asked
def split_examples(examples, question):
    examples_true = []
    examples_false = []
    for example in examples:
        if question.filter(example):
            examples_true.append(example)
        else:
            examples_false.append(example)
    return examples_true, examples_false


# Calculate the Gini Index based on a list of examples
def calculate_gini(examples):
    label_counts = find_unique_label_counts(examples)
    uncertainty = 1
    for label in label_counts:
        probability_of_label = label_counts[label] / float(len(examples))
        uncertainty -= probability_of_label ** 2
    return uncertainty


# Calculate the information gain based on the left gini, right gini, and current uncertainty
def calculate_information_gain(left_gini, right_gini, current_uncertainty):
    total = len(left_gini) + len(right_gini)
    gini_left = calculate_gini(left_gini)
    entropy_left = len(left_gini) / total * gini_left
    gini_right = calculate_gini(right_gini)
    entropy_right = len(right_gini) / total * gini_right
    uncertainty_after = entropy_left + entropy_right
    information_gain = current_uncertainty - uncertainty_after
    return information_gain


# Fine the best split for a list of examples based on its features
def find_best_split(examples, number_of_features):
    best_gain = 0
    best_question = None
    current_uncertainty = calculate_gini(examples)
    for feature_index in range(number_of_features):
        values = set([example[feature_index] for example in examples])
        for value in values:
            question = Question(feature_index, value)
            examples_true, examples_false = split_examples(examples, question)
            if len(examples_true) != 0 or len(examples_false) != 0:
                gain = calculate_information_gain(examples_true, examples_false, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question
    return best_gain, best_question


# Build the decision tree
def build_tree(examples):
    gain, question = find_best_split(examples, len(examples[0]) - 1)
    if gain == 0:
        return ExamplesNode(examples)
    print('Best question : ', question.to_string(), '\t', 'Info gain: ', "{0:.3f}".format(gain))
    examples_true, examples_false = split_examples(examples, question)
    branch_true = build_tree(examples_true)
    branch_false = build_tree(examples_false)
    return DecisionNode(question, branch_true, branch_false)


def print_tree(node, indentation=''):
    # The examples in the current ExamplesNode
    if isinstance(node, ExamplesNode):
        print(indentation + 'Examples', node.examples)
        return
    # The question for the current DecisionNode
    print(indentation + str(node.question.to_string()))
    # Find the 'True' examples for the current DecisionNode recursively
    print(indentation + '---> True:')
    print_tree(node.branch_true, indentation + '\t')
    # Find the 'False' examples for the current DecisionNode recursively
    print(indentation + '---> False:')
    print_tree(node.branch_false, indentation + '\t')


tree = build_tree(feature_examples)
print_tree(tree)
