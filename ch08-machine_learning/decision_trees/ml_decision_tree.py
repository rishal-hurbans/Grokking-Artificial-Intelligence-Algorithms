
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


class Question:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def filter(self, example):
        value = example[self.feature]
        return value >= self.value

    def to_string(self):
        return 'Is ' + feature_names[self.feature] + ' >= ' + str(self.value) + '?'


class ExamplesNode:
    def __init__(self, examples):
        self.examples = find_unique_label_counts(examples)


class DecisionNode:
    def __init__(self, question, branch_true, branch_false):
        self.question = question
        self.branch_true = branch_true
        self.branch_false = branch_false


def find_unique_label_counts(examples):
    class_count = {}
    for example in examples:
        label = example[-1]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count


def split_examples(examples, question):
    examples_true = []
    examples_false = []
    for example in examples:
        if question.filter(example):
            examples_true.append(example)
        else:
            examples_false.append(example)
    return examples_true, examples_false


def calculate_gini(examples):
    label_counts = find_unique_label_counts(examples)
    uncertainty = 1
    for label in label_counts:
        probability_of_label = label_counts[label] / float(len(examples))
        uncertainty -= probability_of_label ** 2
    return uncertainty


def calculate_information_gain(left, right, current_uncertainty):
    total = len(left) + len(right)
    gini_left = calculate_gini(left)
    entropy_left = len(left) / total * gini_left
    gini_right = calculate_gini(right)
    entropy_right = len(right) / total * gini_right
    uncertainty_after = entropy_left + entropy_right
    information_gain = current_uncertainty - uncertainty_after
    return information_gain


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


def build_tree(examples):
    gain, question = find_best_split(examples, len(examples[0]) - 1)
    if gain == 0:
        return ExamplesNode(examples)
    print('BEST QUESTION :', question.to_string(), '\t', 'INFO GAIN: ', "{0:.3f}".format(gain))
    examples_true, examples_false = split_examples(examples, question)
    branch_true = build_tree(examples_true)
    branch_false = build_tree(examples_false)
    return DecisionNode(question, branch_true, branch_false)


def print_tree(node, indentation=''):
    # The examples in the current ExampleNode
    if isinstance(node, ExamplesNode):
        print(indentation + 'Examples', node.examples)
        return
    # The question for the current DecisionNode
    print(indentation + str(node.question.to_string()))
    # Find the 'True' examples for the current Decision node recursively
    print(indentation + '---> True:')
    print_tree(node.branch_true, indentation + '\t')
    # Find the 'False' examples for the current Decision node recursively
    print(indentation + '---> False:')
    print_tree(node.branch_false, indentation + '\t')


tree = build_tree(feature_examples)
print_tree(tree)
