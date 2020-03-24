import collections
from sklearn import tree
from sklearn import metrics
import sys
sys.path.append('../')
from ml_data_preparation import Data
import pydotplus


# Encode the string values for "cut", "color", and "clarity" as integer values
encoding_categories = {'cut': {'Fair': 1, 'Good': 1, 'Very Good': 2, 'Premium': 2, 'Ideal': 2},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}

# "no","carat","cut","color","clarity","depth","table","price","x size","y size","z size"
data = Data('../diamonds.csv', ['no', 'color', 'clarity', 'depth', 'table', 'x size', 'y size', 'z size'], 'cut', encoding_categories)
print(data.X_train.head())

data_X = [[0.21, 327],   # 1
          [0.39, 497],   # 1
          [0.50, 1122],  # 2
          [0.76, 907],   # 1
          [0.87, 2757],  # 1
          [0.98, 2865],  # 1
          [1.13, 3045],  # 2
          [1.34, 3914],  # 2
          [1.67, 4849],  # 2
          [1.81, 5688]]  # 2

data_Y = ['1', '1', '2', '1', '1', '1', '2', '2', '2', '2']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_X, data_Y)

dot_data = tree.export_graphviz(clf,
                                feature_names=['carat', ['price']],
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('cyan', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
