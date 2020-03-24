from sklearn import tree
from sklearn import metrics
import sys
sys.path.append('../')
from ml_data_preparation import Data


# Encode the string values for "cut", "color", and "clarity" as integer values
encoding_categories = {'cut': {'Fair': 1, 'Good': 1, 'Very Good': 2, 'Premium': 2, 'Ideal': 2},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}

# "no","carat","cut","color","clarity","depth","table","price","x size","y size","z size"
data = Data('../diamonds.csv', ['no', 'color', 'clarity', 'depth', 'table', 'x size', 'y size', 'z size'], 'cut', encoding_categories)
print(data.X_train.head())
# clf = ensemble.RandomForestClassifier()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data.X_train, data.y_train)
prediction = clf.predict(data.X_test)
print("Prediction Accuracy: ", metrics.accuracy_score(prediction, data.y_test))
