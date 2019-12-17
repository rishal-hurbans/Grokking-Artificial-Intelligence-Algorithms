import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("diamonds.csv")
print(data.describe())
print(data.dtypes)

feature_x = 'carat'
feature_y = 'price'

fair_diamonds = data[data['cut'] == "Fair"]
good_diamonds = data[data['cut'] == "Good"]
very_good_diamonds = data[data['cut'] == "Very Good"]
premium_diamonds = data[data['cut'] == "Premium"]
ideal_diamonds = data[data['cut'] == "Ideal"]

fig = plt.figure()
plt.title(feature_x + ' vs ' + feature_y)

plt.scatter(fair_diamonds[feature_x], fair_diamonds[feature_y], label="Fair", s=1.8)
plt.scatter(good_diamonds[feature_x], good_diamonds[feature_y], label="Good", s=1.8)
plt.scatter(very_good_diamonds[feature_x], very_good_diamonds[feature_y], label="Very Good", s=1.8)
plt.scatter(premium_diamonds[feature_x], premium_diamonds[feature_y], label="Premium", s=1.8)
plt.scatter(ideal_diamonds[feature_x], ideal_diamonds[feature_y], label="Ideal", s=1.8)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.show()


encoding_categories = {'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}

data.replace(encoding_categories, inplace=True)

data_subset = data[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']]
cor = data_subset.corr()
sns.heatmap(cor, square=True)
plt.show()
