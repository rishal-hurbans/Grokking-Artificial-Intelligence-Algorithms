import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from ml_data_preparation import Data


encoding_categories = {'cut': {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5},
                       'color': {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
                       'clarity': {'FL': 11, 'IF': 10, 'VVS1': 9, 'VVS2': 8, 'VS1': 7, 'VS2': 6, 'SI1': 5, 'SI2': 4, 'I1': 3, 'I2': 2, 'I3': 1}}


data = Data('../diamonds.csv', ['no'], 'clarity', encoding_categories)
reg = linear_model.LinearRegression()
reg_train_x = data.X_train['price'].values[:-1]
reg_train_y = data.X_train['carat'].values[:-1]
reg = reg.fit(reg_train_x.reshape(-1, 1), reg_train_y.reshape(-1, 1))

reg_test_x = data.X_test['price'].values[:]
reg_test_y = data.X_test['carat'].values[:]
prediction_y = reg.predict(reg_test_x.reshape(-1, 1))
prediction_y = prediction_y.reshape(-1, 1)
# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print('Mean squared error: ', metrics.mean_squared_error(reg_test_y, prediction_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: ', metrics.r2_score(reg_test_y, prediction_y))

plt.scatter(reg_test_x, reg_test_y,  color='black')
plt.plot(reg_test_x, prediction_y, color='red', linewidth=3)
plt.show()
