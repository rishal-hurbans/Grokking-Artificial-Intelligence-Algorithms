import pandas as pd
from sklearn.model_selection import train_test_split


# A data class to read and prepare data into training and testing data
class Data:
    def __init__(self, data_file_name, excluded_features, label, encoded_categories):
        data_file = pd.read_csv(data_file_name)
        data_file.replace(encoded_categories, inplace=True)
        X = data_file.drop(columns=excluded_features)
        y = data_file[label]
        X = X.drop(columns=label)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.5)

    # def enumerate_categories(self, encoded_categories):
    #     self.X_train.replace(encoded_categories, inplace=True)
    #     self.X_test.replace(encoded_categories, inplace=True)
    #     self.y_train.replace(encoded_categories, inplace=True)
    #     self.y_test.replace(encoded_categories, inplace=True)
