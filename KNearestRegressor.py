import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class KNearestRegressor:
    def __init__(self, k, X_train, y_train):
        if not (isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray)):
            raise Exception('Wrong datatype, please pass numpy arrays')
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.columns_of_train = X_train.shape[1]
        # print('X_train: ',X_train.shape)
        # print('Y_train:',len(y_train))
    def fit(self):
        print('Training completed')

    def predict(self, testing_data):
        # Checking if the input is a numpy array with correct dimensions
        # print(testing_data.shape)
        if len(testing_data.shape) > 1:
            columns_of_test = testing_data.shape[1]
        else:
            raise Exception('Please pass in a 2d array with same number of dimensions as in x_train')
        if columns_of_test != self.columns_of_train:
            raise Exception(
                'X_train column: {}\nTesting data column: {}'.format(self.X_train.shape[1], testing_data.shape[1]))

        distance = dict()  # stores x_train_index:distance_from_testing_data
        index_position = 0  # Index of X_train (this is used to get the y value of the testing_data from y_train
        predictions = np.array([], dtype='int')

        # Computing the distance of each testing data row with all other training data rows
        for i in testing_data:
            for j in self.X_train:
                distance[index_position] = np.sqrt(np.sum((i - j) ** 2))
                index_position = index_position + 1
            distance = sorted(distance.items(), key=lambda x: (x[1], x[0]))
            # classification = np.append(classification, [[self.classify(i, distance[0:self.k])]])
            prediction_of_single = self.averageOut(distance[0:self.k])
            predictions = np.append(predictions, prediction_of_single)
            # print('Prediction:',predictions)
            distance = dict()
            index_position = 0
        return predictions

    def averageOut(self, distance):
        value = np.array([], dtype=int)
        for i in distance:
            index = i[0]
            temp = self.y_train[index]
            value = np.append(value, temp)
        # print('Value:',value)
        return np.average(value)
