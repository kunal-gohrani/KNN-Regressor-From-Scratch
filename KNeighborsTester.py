import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from KNearestRegressor import KNearestRegressor
from math import *
import matplotlib.pyplot as plt
import time
import concurrent.futures

df = pd.read_csv('Train.csv')
# missing values in Item_weight and Outlet_size needs to be imputed
mean = df['Item_Weight'].mean()  # imputing item_weight with mean
df['Item_Weight'].fillna(mean, inplace=True)

mode = df['Outlet_Size'].mode()  # imputing outlet size with mode
df['Outlet_Size'].fillna(mode[0], inplace=True)

df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
df = pd.get_dummies(df)
train, test = train_test_split(df, test_size=0.3, random_state=4)

x_train = train.drop('Item_Outlet_Sales', axis=1).iloc[:, :].values
y_train = train['Item_Outlet_Sales'].iloc[:].values

x_test = test.drop('Item_Outlet_Sales', axis=1).iloc[:, :].values
y_test = test['Item_Outlet_Sales'].iloc[:].values

# Scaling the X_train and X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


def batch_input():
    k = int(input('Enter K value: '))
    knn = KNearestRegressor(k=k, X_train=X_train, y_train=y_train)
    knn.fit()
    print('In this method, we will use X_test as a batch input.')
    answer = knn.predict(X_test).tolist()
    error = sqrt(mean_squared_error(answer, y_test))  # Calculating RMSE
    # print('R2:',r2_score(answer,y_test))
    print('Error:', error)


# The below method uses single core single thread to find the optimal k value
# def find_k_single_thread():
#     ks = []
#     errors = []
#     for k in range(2, 20):
#         knn = KNearestRegressor(k=k, X_train=X_train, y_train=y_train)
#         knn.fit()
#         print('In this method, we will use X_test as a batch input.')
#         answer = knn.predict(X_test).tolist()
#         error = sqrt(mean_squared_error(answer, y_test))  # Calculating RMSE
#         result.append({'k': k, 'E': error})
#         print('R2:', r2_score(answer, y_test))
#         ks.append(k)
#         errors.append(error)
#         print('K:', k, '\nError:', error)
#     plt.plot(ks, errors)
#     plt.show()


def find_k(k):
    knn = KNearestRegressor(k=k, X_train=X_train, y_train=y_train)
    knn.fit()
    print('k:', k)
    answer = knn.predict(X_test).tolist()
    error = sqrt(mean_squared_error(answer, y_test))  # Calculating RMSE
    result.append({'k': k, 'E': error})
    return tuple((k, error))


def find():
    results = []
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # results = [executor.submit(find_k_multi,k) for k in range(2,21)]
        temp = [x for x in range(2, 14)]
        results = executor.map(find_k, temp)
    for r in results:
        print(r)
    finish = time.perf_counter()
    print('Finished in ', round(finish - start, 2), 'seconds')
    with open('Results.pkl', 'wb') as f:
        pickle.dump(list(results), f)


choice = int(
    input('1. Choose 1 for Batch input.\n2. Press 2 to find the optimal K value (between 2 and 13).\nChoice: '))
if choice == 1:
    batch_input()
elif choice == 2:
    find()
