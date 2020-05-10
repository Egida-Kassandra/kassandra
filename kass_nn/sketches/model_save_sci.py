from parse_logs import LogParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import eif as iso
import time

import numpy as np

from sklearn.decomposition import PCA
from sklearn import datasets

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition

# https://stackabuse.com/scikit-learn-save-and-restore-models/


def load_data_pandas(logpar, filename, is_train, column_array):
    """
    Loads log file and returns pandas data frames
    :param filename: name of the file
    :param is_train: boolean, if the file has training or testing logs
    :param column_array: int[], columns used for training and testing
    """

    data_train = logpar.parse_file(filename, is_train)
    data_pandas = pd.DataFrame(data_train)[column_array]
    return data_pandas


def load_data_float(data_pandas):
    """
    Returns numpy array of float from pandas data frame
    :param data_pandas: pandas data frame
    """
    return data_pandas.to_numpy().astype(np.float)


def plot_data(data_train, data_test, col_X, colY, anomaly_scores, clf):
    """
    Plots 2D data set training and testing
    :param data_train: pandas data frame train
    :param data_test: pandas data frame test
    :param col_X: int, x axis
    :param col_Y: int, y axis
    :param anomaly_scores: array of anomaly scores
    """

    #xx, yy = np.meshgrid(np.linspace(np.min(data_train[col_X]), np.max(data_train[colY]), 50), np.linspace(np.min(data_train[col_X]), np.max(data_train[colY]), 50))
    xx, yy = np.meshgrid(np.linspace(-1000, 2000, 50),
                         np.linspace(-1000, 2000, 50))
    S0 = clf.compute_paths(np.c_[xx.ravel(), yy.ravel()])
    S0 = S0.reshape(xx.shape)
    ss0 = np.argsort(anomaly_scores)
    f = plt.figure(figsize=(15, 6))
    ax1 = f.add_subplot(121)
    levels = np.linspace(np.min(S0), np.max(S0), 5)
    CS = ax1.contourf(xx, yy, S0, levels, cmap=plt.cm.YlOrRd)


    # Train data
    x = np.array(data_train[col_X].tolist())
    y = np.array(data_train[colY].tolist())
    plt.scatter(x, y, s=60, c='g', edgecolor='g')
    # Test data
    x = np.array(data_test[col_X].tolist())
    y = np.array(data_test[colY].tolist())
    plt.scatter(x, y, s=20, c='b', edgecolor='b')
    plt.scatter(x[ss0[-3:]], y[ss0[-3:]], s=20, c='r')
    plt.scatter(x[ss0[:3]], y[ss0[:3]], s=20, c='k')
    plt.title('extended')
    plt.show()


#def transform_point(x, y, classX, classY):
dictionary1 = {}
dictionary2 = {}
dictionary3 = {}
dictionary4 = {}
middle_point_A = 0
middle_point_B = 0
middle_point_C = 0
def create_dict(data_list, data_class, points): # atencion gochada
    data_list = data_list.to_numpy()
    data_list_red = []
    for elem in data_list:
        if elem[1] == data_class:
            data_list_red.append(elem)
    chunk = int(len(data_list_red)/4)
    for i in range(0,chunk):
        elem = data_list_red[i]
        if elem[0] not in dictionary1:
            id_p = len(dictionary1)+1
            dictionary1[elem[0]] = id_p
            points.append([id_p, id_p])
        else:
            #dictionary[elem[0]][1] = int(dictionary[elem[0]][1]) + 1
            points.append([dictionary1[elem[0]], dictionary1[elem[0]]])
    for i in range(chunk,chunk*2):
        elem = data_list_red[i]
        if elem[0] not in dictionary2:
            id_p = len(dictionary2)+1
            dictionary2[elem[0]] = id_p
            points.append([id_p, -id_p])
        else:
            points.append([dictionary2[elem[0]], -dictionary2[elem[0]]])

    for i in range(chunk*2,chunk*3):
        elem = data_list_red[i]
        if elem[0] not in dictionary3:
            id_p = len(dictionary3)+1
            dictionary3[elem[0]] = id_p
            points.append([-id_p, -id_p])
        else:
            points.append([-dictionary3[elem[0]], -dictionary3[elem[0]]])

    for i in range(chunk*3,len(data_list_red)):
        elem = data_list_red[i]
        if elem[0] not in dictionary4:
            id_p = len(dictionary4)+1
            dictionary4[elem[0]] = id_p
            points.append([-id_p, id_p])
        else:
            points.append([-dictionary4[elem[0]], dictionary4[elem[0]]])


    """       
    for elem in data_list:
        if elem[1] == data_class :

            if elem[0] not in dictionary:
                id_p = len(dictionary)
                dictionary[elem[0]] = id_p
                points.append([id_p, id_p])
            else:
                #dictionary[elem[0]][1] = int(dictionary[elem[0]][1]) + 1
                points.append([elem[0], elem[0]])
                
    """

    return [dictionary1,dictionary2,dictionary3,dictionary4]


def insert_test_points(point, data_class, weight):
    print(point)
    new_point = []
    if point[1] == data_class:
        if point[0] <= middle_point_A:
            if point[0] not in dictionary1:
                id_p = len(dictionary1) + weight
                new_point.append(id_p)
                new_point.append(id_p)
            else:
                new_point.append(dictionary1[point[0]])
                new_point.append(dictionary1[point[0]])
        elif point[0] > middle_point_A and point[0] <= middle_point_B:
            if point[0] not in dictionary2:
                id_p = len(dictionary2) + weight
                new_point.append(id_p)
                new_point.append(-id_p)
            else:
                new_point.append(dictionary2[point[0]])
                new_point.append(dictionary2[point[0]])
        elif point[0] > middle_point_B and point[0] <= middle_point_C:
            if point[0] not in dictionary3:
                id_p = len(dictionary1) + weight
                new_point.append(-id_p)
                new_point.append(-id_p)
            else:
                new_point.append(dictionary3[point[0]])
                new_point.append(dictionary3[point[0]])
        else:
            if point[0] not in dictionary4:
                id_p = len(dictionary1) + weight
                new_point.append(-id_p)
                new_point.append(id_p)
            else:
                new_point.append(dictionary4[point[0]])
                new_point.append(dictionary4[point[0]])
    return new_point



def cross_distribution():
    # Columns used for training and testing
    columns = [0, 1]
    logpar = LogParser()

    # Loading training data
    train = load_data_pandas(logpar, '../train_logs/access3_features.log', True, columns)

    train.sort_values(by=[0], inplace=True)
    # X_train = load_data_float(data_pandas)
    points = []
    create_dict(train, 1500, points)

    data_pandas = pd.DataFrame(points)

    middle_point_A = (list(dictionary2)[0] - list(dictionary1)[-1]) / 2.0 + list(dictionary1)[-1]
    middle_point_B = (list(dictionary3)[0] - list(dictionary2)[-1]) / 2.0 + list(dictionary2)[-1]
    middle_point_C = (list(dictionary4)[0] - list(dictionary3)[-1]) / 2.0 + list(dictionary3)[-1]

    points_test = []
    test = load_data_pandas(logpar, '../test_logs/BIG_TEST_TRANS2.txt', True, columns)

    test.sort_values(by=[0], inplace=True)
    print(test)
    # print(create_dict(test, 2, points_test))
    test = test.to_numpy()
    for elem in test:
        points_test.append(insert_test_points(elem, 1500, 100))
    print(points_test)

    datatest_pandas = pd.DataFrame(points_test)
    X_train = load_data_float(data_pandas)
    X_test = load_data_float(datatest_pandas)

    clf = iso.iForest(X_train, ntrees=2000, sample_size=1500, ExtensionLevel=1)  # 2000, 20000, ext 1 ###### cuidaooo

    anomaly_scores = clf.compute_paths(X_test)
    print(anomaly_scores)
    plot_data(data_pandas, datatest_pandas, 0, 1, anomaly_scores, clf)



if __name__ == '__main__':
    logpar = LogParser()
    # Columns used for training and testing
    columns = [0,5]
    
    # Loading training data
    data_pandas = load_data_pandas(logpar, '../train_logs/access3_features.log', True, columns)
    X_train = load_data_float(data_pandas)
    np.random.seed(123)
    pca = decomposition.PCA(n_components=2)
    #pca.fit(X_train)
    #X_train = pca.transform(X_train)
    
    # Loading testing data
    datatest_pandas = load_data_pandas(logpar, '../test_logs/BIG_TEST_TRANS.txt', True, columns)
    X_test = load_data_float(datatest_pandas)
    #pca.fit(X_test)
    #X_test = pca.transform(X_test)
    print(X_test)
    # Train block
    print("TRAIN")
    start = time.time()

    clf = iso.iForest(X_train, ntrees=2000, sample_size=20000, ExtensionLevel=1) # 2000, 20000, ext 1 ###### cuidaooo

    end = time.time()
    print(end - start)
    
    # Predict block
    print("PREDICT")
    start = time.time()
    anomaly_scores = clf.compute_paths(X_test)

    end = time.time()
    print(end - start)

    # Debug block
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    print(indices_with_preds)
    print((anomaly_scores))

    # Plot block
    data_pandas = pd.DataFrame(X_train)
    datatest_pandas = pd.DataFrame(X_test)
    plot_data(data_pandas, datatest_pandas, 0, 1, anomaly_scores, clf)

