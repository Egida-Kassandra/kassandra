from kass_nn.parse_logs import LogParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import eif as iso

# https://stackabuse.com/scikit-learn-save-and-restore-models/


def load_data_pandas(filename, is_train, column_array):
    """
    Loads log file and returns pandas data frames
    :param filename: name of the file
    :param is_train: boolean, if the file has training or testing logs
    :param column_array: int[], columns used for training and testing
    """
    logpar = LogParser()
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
    xx, yy = np.meshgrid(np.linspace(-5, 15, 50),
                         np.linspace(-5, 15, 50))
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

    data_list_red = []
    for elem in data_list:
        if elem[1] == data_class:
            data_list_red.append(elem)
    print(len(data_list_red))
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


def insert_test_points(point, data_class):
    new_point = []
    if point[0] <= middle_point_A:
        if point[0] not in dictionary1:
            id_p = len(dictionary1) + 1
            new_point.append(id_p)
            new_point.append(id_p)
    elif point[0] > middle_point_A and point[0] <= middle_point_B:
        if point[0] not in dictionary2:
            id_p = len(dictionary2) + 1
            new_point.append(id_p)
            new_point.append(id_p)
    elif point[0] > middle_point_B and point[0] <= middle_point_C:
        if point[0] not in dictionary1:
            id_p = len(dictionary1) + 1
            new_point.append(id_p)
            new_point.append(id_p)
    else:
        if point[0] not in dictionary1:
            id_p = len(dictionary1) + 1
            new_point.append(id_p)
            new_point.append(id_p)
    return new_point




if __name__ == '__main__':

    train = [[1,2],
             [2,2],
             [3,4],
             [4,4],
             [5,2],
             [6,4],
             [7,2],
             [8,4],
             [9,4],
             [10,4]
             ]

    points = []
    print(create_dict(train, 2, points))
    print(points)
    data_pandas = pd.DataFrame(points)

    middle_point_A = (list(dictionary2)[0] - list(dictionary1)[-1])/2.0 + list(dictionary1)[-1]
    middle_point_B = (list(dictionary3)[0] - list(dictionary2)[-1])/2.0 + list(dictionary2)[-1]
    middle_point_C = (list(dictionary4)[0] - list(dictionary3)[-1])/2.0 + list(dictionary3)[-1]
    print(middle_point_A, middle_point_B, middle_point_C)

    test = [[1, 4],
             [2, 2],
             [3, 4],
             [4, 2],
             [5, 6],
             [6, 4],
             [7, 4],
             [8, 2],
             [9, 2],
             [10, 2]]

    points_test = []
    print(create_dict(test, 2, points_test))
    print(points_test)
    
    datatest_pandas = pd.DataFrame(points_test)
    X_train = load_data_float(data_pandas)
    X_test = load_data_float(datatest_pandas)



    clf = iso.iForest(X_train, ntrees=1000, sample_size=4, ExtensionLevel=1)  # 2000, 20000, ext 1 ###### cuidaooo

    anomaly_scores = clf.compute_paths(X_test)
    plot_data(data_pandas, datatest_pandas, 0, 1, anomaly_scores, clf)

    """
    # Columns used for training and testing
    columns = [3, 4]
    
    # Loading training data
    data_pandas = load_data_pandas('train_logs/access3_features.log', True, columns)
    X_train = load_data_float(data_pandas)
    
    # Loading testing data
    datatest_pandas = load_data_pandas('test_logs/BIG_TEST_TRANS.txt', True, columns)
    X_test = load_data_float(datatest_pandas)

    # Train block
    print("TRAIN")
    start = time.time()
    clf = iso.iForest(X_train, ntrees=2000, sample_size=15000, ExtensionLevel=1) # 2000, 20000, ext 1 ###### cuidaooo
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
    print(np.sort(anomaly_scores))

    # Plot block
    plot_data(data_pandas, datatest_pandas, 3, 4, anomaly_scores, clf)
    """
