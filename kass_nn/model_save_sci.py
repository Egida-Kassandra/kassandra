from kass_nn.parse_logs import LogParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import eif as iso

# https://stackabuse.com/scikit-learn-save-and-restore-models/


def load_data_pandas(filename, is_train, column_array):
    """
    Load
    """
    logpar = LogParser()
    data_train = logpar.parse_file(filename, is_train)
    data_pandas = pd.DataFrame(data_train)[column_array]
    return data_pandas


def load_data_float(data_pandas):
    return data_pandas.to_numpy().astype(np.float)


def plot_data(data_train, data_test, col_X, colY):
    ss0 = np.argsort(anomaly_scores)
    f = plt.figure(figsize=(15, 6))
    # Train data
    x = np.array(data_train[col_X].tolist())
    y = np.array(data_train[colY].tolist())
    plt.scatter(x, y, s=60, c='g', edgecolor='g')
    # Test data
    x = np.array(data_test[col_X].tolist())
    y = np.array(data_test[colY].tolist())
    plt.scatter(x, y, s=20, c='b', edgecolor='b')
    plt.scatter(x[ss0[-5:]], y[ss0[-5:]], s=20, c='r')
    plt.scatter(x[ss0[:1]], y[ss0[:1]], s=20, c='k')
    plt.title('extended')
    plt.show()


if __name__ == '__main__':

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
    clf = iso.iForest(X_train, ntrees=1500, sample_size=15000, ExtensionLevel=1) # 2000, 20000, ext 1 ###### cuidaooo
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
    plot_data(data_pandas, datatest_pandas, 3, 4)
