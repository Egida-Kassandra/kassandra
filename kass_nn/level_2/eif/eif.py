import numpy as np
import eif as iso
import pandas as pd
from kass_nn.util import translate_to_circumference as circ


def train_model(X_train, characteristic):
    print("\tTRAINING")
    X_train = pd.DataFrame(X_train)
    X_train = load_data_float(X_train)
    # Train block
    train_len = len(X_train)
    if train_len > 1000:
        clf = iso.iForest(X_train, ntrees=characteristic.ntrees, sample_size=characteristic.sample_size, ExtensionLevel=1)  # 2000, 15000, ext 1 ###### cuidaooo || 1000 2000
    else:
        clf = iso.iForest(X_train, ntrees=5000, sample_size=train_len, ExtensionLevel=1)
    return clf

def predict_w_train(X_test, clf, X_train):
    print("\tPREDICTING")
    X_test = np.array(X_test).astype(np.float)
    # Predict block
    anomaly_scores = clf.compute_paths(X_test)
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    print(indices_with_preds)
    print(np.sort(anomaly_scores))
    return anomaly_scores


def predict_wo_train(X_test, clf):
    print("\tPREDICTING")
    # Predict block
    X_test = np.array(X_test).astype(np.float)
    anomaly_scores = [None]
    try:
        anomaly_scores = clf.compute_paths(X_test)
    finally:
        return anomaly_scores


def predict_plot_hours(X_test, clf, X_train):
    print("\tPREDICTING")
    # Predict block
    anomaly_scores = clf.compute_paths(X_test)
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    return anomaly_scores


def load_data_float(data_pandas):
    """
    Returns numpy array of float from pandas data frame
    :param data_pandas: pandas data frame
    """
    return data_pandas.to_numpy().astype(np.float)


def load_data_pandas(filename, is_train, logpar, columns):
    """
    Loads log file and returns pandas data frames
    :param filename: name of the file
    :param is_train: boolean, if the file has training or testing logs
    """
    if is_train:
        return pd.DataFrame(logpar.parsed_train_data)[columns]
    data_train = logpar.parse_file(filename, is_train)
    return pd.DataFrame(data_train)[columns]


def load_parsed_data(filename, is_train, charac):
    print("\tLOADING DATA")
    train = load_data_pandas(filename, is_train, charac.logpar, charac.columns)
    if len(charac.columns) == 2: # Two columns
        train = train.drop(train[(train[charac.columns[0]] < 0) | (train[charac.columns[1]] < 0)].index)
    elif len(charac.columns) == 3: # Three columns
        train = train.drop(train[(train[charac.columns[0]] < 0) | (train[charac.columns[1]] < 0) | (train[charac.columns[2]] < 0)].index)
    if is_train:
        if len(charac.columns) == 2:
            X_train = circ.parse_sc_to_scp(train, charac)
        else:
            X_train = group_by(circ.parse_sc_to_scp(train, charac), charac)
    else:
        X_train = circ.parse_sc_to_scp(train, charac)

    return X_train


def group_by(data, charac):
    grouped_lists = {}
    print("FROM GROUP BY")
    for log in data:
        log_array = [log[0], log[1]]
        criteria = charac.get_group_criteria(log)
        if criteria not in grouped_lists:
            grouped_lists[criteria] = [log_array]
        else:
            grouped_lists[criteria].append(log_array)
    return grouped_lists

