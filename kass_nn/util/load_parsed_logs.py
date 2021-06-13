from kass_nn.util import translate_to_circumference as circ
import pandas as pd
import numpy as np


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
    # if test input is a file
    #data_train = logpar.parse_file(filename, is_train) 
    # if test input by console
    data_train = logpar.public_parse_line(filename, is_train)
    try:
        return pd.DataFrame(data_train)[columns]
    except:
        return None


def load_parsed_data(filename, is_train, charac):
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
    for log in data:
        log_array = [log[0], log[1]]
        criteria = charac.get_group_criteria(log)
        if criteria not in grouped_lists:
            grouped_lists[criteria] = [log_array]
        else:
            grouped_lists[criteria].append(log_array)
    return grouped_lists