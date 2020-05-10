import numpy as np
import eif as iso
import pandas as pd
import kass_nn.util.translate_to_circumference as circ
#import kass_nn.util.kass_plotter as plt

from kass_nn.util import kass_plotter as plt
from kass_nn.util import translate_to_circumference as circ


def train_model(X_train):
    print("TRAINING")
    # Train block
    clf = iso.iForest(X_train, ntrees=2000, sample_size=1000, ExtensionLevel=1)  # 2000, 15000, ext 1 ###### cuidaooo

    # Save block
    """
    pickle.dumps(self.clf)
    with open('min_vs_meth.kass', 'wb') as model_file:
        pickle.dump(self.clf, model_file)
        """
    return clf

def predict(X_test, clf, X_train):
    print("PREDICTING")
    # Predict block
    anomaly_scores = clf.compute_paths(X_test)
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    print(indices_with_preds)
    print(np.sort(anomaly_scores))
    return anomaly_scores


def predict_plot_hours(X_test, clf, X_train):
    print("PREDICTING")
    # Predict block
    anomaly_scores = clf.compute_paths(X_test)
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    print(indices_with_preds)
    print(np.sort(anomaly_scores))
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

    data_train = logpar.parse_file(filename, is_train)
    return pd.DataFrame(data_train)[columns]


def load_parsed_data(filename, is_train, logpar, columns, radius1, radius2):
    train = load_data_pandas(filename, is_train, logpar, columns)
    data_pandas = pd.DataFrame(circ.parse_sc_to_scp(train, columns, radius1, radius2))
    X_train = load_data_float(data_pandas)
    return X_train