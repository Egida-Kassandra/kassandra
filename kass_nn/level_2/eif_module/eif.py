import numpy as np
import eif as iso
import pandas as pd
from kass_nn.util import load_parsed_logs as lp


def train_model(X_train, characteristic):
    X_train = pd.DataFrame(X_train)
    X_train = lp.load_data_float(X_train)
    # Train block
    train_len = len(X_train)
    if train_len > 1000:
        clf = iso.iForest(X_train, ntrees=characteristic.ntrees, sample_size=characteristic.sample_size, ExtensionLevel=1)
    else:
        clf = iso.iForest(X_train, ntrees=5000, sample_size=train_len, ExtensionLevel=1)
    return clf

def predict_w_train(X_test, clf, X_train, n_threads):
    X_test = np.array(X_test).astype(np.float)
    # Predict block
    anomaly_scores = clf.compute_paths(X_test, n_threads)
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    print(indices_with_preds)
    print(np.sort(anomaly_scores))
    return anomaly_scores


def predict_wo_train(X_test, clf, n_threads):
    # Predict block
    X_test = np.array(X_test).astype(np.float)
    anomaly_scores = [None]
    try:
        anomaly_scores = clf.compute_paths(X_test, n_threads)
    finally:
        return anomaly_scores


def predict_plot_hours(X_test, clf, X_train, n_threads):
    # Predict block
    anomaly_scores = clf.compute_paths(X_test, n_threads)
    anomaly_scores_sorted = np.argsort(anomaly_scores)
    indices_with_preds = anomaly_scores_sorted[-int(np.ceil(0.9 * X_train.shape[0])):]
    return anomaly_scores






