from kass_nn.eif import eif
from kass_nn.util.parse_logs import LogParser
from kass_nn.util import kass_plotter as plt
import numpy as np


class IPMinURL:
    def __init__(self, logpar):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 2, 0]
        self.radius1 = 500000
        self.radius2 = 300000
        self.mesh = 2500
        self.logpar = logpar
        self.X_train = []
        self.X_test = []
        self.clf = None ##
        self.clfs_by_ip = {}


    def split_by_ip(self):
        for req in self.X_train:
            pass

    def get_group_criteria(self, log):
        """
        Returns the IP ID, which will be the key for the grouped list dictionary
        """
        return log[2]


if __name__ == '__main__':

    train_filename = "../train_logs/access3_features_sint.log"
    test_filename = "../test_logs/foreach_ip_min_vs_url/BIG_TEST_TRANS_foreach_ip_min_vs_url.txt"
    logpar = LogParser(train_filename)
    characteristic = IPMinURL(logpar)

    # Loading training data
    #X_train = eif.load_parsed_data(train_filename, True, characteristic)
    X_train = eif.load_parsed_data(train_filename, True, characteristic)

    # Loading testing data
    X_test = eif.load_parsed_data(test_filename, False, characteristic)

    print(X_test[:1])
    # Training model
    if isinstance(X_train, dict):
        for key in X_train:
            characteristic.clfs_by_ip[key] = eif.train_model(X_train[key])
    else:
        clf = eif.train_model(X_train)
    # Predicting model
    print("start pred")
    print(X_test[:1])
    #X_test = X_test.values.tolist()
    ip = characteristic.get_group_criteria(X_test[0])

    print(X_test)
    if ip in X_train:
        anomaly_scores = eif.predict_wo_train(X_test, characteristic.clfs_by_ip[ip], characteristic.columns)
    print(anomaly_scores)
    #anomaly_scores = eif.predict_w_train(X_test, clf, X_train)
    # Plotting model
    fig = plt.open_plot()
    plt.plot_model(fig, X_train[ip], X_test, anomaly_scores, characteristic.clfs_by_ip[ip],
                   characteristic.mesh, [2, 2, 1], "Min vs URL by IP")
    plt.close_plot()
    # Plotting with hours
    #plt.plot_model_hours(X_train, X_test, anomaly_scores, clf, 4000)

