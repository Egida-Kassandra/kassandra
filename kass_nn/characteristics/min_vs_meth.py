from kass_nn.eif import eif
from kass_nn.util.parse_logs import LogParser
from kass_nn.util import kass_plotter as plt


class MinMeth:
    def __init__(self):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 2]
        self.logpar = LogParser()


if __name__ == '__main__':

    train_filename = "../train_logs/access3_features_sint.log"
    test_filename = "../test_logs/min_meth/BIG_TEST_TRANS_min_meth.txt"
    radius1 = 500000
    radius2 = 300000
    mesh = 1500
    characteristic = MinMeth()

    # Loading training data
    X_train = eif.load_parsed_data(train_filename, True, characteristic.logpar, characteristic.columns, radius1, radius2)
    # Loading testing data
    X_test = eif.load_parsed_data(test_filename, False, characteristic.logpar, characteristic.columns, radius1, radius2)
    # Training model
    clf = eif.train_model(X_train)
    # Predicting model
    anomaly_scores = eif.predict(X_test, clf, X_train)
    # Plotting model
    plt.plot_model(X_train, X_test, anomaly_scores, clf, mesh)
    # Plotting with hours
    #plt.plot_model_hours(X_train, X_test, anomaly_scores, clf, 4000)

