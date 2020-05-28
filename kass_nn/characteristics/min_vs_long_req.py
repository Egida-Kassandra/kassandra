from kass_nn.eif import eif
from kass_nn.util.parse_logs import LogParser
from kass_nn.util import kass_plotter as plt


class MinLong:
    def __init__(self, logpar):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 6]
        self.radius1 = 500000
        self.radius2 = 100000
        self.mesh = 6400
        self.logpar = logpar
        self.X_train = []
        self.X_test = []
        self.clf = None


if __name__ == '__main__':

    train_filename = "../train_logs/min_dir/sint_data_url_50freq_modified.txt"
    test_filename = "../test_logs/min_dir/BIG_TEST_TRANS_min_directory.txt"
    radius1 = 500000
    radius2 = 100000
    mesh = 6100
    characteristic = MinLong()

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

