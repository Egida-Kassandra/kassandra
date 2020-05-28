from kass_nn.eif import eif
from kass_nn.util.parse_logs import LogParser
from kass_nn.util import kass_plotter as plt


class MinDir:
    def __init__(self,logpar):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 3]
        self.radius1 = 500000
        self.radius2 = 300000
        self.mesh = 4000
        self.logpar = logpar
        self.X_train = []
        self.X_test = []
        self.clf = None


if __name__ == '__main__':

    train_filename = "../train_logs/min_dir/sint_data_url_50freq_modified.txt"
    test_filename = "../test_logs/min_dir/BIG_TEST_TRANS_min_directory.txt"
    logpar = LogParser(train_filename)
    characteristic = MinDir()

    # Loading training data
    X_train = eif.load_parsed_data(train_filename, True, logpar, characteristic)
    # Loading testing data
    X_test = eif.load_parsed_data(test_filename, False, logpar, characteristic)
    # Training model
    clf = eif.train_model(X_train)
    # Predicting model
    anomaly_scores = eif.predict(X_test, clf, X_train)
    # Plotting model
    plt.plot_model(X_train, X_test, anomaly_scores, clf, characteristic.mesh)
    # Plotting with hours
    #plt.plot_model_hours(X_train, X_test, anomaly_scores, clf, 4000)

