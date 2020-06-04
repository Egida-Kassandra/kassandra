import yaml
from kass_nn.level_2.eif_module import eif
from kass_nn.level_2.danger_labeling.dangerousness import get_dangerousness_int
from kass_nn.util.parse_logs import LogParser
from kass_nn.util import kass_plotter as plt
from kass_nn.util import load_parsed_logs as lp


class MinLong:
    def __init__(self, logpar, config_file):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 6]
        self.radius1 = 500000
        self.radius2 = 100000
        self.mesh = None
        self.ntrees = None
        self.sample_size = None
        self.logpar = logpar
        self.X_train = []
        self.X_test = []
        self.clf = None
        self.read_params(config_file)

    def read_params(self, config_file):
        yaml_document = open(config_file)
        params = yaml.safe_load(yaml_document)
        self.ntrees = params["ntrees_min_long"]
        self.sample_size = params["sample_size_min_long"]
        self.mesh = params["mesh_min_long"]


if __name__ == '__main__':

    train_filename = "../level_2/train_logs/min_long/train_min_long.log"
    test_filename = "../level_2/test_logs/min_long/BIG_TEST_TRANS_min_long.txt"
    config_file = "../../config/config.yml"
    logpar = LogParser(train_filename)
    characteristic = MinLong(logpar, config_file)

    # Loading training data
    X_train = lp.load_parsed_data(train_filename, True, characteristic)
    # Loading testing data
    X_test = lp.load_parsed_data(test_filename, False, characteristic)
    # Training model
    clf = eif.train_model(X_train, characteristic)
    # Predicting model
    anomaly_scores = eif.predict_wo_train(X_test, clf)
    i = 0
    for anom in anomaly_scores:
        print("TEST {}\n\tFull anomaly value: {}\n\tDangerousness in range [0-5]: {}".format(i, anom,
                                                                                             get_dangerousness_int(
                                                                                                 anom)))
        i += 1
    # Plotting model
    fig = plt.open_plot()
    plt.plot_model(fig, X_train, X_test, anomaly_scores, clf,
                   characteristic.mesh, [1, 1, 1], "Min vs Request length")
    plt.close_plot()

