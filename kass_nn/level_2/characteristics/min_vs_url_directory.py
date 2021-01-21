from pathlib import Path

from kass_nn.level_2.eif_module import eif
from kass_nn.level_2.danger_labeling.dangerousness import get_dangerousness_int
from kass_nn.util.parse_logs import LogParser
from kass_nn.util import kass_plotter as plt
import yaml
from kass_nn.util import load_parsed_logs as lp


class MinDir:
    def __init__(self,logpar,config_file):
        """Constructor"""
        self.clf = None
        self.X_train = None
        self.columns = [1, 3]
        self.radius1 = 500000
        self.radius2 = 300000
        self.mesh = 4000
        self.ntrees = 2000
        self.sample_size = 10000
        self.ntrees = None
        self.sample_size = None
        self.logpar = logpar
        self.X_train = []
        self.X_test = []
        self.clf = None
        self.n_threads = 1
        self.read_params(config_file)

    def read_params(self,config_file):
        yaml_document = open(config_file)
        params = yaml.safe_load(yaml_document)
        self.ntrees = params["ntrees_min_dir"]
        self.sample_size = params["sample_size_min_dir"]
        self.mesh = params["mesh_min_dir"]
        self.n_threads = params["n_threads"]


def main(test_file):
    kassnn_f = Path("kass_nn")
    train_filename = kassnn_f / "level_2/train_logs/min_dir/train_min_dir.log"
    test_filename = kassnn_f / str("level_2/test_logs/min_dir/" + test_file)
    config_file = kassnn_f / "config/config.yml"
    logpar = LogParser(train_filename)
    characteristic = MinDir(logpar, config_file)

    # Loading training data
    X_train = lp.load_parsed_data(train_filename, True, characteristic)
    # Loading testing data
    X_test = lp.load_parsed_data(test_filename, False, characteristic)
    # Training model
    clf = eif.train_model(X_train, characteristic)
    # Predicting model
    anomaly_scores = eif.predict_wo_train(X_test, clf, characteristic.n_threads)
    i = 0
    for anom in anomaly_scores:
        print("TEST {}\n\tFull anomaly value: {}\n\tDangerousness in range [0-5]: {}".format(i, anom,
                                                                                             get_dangerousness_int(
                                                                                                 anom)))
        i += 1
    # Plotting model
    fig = plt.open_plot()
    plt.plot_model(fig, X_train, X_test, anomaly_scores, clf,
                   characteristic.mesh, [1, 1, 1], "Min vs Dir", characteristic.n_threads)
    plt.close_plot()
    # Plotting with hours
    # plt.plot_model_hours(X_train, X_test, anomaly_scores, clf, 4000)


