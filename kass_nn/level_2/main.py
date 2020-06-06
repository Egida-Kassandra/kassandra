from pathlib import Path
from kass_nn.level_2.kass_main.train_predict import TrainPredict
import sys
from kass_nn.util.parse_logs import LogParser

def run_level_2(train_filename, test_filename, config_file, logpar):
    trainpredict = TrainPredict(train_filename, config_file, logpar)
    trainpredict.train_all()
    trainpredict.predict_all(test_filename)


def main(test_file):
    kassnn_f = Path("kass_nn")
    train_filename = kassnn_f / "level_2/train_logs/main/train_main.log"
    test_filename = kassnn_f / str("level_2/test_logs/main/" + test_file)
    config_file = kassnn_f / "config/config.yml"
    logpar = LogParser(train_filename)
    trainpredict = TrainPredict(train_filename, config_file, logpar)
    trainpredict.train_all()
    trainpredict.predict_all(test_filename)


