from kass_nn.level_2.kass_main.train_predict import TrainPredict
import sys
from kass_nn.util.parse_logs import LogParser

def run_level_2(train_filename, test_filename, config_file, logpar):
    trainpredict = TrainPredict(train_filename, config_file, logpar)
    trainpredict.train_all()
    trainpredict.predict_all(test_filename)


if __name__ == '__main__':
    train_filename = "../level_2/train_logs/main/train_main.log"
    test_filename = "../level_2/test_logs/main/test_main_5.log"
    config_file = "../config/config.yml"
    logpar = LogParser(train_filename)
    print(sys.argv)
    trainpredict = TrainPredict(train_filename, config_file)
    trainpredict.train_all()
    trainpredict.predict_all(test_filename)


