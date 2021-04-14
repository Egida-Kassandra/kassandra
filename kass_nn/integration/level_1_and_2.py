from kass_nn.level_1 import main as level1
from kass_nn.level_2 import main as level2
from kass_nn.util.parse_logs import LogParser


def run_level_1(train_filename, test_filename, config_file, logpar):
    if logpar is None:
        logpar = preparing(train_filename)
    print("*" * 40 + " LEVEL 1 " + "*" * 40)
    return level1.run_level_1(train_filename, test_filename, config_file, logpar)


def run_level_2(train_filename, test_filename, config_file, logpar):
    if logpar is None:
        logpar = preparing(train_filename)
    print("*" * 40 + " LEVEL 2 " + "*" * 40)
    return level2.run_level_2(train_filename, test_filename, config_file, logpar)


def preparing(train_filename):
    print("*" * 40 + " PARSING TRAINING DATA " + "*" * 40)
    logpar = LogParser(train_filename)
    return logpar
    

def run_all_levels(train_filename, test_filename, config_file, logpar=None):
    
    should_run_level_2 = run_level_1(train_filename, test_filename, config_file, logpar)
    if should_run_level_2:
        run_level_2(train_filename, test_filename, config_file, logpar)
