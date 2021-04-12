from kass_nn.level_1.danger_labeling import dangerousness
from kass_nn.util.parse_logs import LogParser
from pathlib import Path


def run_level_1(train_filename, test_filename, config_file, logpar):
    # if test input is a file
    #X_test = logpar.get_string_variables(test_filename)
    #test = X_test[0]
    # if test input by console
    X_test = logpar.get_string_log_list(test_filename)
    test = X_test

    presence_list_single = dangerousness.is_in_training_single(test, logpar)
    presence_list_combined = dangerousness.is_in_training_combined(test, logpar)
    print("="*80)
    print("Dangerousness in range [0-100]: {}".format(
        dangerousness.get_danger_value(presence_list_single, presence_list_combined, config_file)))
    return False not in presence_list_single


def main(test_file):
    kassnn_f = Path("kass_nn")
    train_filename = kassnn_f/"level_1/train_logs/main/train_main.log"
    test_filename = kassnn_f / str("level_1/test_logs/main/" + test_file)
    config_file = kassnn_f/"config/config.yml"
    logpar = LogParser(train_filename)
    X_test = logpar.get_string_variables(test_filename)
    i = 0
    for test in X_test:
        print("TEST {}".format(i))
        presence_list_single = dangerousness.is_in_training_single(test, logpar)
        presence_list_combined = dangerousness.is_in_training_combined(test, logpar)
        print("\tDangerousness in range [0-100]: {}".format(
            dangerousness.get_danger_value(presence_list_single, presence_list_combined, config_file)))
        i += 1

