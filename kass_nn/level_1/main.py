from kass_nn.level_1.danger_labeling import dangerousness
from kass_nn.util.parse_logs import LogParser


def run_level_1(train_filename, test_filename, config_file, logpar):
    X_train = logpar.parsed_train_data
    X_test = logpar.get_string_variables(test_filename)
    run_level_2 = False
    test = X_test[0]
    presence_list_single = dangerousness.is_in_training_single(test, logpar)
    presence_list_combined = dangerousness.is_in_training_combined(test, logpar)
    print("="*80)
    print("Dangerousness in range [0-100]: {}".format(
        dangerousness.get_danger_value(presence_list_single, presence_list_combined, config_file)))
    return False not in presence_list_single


if __name__ == '__main__':
    train_filename = "../level_1/train_logs/main/train_main.log"
    test_filename = "../level_1/test_logs/main/test_main_0.log"
    config_file = "../config/config.yml"
    logpar = LogParser(train_filename)
    X_train = logpar.parsed_train_data
    X_test = logpar.get_string_variables(test_filename)
    i = 0
    for test in X_test:
        print("TEST {}".format(i))
        presence_list_single = dangerousness.is_in_training_single(test, logpar)
        presence_list_combined = dangerousness.is_in_training_combined(test, logpar)
        print("\tDangerousness in range [0-100]: {}".format(dangerousness.get_danger_value(presence_list_single, presence_list_combined, config_file)))
        i += 1
