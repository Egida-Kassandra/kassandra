import integration as integr

if __name__ == '__main__':
    train_filename = "../integration/train_logs/main/train_main.log"
    test_filename = "../integration/test_logs/main/test_main_0.log"
    config_file = "./config/config.yml"
    integr.run_all_levels(train_filename, test_filename, config_file)