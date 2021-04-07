from kass_nn.integration.level_1_and_2 import run_all_levels
from pathlib import Path

if __name__ == '__main__':
    integration_f = Path("kass_nn/integration")
    train_filename = integration_f/"train_logs/main/train_main.log"
    test_filename = integration_f/"test_logs/main/test_main_0.log"
    config_f = Path("kass_nn/config")
    config_file = config_f/"config.yml"
    run_all_levels(train_filename, test_filename, config_file)