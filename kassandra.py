import socket
import sys
from kass_nn.integration.level_1_and_2 import run_all_levels, preparing
from pathlib import Path

if __name__ == '__main__':
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    server_address = ('localhost', 5000)

    sock.bind(server_address)
    print('\nKassandra is preparing herself...\n')
    integration_f = Path("kass_nn/integration")
    train_filename = integration_f/"train_logs/main/train_main.log"

    logpar = preparing(train_filename)
    config_f = Path("kass_nn/config")
    config_file = config_f/"config.yml"

    while True:
    

        print('\nKassandra is listening...')
        data, address = sock.recvfrom(4096)
        http_req = str(data).split('nginx: ')[1][:-1]
        
        run_all_levels(train_filename, http_req, config_file, logpar)