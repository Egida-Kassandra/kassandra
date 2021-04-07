import socket
import sys

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
server_address = ('localhost', 5000)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

while True:
    print('\nwaiting to receive message')
    integration_f = Path("kass_nn/integration")
    train_filename = integration_f/"train_logs/main/train_main.log"
    data, address = sock.recvfrom(4096)

    http_req = str(data).split('nginx: ')[1][:-1]
    config_f = Path("kass_nn/config")
    config_file = config_f/"config.yml"
    run_all_levels(train_filename, http_req, config_file)